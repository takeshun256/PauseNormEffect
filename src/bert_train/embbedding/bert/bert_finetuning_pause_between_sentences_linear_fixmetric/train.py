import itertools
import json
import os
import pickle
import random
import re
import sys
import unicodedata
from pathlib import Path
from pprint import pprint

sys.path.append("/home/takeshun256/PausePrediction")
import pickle as pkl

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertJapaneseTokenizer,
    BertModel,
)

from config import DATA_TAKESHUN256_DIR

# ================================================================================
# 学習時の設定
# ================================================================================


# 乱数シードの固定
def seed_everything(seed=42):
    """Function for consistency of experiment.

    input
        seed: random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ================================================================================
# データのロード
# ================================================================================


def data_load(pause_time_threshold_ms=100, preprocess_type="none", num_labels=1):
    """データのロードを行う関数."""
    # conf
    corpus_name = "jmac"
    exp_name = "03_VAD_Adjusted"
    exp_dir = Path(DATA_TAKESHUN256_DIR) / corpus_name / exp_name
    output_dir = exp_dir / "data_bert"
    #!!!!!!!!!!!!!! 埋め込み学習は正規化処理したものを使用しないため、preprocess_typeはnoneにする
    preprocess_type = "none"
    #!!!!!!!!!!!!!!!!
    df_dir = output_dir / f"{pause_time_threshold_ms}ms" / f"{preprocess_type}"
    try:
        train_df = pkl.load(open(df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_train.pkl", "rb"))
    except FileNotFoundError:
        print(
            "train_df is not found. Expected path is ",
            df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_train.pkl",
        )
        sys.exit(1)
    try:
        val_df = pkl.load(open(df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_val.pkl", "rb"))
    except FileNotFoundError:
        print(
            "val_df is not found. Expected path is ",
            df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_val.pkl",
        )
        sys.exit(1)
    try:
        test_df = pkl.load(open(df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_test.pkl", "rb"))
    except FileNotFoundError:
        print(
            "test_df is not found. Expected path is ",
            df_dir / f"bert_traindata_BetweenSentences_{num_labels}label_test.pkl",
        )
        sys.exit(1)

    #!!!!!!!!!!!!!!!!!!!!meansとvars_がそれぞれ0, 1であることを確認
    assert np.all(train_df["means"] == 0) and np.all(
        train_df["vars"] == 1
    ), f"train_df: {train_df['means']}, {train_df['vars']}"
    assert np.all(val_df["means"] == 0) and np.all(val_df["vars"] == 1), f"val_df: {val_df['means']}, {val_df['vars']}"
    assert np.all(test_df["means"] == 0) and np.all(
        test_df["vars"] == 1
    ), f"test_df: {test_df['means']}, {test_df['vars']}"

    return train_df, val_df, test_df


# ================================================================================
# データセットの作成
# ================================================================================


def create_dataset(
    train_df,
    val_df,
    test_df,
    tokenizer,
    max_length_train,
    max_length_val,
    max_length_test,
    batch_size,
    num_labels=1,  # 1=回帰, 2=2値分類, 3=多値分類
    **kwargs,
):
    """データセットを作成する関数."""
    # 埋め込み用のIDデータの取得
    preprocess_type = kwargs.get("preprocess_type", None)
    if preprocess_type is None:
        raise ValueError("preprocess_type must be not None")
    train_id_groups = train_df[f"id_{preprocess_type}"].tolist()
    val_id_groups = val_df[f"id_{preprocess_type}"].tolist()
    test_id_groups = test_df[f"id_{preprocess_type}"].tolist()

    # Extracting texts and labels for each dataset
    train_texts, train_labels = train_df["texts"].tolist(), train_df["labels"].tolist()
    train_means, train_vars = train_df["means"].tolist(), train_df["vars"].tolist()
    val_texts, val_labels = val_df["texts"].tolist(), val_df["labels"].tolist()
    val_means, val_vars = val_df["means"].tolist(), val_df["vars"].tolist()
    test_texts, test_labels = test_df["texts"].tolist(), test_df["labels"].tolist()
    test_means, test_vars = test_df["means"].tolist(), test_df["vars"].tolist()

    # データセットの作成
    dataset_train = CreateDataset(
        train_texts, train_labels, train_means, train_vars, tokenizer, max_length_train, num_labels, train_id_groups
    )
    dataset_val = CreateDataset(
        val_texts, val_labels, val_means, val_vars, tokenizer, max_length_val, num_labels, val_id_groups
    )
    dataset_test = CreateDataset(
        test_texts, test_labels, test_means, test_vars, tokenizer, max_length_test, num_labels, test_id_groups
    )

    # データローダーの定義時にカスタムコラテ関数を使用
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return (
        dataloader_train,
        dataloader_val,
        dataloader_test,
    )


class CreateDataset(Dataset):
    """データセットの作成."""

    def __init__(self, texts, labels, means, vars, tokenizer, max_length, num_labels=1, id_groups=None):
        self.texts = texts
        self.labels = labels
        self.means = means
        self.vars = vars
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels  # 1=回帰, 2=2値分類, 3=多値分類
        self.id_groups = id_groups

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        mean = self.means[index]
        var = self.vars[index]
        id_group = self.id_groups[index]

        # SEPハ認識してくれないらしいので、[SEP]を挿入
        if " [SEP] " not in text:
            raise ValueError("text must contain [SEP]")
        text1 = text.split(" [SEP] ")[0]
        text_pair = text.split(" [SEP] ")[1]

        encoded_dict = self.tokenizer.encode_plus(
            text=text1,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",  # pad_to_max_lengthをpaddingに変更
            truncation=True,  # truncationを明示的に指定
            return_attention_mask=True,
            return_tensors="pt",
        )

        # # トークナイザーのエンコード部分での修正
        # encoded_dict = self.tokenizer.encode_plus(
        #     text=text,
        #     add_special_tokens=True,
        #     max_length=self.max_length,
        #     padding="max_length",  # pad_to_max_lengthをpaddingに変更
        #     truncation=True,  # truncationを明示的に指定
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )

        # テンソルのコピーに関する修正
        input_ids = encoded_dict["input_ids"].clone().detach().to(torch.long)
        attention_mask = encoded_dict["attention_mask"].clone().detach().to(torch.long)
        label = torch.tensor(label).to(torch.float)
        mean = torch.tensor(mean).to(torch.float)
        var = torch.tensor(var).to(torch.float)
        id_group = torch.tensor(id_group).to(torch.long)

        if self.num_labels == 1:
            # 回帰タスク
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
                "means": mean,
                "vars": var,
                "group_ids": id_group,
            }
        else:
            # 分類タスク
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "means": mean,
                "vars": var,
                "group_ids": id_group,
            }


# ================================================================================
# 損失関数の定義
# ================================================================================


class CustomLoss:
    """損失関数のクラス."""

    def __init__(self, num_labels: int):
        """num_labels: 1=回帰, 2=2値分類, 3=多値分類."""
        if num_labels == 1:
            self.loss_fct = MSELoss()
        elif num_labels in [2, 3]:
            self.loss_fct = CrossEntropyLoss()
        else:
            raise ValueError("num_labels must be 1, 2, or 3")
        self.num_labels = num_labels

    def __call__(self, outputs, labels):
        logits = outputs.logits

        if self.num_labels == 1:
            # 回帰タスク
            logits = logits.squeeze()  # [B, L, C] -> [B * L] (C=1の場合)
            labels = labels.float()  # ラベルをfloat型に変換
            return self.loss_fct(logits, labels)
        else:
            # 分類タスク
            logits = logits.view(-1, self.num_labels)  # [B, L, C] -> [B * L, C]
            labels = labels.view(-1)  # [B, L] -> [B * L]
            return self.loss_fct(logits, labels)


# ================================================================================
# 評価指標の定義
# ================================================================================


def evaluate_regression_model(preds, labels):
    """回帰モデルの評価指標."""
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    r2_adj = 1 - (1 - r2) * (len(labels) - 1) / (len(labels) - 1 - 1)
    output_dict = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "r2_adj": r2_adj,
    }

    return output_dict


# 分類の評価指標
def evaluate_classification_model(preds, labels):
    """分類モデルの評価指標."""
    preds = np.array(preds)
    labels = np.array(labels)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    # ポーズありの箇所だけで評価指標を計算(label=1の箇所)
    only_pause_index = np.where(labels == 1, True, False)
    precision_only_pause = precision_score(labels[only_pause_index], preds[only_pause_index], average="macro")
    recall_only_pause = recall_score(labels[only_pause_index], preds[only_pause_index], average="macro")
    f1_only_pause = f1_score(labels[only_pause_index], preds[only_pause_index], average="macro")
    acc_only_pause = accuracy_score(labels[only_pause_index], preds[only_pause_index])
    # ポーズなしの箇所だけで評価指標を計算(label=0の箇所)
    only_no_pause_index = np.where(labels == 0)
    precision_only_no_pause = precision_score(labels[only_no_pause_index], preds[only_no_pause_index], average="macro")
    recall_only_no_pause = recall_score(labels[only_no_pause_index], preds[only_no_pause_index], average="macro")
    f1_only_no_pause = f1_score(labels[only_no_pause_index], preds[only_no_pause_index], average="macro")
    acc_only_no_pause = accuracy_score(labels[only_no_pause_index], preds[only_no_pause_index])
    # 混同行列を計算
    confusion_matrix_ = confusion_matrix(labels, preds)
    # 評価指標を格納
    output_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "precision_only_pause": precision_only_pause,
        "recall_only_pause": recall_only_pause,
        "f1_only_pause": f1_only_pause,
        "acc_only_pause": acc_only_pause,
        "precision_only_no_pause": precision_only_no_pause,
        "recall_only_no_pause": recall_only_no_pause,
        "f1_only_no_pause": f1_only_no_pause,
        "acc_only_no_pause": acc_only_no_pause,
        "confusion_matrix": confusion_matrix_,
        "preds": preds,
        "labels": labels,
        "preds_only_pause": preds[only_pause_index],
        "labels_only_pause": labels[only_pause_index],
        "preds_only_no_pause": preds[only_no_pause_index],
        "labels_only_no_pause": labels[only_no_pause_index],
    }

    return output_dict


# ================================================================================
# オプティマイザの設定
# ================================================================================


# 最適化アルゴリズムのパラメータを設定
def set_optimizer_params(model, encoder_lr, decoder_lr):
    """最適化アルゴリズムのパラメータを設定する関数."""

    # Pre-trained モデル（Bert）のパラメータか自前の部分のものか判別するのに使用
    def is_backbone(n):
        return "bert" in n

    # モデルのパラメータを名前とともにリストにする
    param_optimizer = list(model.named_parameters())

    # Weight decay を適用しないパラメータの名前のパターン
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # パラメータの種類ごとに最適化アルゴリズムのパラメータ（学習率・Weight decay の強さ）を設定
    optimizer_parameters = [
        # Bert の中のパラメータ．Weight decay させるもの
        {
            "params": [p for n, p in param_optimizer if is_backbone(n) and not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.01,
        },
        # Bert の中のパラメータ．Weight decay させないもの
        {
            "params": [p for n, p in param_optimizer if is_backbone(n) and any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        # Bert 以外のパラメータ．Weight decay させるもの
        {
            "params": [p for n, p in param_optimizer if not is_backbone(n) and not any(nd in n for nd in no_decay)],
            "lr": decoder_lr,
            "weight_decay": 0.01,
        },
        # Bert 以外のパラメータ．Weight decay させないもの
        {
            "params": [p for n, p in param_optimizer if not is_backbone(n) and any(nd in n for nd in no_decay)],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters


# ================================================================================
# トレーニングとエバリュエーション
# ================================================================================


def train(model, dataloader, optimizer, device, loss_fn):
    """トレーニング用の関数."""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        group_ids = batch["group_ids"].to(device)
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, group_ids=group_ids)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, loss_fn, num_labels=1):
    """評価用の関数."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    means = []
    vars_ = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            group_ids = batch["group_ids"].to(device)
            mean = batch["means"].to(device)
            var_ = batch["vars"].to(device)

            # モデルへの入力前に次元をチェックし、必要に応じてバッチ次元を追加
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            # print(input_ids.dim(), attention_mask.dim())
            # print(input_ids.shape, attention_mask.shape)

            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask

            outputs = model(input_ids, attention_mask=attention_mask, group_ids=group_ids)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # 予測値とラベルの取得
            if num_labels == 1:
                # 回帰タスクの予測値
                preds = outputs.logits.squeeze().cpu().numpy().flatten()
                labels = labels.cpu().numpy().flatten()
            else:
                # 分類タスクの予測値
                preds = outputs.logits.argmax(dim=-1).cpu().numpy().flatten()
                labels = labels.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels)
            means.extend(mean.cpu().numpy().flatten())
            vars_.extend(var_.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    means = np.array(means)
    vars_ = np.array(vars_)

    if num_labels == 1:
        # 回帰タスク

        # meansとvars_がそれぞれ0, 1であることを確認
        assert np.all(means == 0) and np.all(vars_ == 1), f"means: {means}, vars_: {vars_}"
        # ポーズを実数値へ逆標準化
        all_preds = all_preds * np.sqrt(vars_) + means
        all_labels = all_labels * np.sqrt(vars_) + means

        output_dict = evaluate_regression_model(np.array(all_preds), np.array(all_labels))
        output_dict["loss"] = loss / len(dataloader)
        return output_dict
    else:
        # 分類タスク
        output_dict = evaluate_classification_model(all_preds, all_labels)
        output_dict["loss"] = loss / len(dataloader)
        return output_dict


# ================================================================================
# モデルの定義
# ================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(
            last_hidden_state * input_mask_expanded, 1
        )  # 事前学習しているため、意味を持たない[PAD]等が含まれているため、それを除いたトークンの和を取っている
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """マスクされた文書中の最大値を取得する.

        Args:
            last_hidden_state(torch.Tensor): BERTの出力(hidden_size, seq_len)
            attention_mask(torch.Tensor): パディングのマスク

        Returns:
            max_embeddings(torch.Tensor): マスクされた文書中の最大値
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class CustomModel(nn.Module):
    """モデルのクラス."""

    def __init__(self, model_name, tokenizer, num_labels=1, linear_pooling_type="cls", layer_num=1, num_groups=1):
        super(CustomModel, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, config=self.config)
        self.meanpooling = MeanPooling()
        self.maxpooling = MaxPooling()
        self.tokenizer = tokenizer

        self.layer_num = layer_num
        self.linear_pooling_type = linear_pooling_type
        self.num_labels = num_labels
        self.num_groups = num_groups

        # 最終隠れ層に加算する埋め込み層
        self.embedding = nn.Embedding(num_groups, self.config.hidden_size)

        # Linear層
        # 上で事前学習済みモデルの情報をconfigとして読み込んでいるので、hidden_sizeを呼び出せる。
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # 線形層に通す直前のfeatureを取りだす(ex, CLS or Pooled Hiddenlayer)
    def feature(self, **inputs):
        outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden_states = outputs[0]  # 出力の中のlast_hidden_layerだけ取り出す

        # 埋め込み層を作成
        group_ids = inputs.get("group_ids", None)
        if group_ids is None:
            raise ValueError("group_ids is not None")
        if group_ids.size(0) != inputs["input_ids"].size(0):
            raise ValueError("group_ids.size(0) != input_ids.size(0)")
        embedding = self.embedding(group_ids)
        embedding = embedding.unsqueeze(1).repeat(1, last_hidden_states.size(1), 1)

        # 埋め込み層を加算
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        last_hidden_states_embbed = last_hidden_states + embedding

        # input_idsから[SEP]トークンの位置を探す
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (inputs["input_ids"] == sep_token_id).nonzero()

        # 複数の[SEP]トークンがある場合、最初のものを使用
        first_sep_position = sep_positions[0, 1] if len(sep_positions) > 0 else None

        if self.linear_pooling_type == "cls":
            feature = last_hidden_states[:, 0, :].view(-1, self.config.hidden_size)
        elif first_sep_position is not None and self.linear_pooling_type == "sep":
            feature = last_hidden_states_embbed[:, first_sep_position, :].view(-1, self.config.hidden_size)
        elif self.linear_pooling_type == "mean_pooling":
            feature = self.meanpooling(last_hidden_states, inputs["attention_mask"])
        elif self.linear_pooling_type == "max_pooling":
            feature = self.maxpooling(last_hidden_states, inputs["attention_mask"])
        else:
            raise ValueError("linear_pooling_type must be cls or mean_pooling or max_pooling")

        return feature

    # 線形層
    def forward(self, input_ids, attention_mask, **inputs):
        feature = self.feature(input_ids=input_ids, attention_mask=attention_mask, group_ids=inputs["group_ids"])
        output = self.fc(feature)  # feature関数からの出力をlinear関数に通す
        # outputs.logitsでアクセスできるようにする
        outputs = type("Outputs", (object,), {"logits": output})
        return outputs


# ================================================================================
# 早期終了のためのクラス
# ================================================================================
class EarlyStopping:
    """早期終了のためのクラス."""

    def __init__(self, patience=3, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"早期終了カウンター: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"検証損失が減少しました。 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  モデルを保存します..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        self.trace_func(
            f"ベストモデルを {self.counter} エポック時点の検証損失: {self.val_loss_min:.6f} で読み込みました。"
        )

    def delete_checkpoint(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        self.trace_func("checkpointを削除しました。")


# ================================================================================
# トレーニングの実行
# ================================================================================


def run_training(cfg):
    """トレーニングを実行する関数."""
    # 乱数シードの固定
    seed_everything(cfg.seed)

    # 埋め込み層のグループ数を決定
    num_groups_dict = {
        "audiobook": 71,
        # "narrative": 2,
        "speaker": 36,
        "book": 20,
        # "audiobook_narrative": 74 * 2,
        "none": 1,  # 1グループは埋め込みが意味を持たないため、使用しない
        "all": 1,
    }
    num_groups = num_groups_dict.get(cfg.dataset.preprocess_type, None)
    if num_groups is None:
        raise ValueError(f"num_groups is None, please check preprocess_type: {cfg.dataset.preprocess_type}")

    # モデルとトークナイザの初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained(cfg.model.name)
    model = CustomModel(
        cfg.model.name,
        tokenizer,
        cfg.model.num_labels,
        cfg.model.linear_pooling_type,
        cfg.model.layer_num,
        num_groups=num_groups,
    )
    device = torch.device(cfg.device, cfg.gpu_id)
    model.to(device)

    # オプティマイザの設定
    optimizer_grouped_parameters = set_optimizer_params(model, cfg.training.encoder_lr, cfg.training.decoder_lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.encoder_lr, eps=cfg.training.eps)

    # 早期終了の設定
    early_stopping = EarlyStopping(patience=cfg.training.patience, verbose=True)

    # 損失関数の設定
    loss_fn = CustomLoss(cfg.model.num_labels)

    # データセットの作成
    train_df, val_df, test_df = data_load(
        pause_time_threshold_ms=cfg.dataset.pause_time_threshold_ms,
        preprocess_type=cfg.dataset.preprocess_type,
        num_labels=cfg.model.num_labels,
    )
    dataloader_train, dataloader_val, dataloader_test = create_dataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length_train=cfg.training.max_length_train,
        max_length_val=cfg.training.max_length_val,
        max_length_test=cfg.training.max_length_test,
        batch_size=cfg.training.batch_size,
        num_labels=cfg.model.num_labels,
        preprocess_type=cfg.dataset.preprocess_type,
    )

    # トレーニングと評価のメトリクスを保存するリスト
    training_losses = []
    validation_losses = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"{str(cfg.dataset.pause_time_threshold_ms)}ms_{cfg.dataset.preprocess_type}_{cfg.model.num_labels}labels_linear_maxlen[{cfg.training.max_length_train}.{cfg.training.max_length_val}.{cfg.training.max_length_test}]_num_groups{num_groups}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("epochs", cfg.training.epochs)
        mlflow.log_param("max_length_train", cfg.training.max_length_train)
        mlflow.log_param("max_length_val", cfg.training.max_length_val)
        mlflow.log_param("max_length_test", cfg.training.max_length_test)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("encoder_lr", cfg.training.encoder_lr)
        mlflow.log_param("decoder_lr", cfg.training.decoder_lr)
        mlflow.log_param("eps", cfg.training.eps)
        mlflow.log_param("optimizer", optimizer)  # AdamW
        mlflow.log_param("MODEL_NAME", cfg.model.name)  # cl-tohoku/bert-base-japanese-whole-word-masking
        mlflow.log_param("device", device)  # cuda or cpu
        mlflow.log_param("gpu_id", cfg.gpu_id)
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("patience", cfg.training.patience)
        mlflow.log_param("num_labels", cfg.model.num_labels)
        mlflow.log_param("preprocess_type", cfg.dataset.preprocess_type)
        mlflow.log_param("pause_time_threshold_ms", cfg.dataset.pause_time_threshold_ms)
        mlflow.log_param("linear_pooling_type", cfg.model.linear_pooling_type)
        mlflow.log_param("layer_num", cfg.model.layer_num)
        mlflow.log_param("num_groups", num_groups)

        for epoch in range(cfg.training.epochs):
            train_loss = train(model, dataloader_train, optimizer, device, loss_fn)
            training_losses.append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if cfg.model.num_labels == 1:
                # 回帰タスク
                output_dict = evaluate(model, dataloader_val, device, loss_fn, cfg.model.num_labels)

                # メトリクスの保存

                val_loss = output_dict["loss"]
                mae = output_dict["mae"]
                mse = output_dict["mse"]
                rmse = output_dict["rmse"]
                r2 = output_dict["r2"]
                r2_adj = output_dict["r2_adj"]

                validation_losses.append(val_loss)

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_mae", mae, step=epoch)
                mlflow.log_metric("val_mse", mse, step=epoch)
                mlflow.log_metric("val_rmse", rmse, step=epoch)
                mlflow.log_metric("val_r2", r2, step=epoch)
                mlflow.log_metric("val_r2_adj", r2_adj, step=epoch)

                val_metric = rmse
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation loss: {val_loss} || MAE: {mae} || MSE: {mse} || RMSE: {rmse} || R2: {r2} || R2_adj: {r2_adj}"
                )
            else:
                # 分類タスク
                output_dict = evaluate(model, dataloader_val, device, loss_fn, cfg.model.num_labels)

                # メトリクスの保存
                val_loss = output_dict["loss"]
                val_f1 = output_dict["f1"]
                val_acc = output_dict["acc"]
                val_precision = output_dict["precision"]
                val_recall = output_dict["recall"]
                val_f1_only_pause = output_dict["f1_only_pause"]
                val_acc_only_pause = output_dict["acc_only_pause"]
                val_precision_only_pause = output_dict["precision_only_pause"]
                val_recall_only_pause = output_dict["recall_only_pause"]
                val_f1_only_no_pause = output_dict["f1_only_no_pause"]
                val_acc_only_no_pause = output_dict["acc_only_no_pause"]
                val_precision_only_no_pause = output_dict["precision_only_no_pause"]
                val_recall_only_no_pause = output_dict["recall_only_no_pause"]
                # val_confusion_matrix = output_dict["confusion_matrix"]

                validation_losses.append(val_loss)

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("val_precision", val_precision, step=epoch)
                mlflow.log_metric("val_recall", val_recall, step=epoch)
                mlflow.log_metric("val_f1_only_pause", val_f1_only_pause, step=epoch)
                mlflow.log_metric("val_acc_only_pause", val_acc_only_pause, step=epoch)
                mlflow.log_metric("val_precision_only_pause", val_precision_only_pause, step=epoch)
                mlflow.log_metric("val_recall_only_pause", val_recall_only_pause, step=epoch)
                mlflow.log_metric("val_f1_only_no_pause", val_f1_only_no_pause, step=epoch)
                mlflow.log_metric("val_acc_only_no_pause", val_acc_only_no_pause, step=epoch)
                mlflow.log_metric("val_precision_only_no_pause", val_precision_only_no_pause, step=epoch)
                mlflow.log_metric("val_recall_only_no_pause", val_recall_only_no_pause, step=epoch)
                # mlflow.log_metric("val_confusion_matrix", val_confusion_matrix, step=epoch)

                val_metric = val_f1

                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation loss: {val_loss} || Validation f1: {val_f1} || Validation acc: {val_acc}"
                )

            # 早期終了判定
            # early_stopping(val_loss)
            early_stopping(val_metric, model)
            if early_stopping.early_stop:
                print("早期終了")
                break

        # テストデータでの評価
        early_stopping.load_checkpoint(model)

        # メトリクスの保存
        test_output_dict = evaluate(model, dataloader_test, device, loss_fn, cfg.model.num_labels)
        if cfg.model.num_labels == 1:
            # 回帰タスク
            rmse = test_output_dict["rmse"]
            mlflow.log_metric("test_rmse", rmse)
            print(f"Test RMSE: {rmse}")
        else:
            # 分類タスク
            metrics_to_log = [
                "precision",
                "recall",
                "f1",
                "acc",
                "precision_only_pause",
                "recall_only_pause",
                "f1_only_pause",
                "acc_only_pause",
                "precision_only_no_pause",
                "recall_only_no_pause",
                "f1_only_no_pause",
                "acc_only_no_pause",
            ]
            for metric in metrics_to_log:
                value = test_output_dict[metric]
                mlflow.log_metric(f"test_{metric}", value)
                print(f"Test {metric.replace('_', ' ').capitalize()}: {value}")

            # 混合行列を画像として保存
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            confusion_matrix_ = test_output_dict["confusion_matrix"]
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_)
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.savefig("test_confusion_matrix.png")
            mlflow.log_artifact("test_confusion_matrix.png")

        # モデルの保存
        mlflow.pytorch.log_model(model, "model")
        # 早期終了のためのチェックポイントを削除
        early_stopping.delete_checkpoint()

        # Hydraの成果物をArtifactに保存
        # log_dir = cfg.log_path
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/config.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/hydra.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/overrides.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, "main.log"))
