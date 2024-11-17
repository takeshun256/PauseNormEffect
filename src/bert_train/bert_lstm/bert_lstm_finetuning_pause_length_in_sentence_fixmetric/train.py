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

# import own library
sys.path.append("/home/takeshun256/PausePrediction")
from config import DATA_TAKESHUN256_DIR

# ========================================================================================================
# 以下、学習の設定
# ========================================================================================================


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


# ========================================================================================================
# 以下、データのロードと前処理
# ========================================================================================================


def data_load(pause_time_threshold_ms=100, preprocess_type="none", num_labels=1):
    """データのロードを行う関数."""
    # conf
    corpus_name = "jmac"
    exp_name = "03_VAD_Adjusted"
    exp_dir = Path(DATA_TAKESHUN256_DIR) / corpus_name / exp_name
    output_dir = exp_dir / "data_bert"
    df_dir = output_dir / f"{pause_time_threshold_ms}ms" / f"{preprocess_type}"
    try:
        train_df = pkl.load(open(df_dir / f"bert_traindata_{num_labels}label_train.pkl", "rb"))
    except FileNotFoundError:
        print("train_df is not found. Expected path is ", df_dir / f"bert_traindata_{num_labels}label_train.pkl")
        sys.exit(1)
    try:
        val_df = pkl.load(open(df_dir / f"bert_traindata_{num_labels}label_val.pkl", "rb"))
    except FileNotFoundError:
        print("val_df is not found. Expected path is ", df_dir / f"bert_traindata_{num_labels}label_val.pkl")
        sys.exit(1)
    try:
        test_df = pkl.load(open(df_dir / f"bert_traindata_{num_labels}label_test.pkl", "rb"))
    except FileNotFoundError:
        print("test_df is not found. Expected path is ", df_dir / f"bert_traindata_{num_labels}label_test.pkl")
        sys.exit(1)

    return train_df, val_df, test_df


# ========================================================================================================
# 以下、データセットの作成
# ========================================================================================================


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
):
    """データセットを作成する関数."""
    # Extracting texts and labels for each dataset
    train_texts, train_labels = train_df["texts"].tolist(), train_df["labels"].tolist()
    train_means, train_vars = train_df["means"].tolist(), train_df["vars"].tolist()
    val_texts, val_labels = val_df["texts"].tolist(), val_df["labels"].tolist()
    val_means, val_vars = val_df["means"].tolist(), val_df["vars"].tolist()
    test_texts, test_labels = test_df["texts"].tolist(), test_df["labels"].tolist()
    test_means, test_vars = test_df["means"].tolist(), test_df["vars"].tolist()

    dataset_train = CreateDataset(
        train_texts, train_labels, train_means, train_vars, tokenizer, max_length_train, num_labels
    )
    dataset_val = CreateDataset(val_texts, val_labels, val_means, val_vars, tokenizer, max_length_val, num_labels)
    dataset_test = CreateDataset(
        test_texts, test_labels, test_means, test_vars, tokenizer, max_length_test, num_labels
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

    def __init__(self, texts, labels, means, vars, tokenizer, max_length, num_labels=1):
        self.texts = texts
        self.labels = labels
        self.means = means
        self.vars = vars
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels  # 1=回帰, 2=2値分類, 3=多値分類

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        morphs = self.texts[index]
        label = self.labels[index]
        mean = self.means[index]
        var = self.vars[index]
        input_ids = [self.tokenizer.cls_token_id]
        boundary = [False]
        label_ids = [-100]  # [CLS] はラベルなし

        for m, la in zip(morphs, label):
            morph_toks = self.tokenizer.encode(m, add_special_tokens=False)
            if len(input_ids) + len(morph_toks) + 1 > self.max_length:
                break
            input_ids.extend(morph_toks)
            boundary.extend([False] * (len(morph_toks) - 1) + [True])
            label_ids.extend([la] * len(morph_toks))

        input_ids.append(self.tokenizer.sep_token_id)
        boundary.append(False)
        label_ids.append(-100)  # [SEP] はラベルなし

        # パディング
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        boundary += [False] * padding_length
        label_ids += [-100] * padding_length

        # アテンションマスクの計算
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        if self.num_labels == 1:
            # 回帰タスク
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "boundary": torch.tensor(boundary, dtype=torch.bool),
                "labels": torch.tensor(label_ids, dtype=torch.float),
                "means": torch.tensor(mean, dtype=torch.float),
                "vars": torch.tensor(var, dtype=torch.float),
            }
        else:
            # 分類タスク
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "boundary": torch.tensor(boundary, dtype=torch.bool),
                "labels": torch.tensor(label_ids, dtype=torch.long),
                "means": torch.tensor(mean, dtype=torch.float),
                "vars": torch.tensor(var, dtype=torch.float),
            }


# ========================================================================================================
# 以下、Loss関数の定義
# ========================================================================================================


class CustomLoss:
    """損失関数のクラス."""

    def __init__(self, num_labels: int):
        """num_labels: 1=回帰, 2=2値分類, 3=多値分類."""
        if num_labels == 1:
            self.loss_fct = MSELoss()
        elif num_labels == 2:
            self.loss_fct = CrossEntropyLoss()
        elif num_labels == 3:
            self.loss_fct = CrossEntropyLoss()
        else:
            raise ValueError("num_labels must be 1, 2, or 3")
        self.num_labels = num_labels

    def __call__(self, outputs, labels, boundaries):
        logits = outputs.logits

        if self.num_labels == 1:
            # 回帰タスク
            # [B, L, C] -> [B * L]
            predictions = logits.view(-1)
            labels = labels.view(-1)
            boundaries = boundaries.view(-1)
            assert predictions.shape == labels.shape == boundaries.shape
            # 損失を計算
            loss = self.loss_fct(predictions[boundaries], labels[boundaries])

            # # B=batch_size, L=max_length, C=1
            # logits = logits.float().view(-1)  # [B, L, C] -> [B * L, C]
            # labels = labels.float().view(-1)  # [B, L, C] -> [B * L, C]
            # active_loss = boundaries.view(-1)  # [B, L] -> [B * L]
            # active_logits = logits[active_loss]  # [B * L, C] -> [B * L, C]
            # active_labels = labels[active_loss]  # [B * L, C] -> [B * L, C]
            # if self.loss_ignoring_no_pause:
            #     # [NO_PAUSE] は損失計算に使用しない
            #     is_pause = active_labels != 0
            #     active_logits = active_logits[is_pause]
            #     active_labels = active_labels[is_pause]
            # return self.loss_fct(active_logits, active_labels)  # [B * L, C], [B * L, C]
        else:
            # 分類タスクのためのCrossEntropy損失を計算
            logits = logits.view(-1, self.num_labels)  # [B, L, C] -> [B * L, C]
            labels = labels.view(-1)  # [B, L] -> [B * L]
            boundaries = boundaries.view(-1)  # [B, L] -> [B * L]
            active_logits = logits[boundaries == 1]  # [B * L, C] -> [B_active * L, C]
            active_labels = labels[boundaries == 1]  # [B * L] -> [B_active * L]
            assert active_logits.shape[0] == active_labels.shape[0]
            # 損失を計算
            loss = self.loss_fct(active_logits, active_labels)
            # # B=batch_size, L=max_length, C=num_labels
            # logits = logits.view(-1, self.num_labels)  # [B, L, C] -> [B * L, C]
            # labels = labels.view(-1)  # [B, L] -> [B * L]
            # active_loss = boundaries.view(-1)  # [B, L] -> [B * L]
            # active_logits = logits[active_loss]  # [B * L, C] -> [B * L, C]
            # active_labels = labels[active_loss]  # [B * L] -> [B * L]
            # return self.loss_fct(active_logits, active_labels)  # [B * L, C], [B * L]
        return loss


# ========================================================================================================
# 以下、オプティマイザの設定
# ========================================================================================================


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


# ========================================================================================================
# 以下、訓練と評価の関数
# ========================================================================================================


def train(model, dataloader, optimizer, device, loss_fn):
    """トレーニング用の関数."""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundary"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels, boundaries)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def predict_and_evaluate(model, dataloader, device):
    """予測値とラベルを取得して評価する関数."""
    model.eval()
    # 予測する
    predictions = []
    boundarys = []
    labels = []
    means = []
    vars_ = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundarys.extend(batch["boundary"].cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
            means.extend(batch["means"].cpu().numpy())
            vars_.extend(batch["vars"].cpu().numpy())
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if model.num_labels == 1:
                # 回帰タスクの場合、出力をそのまま予測値として使用
                predictions.extend(logits.squeeze().cpu().numpy())
            else:
                # 分類タスクの場合、最も確率の高いクラスを予測値として使用
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
    predictions = np.array(predictions)
    labels = np.array(labels)
    boundarys = np.array(boundarys)
    means = np.array(means)
    vars_ = np.array(vars_)

    # 評価指標を計算する
    def calc_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def calc_precision(y_true, y_pred):
        return precision_score(y_true, y_pred, average="macro")

    def calc_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, average="macro")

    def calc_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    def calc_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    # 評価指標を計算して結果を格納
    if model.num_labels == 1:
        # broadcast用にreshape
        labels = labels.reshape(-1, labels.shape[-1])
        predictions = predictions.reshape(-1, predictions.shape[-1])
        boundarys = boundarys.reshape(-1, boundarys.shape[-1])
        means = means.reshape(-1, 1)
        vars_ = vars_.reshape(-1, 1)
        # ポーズを実測値へ逆標準化
        labels = labels * np.sqrt(np.array(vars_)) + np.array(means)
        predictions = predictions * np.sqrt(np.array(vars_)) + np.array(means)
        # 計算する箇所のみを抽出
        assert labels.shape == predictions.shape == boundarys.shape
        active_labels = np.array(labels)[np.array(boundarys)]
        active_predictions = np.array(predictions)[np.array(boundarys)]
        # 評価指標を計算
        rmse = calc_rmse(active_labels, active_predictions)
        output_dict = {
            "rmse": rmse,
            "predictions": active_predictions,
            "labels": active_labels,
            "means": means,
            "vars": vars_,
        }
    elif model.num_labels == 2:
        # 有効な箇所のみを抽出
        active_labels = np.array(labels)[np.array(boundarys)]
        active_predictions = np.array(predictions)[np.array(boundarys)]

        # TODO: ポーズありの箇所だけで評価指標も計算する
        # ポーズ有無含めた評価指標を計算
        precision = calc_precision(active_labels, active_predictions)
        recall = calc_recall(active_labels, active_predictions)
        f1 = calc_f1(active_labels, active_predictions)
        accuracy = calc_accuracy(active_labels, active_predictions)
        # ポーズありの箇所だけで評価指標を計算(label=1の箇所)
        only_pause_index = np.where(active_labels == 1)
        active_labels_only_pause = active_labels[only_pause_index]
        active_predictions_only_pause = active_predictions[only_pause_index]
        precision_only_pause = calc_precision(active_labels_only_pause, active_predictions_only_pause)
        recall_only_pause = calc_recall(active_labels_only_pause, active_predictions_only_pause)
        f1_only_pause = calc_f1(active_labels_only_pause, active_predictions_only_pause)
        accuracy_only_pause = calc_accuracy(active_labels_only_pause, active_predictions_only_pause)
        # ポーズなしの箇所だけで評価指標を計算(label=0の箇所)
        only_no_pause_index = np.where(active_labels == 0)
        active_labels_only_no_pause = active_labels[only_no_pause_index]
        active_predictions_only_no_pause = active_predictions[only_no_pause_index]
        precision_only_no_pause = calc_precision(active_labels_only_no_pause, active_predictions_only_no_pause)
        recall_only_no_pause = calc_recall(active_labels_only_no_pause, active_predictions_only_no_pause)
        f1_only_no_pause = calc_f1(active_labels_only_no_pause, active_predictions_only_no_pause)
        accuracy_only_no_pause = calc_accuracy(active_labels_only_no_pause, active_predictions_only_no_pause)
        # 混合行列を計算
        confusion_matrix_ = confusion_matrix(active_labels, active_predictions)
        # 評価指標を格納
        output_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "precision_only_pause": precision_only_pause,
            "recall_only_pause": recall_only_pause,
            "f1_only_pause": f1_only_pause,
            "accuracy_only_pause": accuracy_only_pause,
            "precision_only_no_pause": precision_only_no_pause,
            "recall_only_no_pause": recall_only_no_pause,
            "f1_only_no_pause": f1_only_no_pause,
            "accuracy_only_no_pause": accuracy_only_no_pause,
            "predictions": active_predictions,
            "labels": active_labels,
            "only_pause_predictions": active_predictions_only_pause,
            "only_pause_labels": active_labels_only_pause,
            "only_no_pause_predictions": active_predictions_only_no_pause,
            "only_no_pause_labels": active_labels_only_no_pause,
            "confusion_matrix": confusion_matrix_,
        }
    else:
        # エラー
        raise ValueError(f"num_labels must be 1 or 2, get {model.num_labels} instead.")
    return output_dict


# def evaluate(model, dataloader, optimizer, device, loss_fn, num_labels=1):
#     """評価用の関数."""
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             boundaries = batch["boundary"].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask)
#             loss = loss_fn(outputs, labels, boundaries)
#             total_loss += loss.item()

#             # 予測値とラベルの取得
#             preds = outputs.logits.argmax(dim=-1)[boundaries]
#             active_labels = labels[boundaries]
#             all_preds.extend(preds.cpu().numpy().flatten())
#             all_labels.extend(active_labels.cpu().numpy().flatten())

#     if num_labels == 1:
#         # 回帰タスク
#         return None, None, total_loss / len(dataloader)
#     else:
#         # 分類タスク
#         f1, acc = evaluate_classification_model(all_preds, all_labels)
#         return f1, acc, total_loss / len(dataloader)


# def predict(model, dataloader, device):
#     """予測値とラベルを取得する関数."""
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Predicting"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             boundaries = batch["boundary"].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask)

#             # 予測値とラベルの取得
#             logits = outputs.logits.float().squeeze().cpu().numpy()
#             labels = labels.float().squeeze().cpu().numpy()
#             boundaries = boundaries.cpu().numpy()
#             for i, (logit, label, boundary) in enumerate(zip(logits, labels, boundaries)):
#                 active_logit = logit[boundary]
#                 active_label = label[boundary]
#                 all_preds.append(active_logit)
#                 all_labels.append(active_label)

#     return all_preds, all_labels


# ========================================================================================================
# 以下、モデルの定義
# ========================================================================================================
class CustomModel(nn.Module):
    """モデルのクラス."""

    def __init__(self, model_name, tokenizer, num_labels=1, linear_pooling_type="none", layer_num=1, num_groups=1):
        """モデルの初期化.

        param:
            model_name: str, モデルの名前
            tokenizer: object, トークナイザ
            num_labels: int, 分類タスクのクラス数
            linear_pooling_type: str, 線形層に通す直前の特徴量を取り出す方法
            layer_num: int, BiLSTMの層数
            num_groups: int, グループ数
        """
        super(CustomModel, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        self.meanpooling = MeanPooling()
        self.maxpooling = MaxPooling()
        self.tokenizer = tokenizer

        self.layer_num = layer_num
        self.linear_pooling_type = linear_pooling_type
        self.num_labels = num_labels

        # 最終隠れ層に加算する埋め込み層
        self.embedding = nn.Embedding(num_groups, self.config.hidden_size)

        # BiLSTM層
        self.bilstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True,
        )

        # 線形層
        self.fc = nn.Linear(self.config.hidden_size * 2, num_labels)  # BiLSTMなのでhidden_size * 2
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
        outputs = self.bert(**inputs)
        # last_hidden_states = outputs[0]  # 出力の中のlast_hidden_layerだけ取り出す
        last_hidden_states = outputs[
            "last_hidden_state"
        ]  # 明示的に指定することで、出力の中のlast_hidden_layerだけ取り出す

        first_sep_position = None
        if self.linear_pooling_type == "sep":
            # input_idsから[SEP]トークンの位置を探す
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = (inputs["input_ids"] == sep_token_id).nonzero()
            # 複数の[SEP]トークンがある場合、最初のものを使用
            first_sep_position = sep_positions[0, 1] if len(sep_positions) > 0 else None

        if self.linear_pooling_type == "cls":
            feature = last_hidden_states[:, 0, :].view(-1, self.config.hidden_size)
        elif first_sep_position is not None and self.linear_pooling_type == "sep":
            feature = last_hidden_states[:, first_sep_position, :].view(-1, self.config.hidden_size)
        elif self.linear_pooling_type == "mean_pooling":
            feature = self.meanpooling(last_hidden_states, inputs["attention_mask"])
        elif self.linear_pooling_type == "max_pooling":
            feature = self.maxpooling(last_hidden_states, inputs["attention_mask"])
        elif self.linear_pooling_type == "none":
            feature = last_hidden_states
        else:
            raise ValueError("linear_pooling_type must be cls or mean_pooling or max_pooling")

        return feature

    # 線形層
    def forward(self, input_ids, attention_mask, **inputs):
        feature = self.feature(input_ids=input_ids, attention_mask=attention_mask)

        # 埋め込み層を作成
        # group_ids = inputs.get("group_ids", None)
        # if group_ids is not None:
        #     raise ValueError("group_ids is not None")
        # if group_ids.size(0) != input_ids.size(0):
        #     raise ValueError("group_ids.size(0) != input_ids.size(0)")
        # embedding = self.embedding(group_ids)
        # embedding = embedding.unsqueeze(1).repeat(1, feature.size(1), 1)

        # 埋め込み層を加算
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        # feature_embedded = feature + embedding
        feature_embedded = feature

        # BiLSTM層を通す
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size * 2]
        lstm_output, (hidden, cell) = self.bilstm(feature_embedded)

        # 線形層に通す
        # [batch_size, seq_len, hidden_size * 2] -> [batch_size, seq_len, num_labels]
        output = self.fc(lstm_output)  # feature関数からの出力をlinear関数に通す

        # outputs.logitsでアクセスできるようにする
        outputs = type("Outputs", (object,), {"logits": output})
        return outputs


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


# ========================================================================================================
# 以下、EarlyStoppingクラス
# ========================================================================================================


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


# ========================================================================================================
# 以下、トレーニングの実行
# ========================================================================================================


def run_training(cfg):
    """トレーニングを実行する関数."""
    # 乱数シードの固定
    seed_everything(cfg.seed)

    # モデルとトークナイザの初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained(cfg.model.name)
    # model = BertForTokenClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_labels)
    model = CustomModel(
        model_name=cfg.model.name,
        tokenizer=tokenizer,
        num_labels=cfg.model.num_labels,
        linear_pooling_type=cfg.model.linear_pooling_type,
        layer_num=cfg.model.layer_num,
        # num_groups=cfg.model.num_groups,
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
    )

    # トレーニングと評価のメトリクスを保存するリスト
    training_losses = []
    validation_metrics = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"{str(cfg.dataset.pause_time_threshold_ms)}ms_{cfg.dataset.preprocess_type}_{cfg.model.num_labels}labels_maxlen[{cfg.training.max_length_train}.{cfg.training.max_length_val}.{cfg.training.max_length_test}]"

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

        for epoch in range(cfg.training.epochs):
            train_loss = train(model, dataloader_train, optimizer, device, loss_fn)
            training_losses.append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if cfg.model.num_labels == 1:
                # 回帰タスク
                val_output_dict = predict_and_evaluate(model, dataloader_val, device)

                # メトリクスの保存
                rmse = val_output_dict["rmse"]
                val_metric = rmse
                validation_metrics.append(val_metric)
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation RMSE: {val_metric}"
                )
                mlflow.log_metric("val_rmse", rmse, step=epoch)

            else:
                # 分類タスク
                val_output_dict = predict_and_evaluate(model, dataloader_val, device)

                # メトリクスの保存
                precision = val_output_dict["precision"]
                recall = val_output_dict["recall"]
                f1 = val_output_dict["f1"]
                accuracy = val_output_dict["accuracy"]
                precision_only_pause = val_output_dict["precision_only_pause"]
                recall_only_pause = val_output_dict["recall_only_pause"]
                f1_only_pause = val_output_dict["f1_only_pause"]
                accuracy_only_pause = val_output_dict["accuracy_only_pause"]
                precision_only_no_pause = val_output_dict["precision_only_no_pause"]
                recall_only_no_pause = val_output_dict["recall_only_no_pause"]
                f1_only_no_pause = val_output_dict["f1_only_no_pause"]
                accuracy_only_no_pause = val_output_dict["accuracy_only_no_pause"]
                confusion_matrix_ = val_output_dict["confusion_matrix"]
                val_metric = f1
                validation_metrics.append(val_metric)

                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_f1", f1, step=epoch)
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
                mlflow.log_metric("val_precision_only_pause", precision_only_pause, step=epoch)
                mlflow.log_metric("val_recall_only_pause", recall_only_pause, step=epoch)
                mlflow.log_metric("val_f1_only_pause", f1_only_pause, step=epoch)
                mlflow.log_metric("val_accuracy_only_pause", accuracy_only_pause, step=epoch)
                mlflow.log_metric("val_precision_only_no_pause", precision_only_no_pause, step=epoch)
                mlflow.log_metric("val_recall_only_no_pause", recall_only_no_pause, step=epoch)
                mlflow.log_metric("val_f1_only_no_pause", f1_only_no_pause, step=epoch)
                mlflow.log_metric("val_accuracy_only_no_pause", accuracy_only_no_pause, step=epoch)
                # mlflow.log_metric("val_confusion_matrix", json.dumps(confusion_matrix_.tolist()), step=epoch) # リストは保存できないためコメントアウト

                print(f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation F1: {f1}")

            # 早期終了判定
            early_stopping(val_loss=val_metric, model=model)
            if early_stopping.early_stop:
                print("早期終了")
                break

        # テストデータでの評価
        early_stopping.load_checkpoint(model)
        test_output_dict = predict_and_evaluate(model, dataloader_test, device)

        # メトリクスの保存
        if cfg.model.num_labels == 1:
            # 回帰タスク
            rmse = test_output_dict["rmse"]
            mlflow.log_metric("test_rmse", rmse)
            print(f"Test RMSE: {rmse}")
        else:
            # 分類タスク
            precision = test_output_dict["precision"]
            recall = test_output_dict["recall"]
            f1 = test_output_dict["f1"]
            accuracy = test_output_dict["accuracy"]
            precision_only_pause = test_output_dict["precision_only_pause"]
            recall_only_pause = test_output_dict["recall_only_pause"]
            f1_only_pause = test_output_dict["f1_only_pause"]
            accuracy_only_pause = test_output_dict["accuracy_only_pause"]
            precision_only_no_pause = test_output_dict["precision_only_no_pause"]
            recall_only_no_pause = test_output_dict["recall_only_no_pause"]
            f1_only_no_pause = test_output_dict["f1_only_no_pause"]
            accuracy_only_no_pause = test_output_dict["accuracy_only_no_pause"]
            confusion_matrix_ = test_output_dict["confusion_matrix"]

            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision_only_pause", precision_only_pause)
            mlflow.log_metric("test_recall_only_pause", recall_only_pause)
            mlflow.log_metric("test_f1_only_pause", f1_only_pause)
            mlflow.log_metric("test_accuracy_only_pause", accuracy_only_pause)
            mlflow.log_metric("test_precision_only_no_pause", precision_only_no_pause)
            mlflow.log_metric("test_recall_only_no_pause", recall_only_no_pause)
            mlflow.log_metric("test_f1_only_no_pause", f1_only_no_pause)
            mlflow.log_metric("test_accuracy_only_no_pause", accuracy_only_no_pause)
            # mlflow.log_metric("test_confusion_matrix", json.dumps(confusion_matrix_.tolist())) # リストは保存できないためコメントアウト
            # 混合行列を画像として保存
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_)
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.savefig("test_confusion_matrix.png")
            mlflow.log_artifact("test_confusion_matrix.png")

            print(f"Test F1: {f1}")

        # モデルの保存
        mlflow.pytorch.log_model(model, "model")
        early_stopping.delete_checkpoint()

        # Hydraの成果物をArtifactに保存
        # log_dir = cfg.log_path
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/config.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/hydra.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/overrides.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, "main.log"))
