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

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForTokenClassification, BertJapaneseTokenizer


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


def data_load(pause_time_threshold_ms=80, preprocess_type="all", num_labels=1):
    """データの読み込み."""
    sys.path.append("/home/takeshun256/PausePrediction")

    # import own library
    from config import DATA_DIR, DATA_IN_ROOT_DIR, DATA_TAKESHUN256_DIR, SRC_DIR

    # define path
    corpus_name = "jmac"
    exp_name = "03_VAD_Adjusted"

    exp_dir = Path(DATA_TAKESHUN256_DIR) / corpus_name / exp_name

    assert exp_dir.exists()

    # データの読み込み

    filename = f"bert_traindata_pause_position_with_length_wo_sokuon_{str(pause_time_threshold_ms)}ms_normalized.pkl"
    print(f"Loading {filename} ...")
    assert (exp_dir / filename).exists()

    with open(exp_dir / filename, "rb") as f:
        df = pickle.load(f)

    # 奇数番号の要素をtextとして、偶数番号の要素をlabelとして取得
    texts = []
    labels = []

    column_dict = {
        "none": "morp_pause_clip_no_pause",
        "all": f"morp_pause_clip_no_pause_normalized_{str(pause_time_threshold_ms)}ms_all",
        "narrative": f"morp_pause_clip_no_pause_normalized_{str(pause_time_threshold_ms)}ms_narrative",
        "audiobook": f"morp_pause_clip_no_pause_normalized_{str(pause_time_threshold_ms)}ms_audiobook",
        "audiobook_narrative": f"morp_pause_clip_no_pause_normalized_{str(pause_time_threshold_ms)}ms_audiobook_narrative",
    }
    column_name = column_dict[preprocess_type]
    print(f"Using {column_name} ...")
    for a in df[column_name].values:
        if len(a) == 0:
            texts.append([])
            labels.append([])
            continue
        a = a[1:]  # 最初の要素は、[PAUSE] or [NO_PAUSE] なので削除
        texts.append(a[::2])
        labels.append(a[1::2])
        assert len(texts[-1]) == len(labels[-1])

    df["texts"] = texts
    df["labels_str"] = labels


    if num_labels == 1:
        # 回帰タスク
        # [PAUSE 0.5] などの文字列から、0.5 の部分を取得, [NO_PAUSE] は 0 にする。[PAUSE -0.5] などもあり得るので注意
        # df["labels"] = df["labels_str"].apply(
        #     lambda x: [float(re.findall(r"\d+\.\d+", a)[0]) if a.startswith("[PAUSE") else 0 for a in x]
        # )
        def lam(x):
            out = []
            for a in x:
                if a.startswith("[PAUSE"):
                    out.append(float(a.split()[1][:-1]))
                else:
                    out.append(0)
            return out

        df["labels"] = df["labels_str"].apply(lam)

    elif num_labels == 2:
        # 2値分類タスク
        # [PAUSE 0.5] などの文字列は 1 に、[NO_PAUSE] は 0 にする
        df["labels"] = df["labels_str"].apply(lambda x: [1 if a.startswith("[PAUSE") else 0 for a in x])
    elif num_labels == 3:
        # 多値分類タスク
        # [PAUSE 0.5] などの文字列のうち、数値部分が0以上なら 2、0未満なら 1、[NO_PAUSE] は 0 にする
        # TODO: これは正規化しており0は平均にあたるが、本来は、谷の部分でラベルを分けるように閾値を設定するべき
        # df["labels"] = df["labels_str"].apply(
        #     lambda x: [
        #         2 if float(re.findall(r"\d+\.\d+", a)[0]) >= 0 else 1 if a.startswith("[PAUSE") else 0 for a in x
        #     ]
        # )
        def lam(x):
            out = []
            for a in x:
                if a.startswith("[PAUSE"):
                    if float(a.split()[1][:-1]) >= 0:
                        out.append(2)
                    else:
                        out.append(1)
                else:
                    out.append(0)
            return out

        df["labels"] = df["labels_str"].apply(lam)
    else:
        raise ValueError("num_labels must be 1, 2, or 3")

    # texts or labels が空のデータがあるので、それを除外する
    df = df[df["texts"].apply(lambda x: len(x) > 0)]
    df = df.reset_index(drop=True)

    return df


def create_dataset(
    df,
    tokenizer,
    max_length,
    batch_size,
    test_size=0.2,
    val_size=0.25,
    random_state=42,
    return_df=False,
    num_labels=1,  # 1=回帰, 2=2値分類, 3=多値分類
):
    """データセットを作成する関数."""
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state
    )  # ex. 0.25 x 0.8 = 0.2

    # Extracting texts and labels for each dataset
    train_texts = train_df["texts"].tolist()
    train_labels = train_df["labels"].tolist()
    val_texts = val_df["texts"].tolist()
    val_labels = val_df["labels"].tolist()
    test_texts = test_df["texts"].tolist()
    test_labels = test_df["labels"].tolist()

    dataset_train = CreateDataset(train_texts, train_labels, tokenizer, max_length, num_labels)
    dataset_val = CreateDataset(val_texts, val_labels, tokenizer, max_length, num_labels)
    dataset_test = CreateDataset(test_texts, test_labels, tokenizer, max_length, num_labels)

    # データローダーの定義時にカスタムコラテ関数を使用
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    if return_df:
        return (
            dataloader_train,
            dataloader_val,
            dataloader_test,
            train_df,
            val_df,
            test_df,
        )

    return dataloader_train, dataloader_val, dataloader_test


class CreateDataset(Dataset):
    """データセットの作成."""

    def __init__(self, texts, labels, tokenizer, max_length, num_labels=1):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels  # 1=回帰, 2=2値分類, 3=多値分類

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        morphs = self.texts[index]
        label = self.labels[index]
        input_ids = [self.tokenizer.cls_token_id]
        boundary = [False]
        label_ids = [-100]  # [CLS] はラベルなし

        for m, la in zip(morphs, label):
            morph_toks = self.tokenizer.encode(m, add_special_tokens=False)
            input_ids.extend(morph_toks)
            boundary.extend([False] * (len(morph_toks) - 1) + [True])
            label_ids.extend([la] * len(morph_toks))

        input_ids.append(self.tokenizer.sep_token_id)
        boundary.append(False)
        label_ids.append(-100)  # [SEP] はラベルなし

        # パディングと切り捨て
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
            }
        else:
            # 分類タスク
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "boundary": torch.tensor(boundary, dtype=torch.bool),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }


# # 回帰のための損失関数 MSEloss
# def calculate_loss(outputs, labels, boundaries):
#     logits = outputs.logits
#     loss_fct = MSELoss()

#     # print(logits.shape, labels.shape, boundaries.shape)
#     # logitsを2次元に変形
#     # batch_size * max_length * 1 -> batch_size * max_length
#     # logits = logits.view(-1, logits.size(-1))
#     logits = logits.float().view(-1)
#     labels = labels.float().view(-1)

#     # 形態素の最後のトークンのみを損失計算に使用
#     # print(boundaries.view(-1))
#     active_loss = boundaries.view(-1)
#     active_logits = logits[active_loss]
#     active_labels = labels[active_loss]

#     # print(active_logits.shape, active_labels.shape)
#     # print(active_logits, active_labels)
#     # print("loss", loss_fct(active_logits, active_labels))

#     return loss_fct(active_logits, active_labels)


class CustomLoss:
    """損失関数のクラス."""

    def __init__(self, num_labels: int, loss_ignoring_no_pause=False):
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
        if loss_ignoring_no_pause:
            if num_labels != 1:
                raise ValueError("num_labels must be 1 when loss_ignoring_no_pause is True")
        self.loss_ignoring_no_pause = loss_ignoring_no_pause

    def __call__(self, outputs, labels, boundaries):
        logits = outputs.logits

        if self.num_labels == 1:
            # 回帰タスク
            # B=batch_size, L=max_length, C=1
            logits = logits.float().view(-1)  # [B, L, C] -> [B * L, C]
            labels = labels.float().view(-1)  # [B, L, C] -> [B * L, C]
            active_loss = boundaries.view(-1)  # [B, L] -> [B * L]
            active_logits = logits[active_loss]  # [B * L, C] -> [B * L, C]
            active_labels = labels[active_loss]  # [B * L, C] -> [B * L, C]
            if self.loss_ignoring_no_pause:
                # [NO_PAUSE] は損失計算に使用しない
                is_pause = active_labels != 0
                active_logits = active_logits[is_pause]
                active_labels = active_labels[is_pause]
            return self.loss_fct(active_logits, active_labels)  # [B * L, C], [B * L, C]
        else:
            # 分類タスク
            # B=batch_size, L=max_length, C=num_labels
            logits = logits.view(-1, self.num_labels)  # [B, L, C] -> [B * L, C]
            labels = labels.view(-1)  # [B, L] -> [B * L]
            active_loss = boundaries.view(-1)  # [B, L] -> [B * L]
            active_logits = logits[active_loss]  # [B * L, C] -> [B * L, C]
            active_labels = labels[active_loss]  # [B * L] -> [B * L]
            return self.loss_fct(active_logits, active_labels)  # [B * L, C], [B * L]


def evaluate_regression_model(preds, labels):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    return mae, mse, rmse, r2


# 分類の評価指標
def evaluate_classification_model(preds, labels):
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    return f1, acc


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


def train(model, dataloader, optimizer, device, loss_fn):
    """トレーニング用の関数."""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundary"].to(device)

        # print(input_ids.shape, attention_mask.shape, labels.shape, boundaries.shape)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels, boundaries)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, optimizer, device, loss_fn, num_labels=1):
    """評価用の関数."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            boundaries = batch["boundary"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels, boundaries)
            total_loss += loss.item()

            # 予測値とラベルの取得
            preds = outputs.logits.argmax(dim=-1)[boundaries]
            active_labels = labels[boundaries]
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(active_labels.cpu().numpy().flatten())

    if num_labels == 1:
        # 回帰タスク
        return None, None, total_loss / len(dataloader)
    else:
        # 分類タスク
        f1, acc = evaluate_classification_model(all_preds, all_labels)
        return f1, acc, total_loss / len(dataloader)


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


# 早期終了のためのクラス
class EarlyStopping:
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

    # def __call__(self, val_loss, model):
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # モデル保存はearly_stopがTrueになったときに外部で行うため、ここではコメントアウト
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"早期終了カウンター: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     if self.verbose:
    #         self.trace_func(f'検証損失が減少しました。 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  モデルを保存します...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss


def run_training(cfg):
    """トレーニングを実行する関数."""
    # 乱数シードの固定
    seed_everything(cfg.seed)

    # モデルとトークナイザの初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained(cfg.model.name)
    model = BertForTokenClassification.from_pretrained(cfg.model.name, num_labels=cfg.model.num_labels)
    device = torch.device(cfg.device, cfg.gpu_id)
    model.to(device)

    # オプティマイザの設定
    optimizer_grouped_parameters = set_optimizer_params(model, cfg.training.encoder_lr, cfg.training.decoder_lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.encoder_lr, eps=cfg.training.eps)

    # 早期終了の設定
    early_stopping = EarlyStopping(patience=cfg.training.patience, verbose=True)

    # 損失関数の設定
    loss_fn = CustomLoss(cfg.model.num_labels, cfg.training.loss_ignoring_no_pause)

    # データセットの作成
    df = data_load(
        pause_time_threshold_ms=cfg.dataset.pause_time_threshold_ms,
        preprocess_type=cfg.dataset.preprocess_type,
        num_labels=cfg.model.num_labels,
    )
    dataloader_train, dataloader_val, dataloader_test = create_dataset(
        df,
        tokenizer,
        cfg.training.max_length,
        cfg.training.batch_size,
        return_df=cfg.data.return_df,
        num_labels=cfg.model.num_labels,
    )

    # トレーニングと評価のメトリクスを保存するリスト
    training_losses = []
    validation_losses = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    if cfg.training.loss_ignoring_no_pause:
        run_name = (
            f"{str(cfg.dataset.pause_time_threshold_ms)}ms_{cfg.dataset.preprocess_type}_{cfg.model.num_labels}labels_ignoring_no_pause"
        )
    else:  
        run_name = (
            f"{str(cfg.dataset.pause_time_threshold_ms)}ms_{cfg.dataset.preprocess_type}_{cfg.model.num_labels}labels"
        )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("epochs", cfg.training.epochs)
        mlflow.log_param("max_length", cfg.training.max_length)
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
        # mlflow.log_param("return_df", cfg.data.return_df)
        mlflow.log_param("preprocess_type", cfg.dataset.preprocess_type)
        mlflow.log_param("pause_time_threshold_ms", cfg.dataset.pause_time_threshold_ms)
        mlflow.log_param("loss_ignoring_no_pause", cfg.training.loss_ignoring_no_pause)

        for epoch in range(cfg.training.epochs):
            train_loss = train(model, dataloader_train, optimizer, device, loss_fn)

            if cfg.model.num_labels == 1:
                # 回帰タスク
                _, _, val_loss = evaluate(model, dataloader_val, optimizer, device, loss_fn, cfg.model.num_labels)

                # メトリクスの保存
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation loss: {val_loss}"
                )
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            else:
                # 分類タスク
                val_f1, val_acc, val_loss = evaluate(
                    model, dataloader_val, optimizer, device, loss_fn, cfg.model.num_labels
                )

                # メトリクスの保存
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation loss: {val_loss} || Validation f1: {val_f1} || Validation acc: {val_acc}"
                )
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)

            # 早期終了判定
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("早期終了")
                break

        # テストデータでの評価
        # test_loss = evaluate(model, dataloader_test, optimizer, device)
        # print(f"Test loss: {test_loss}")
        # mlflow.log_metric("test_loss", test_loss)

        # 分類と回帰で評価関数を変えるのが大変なので、一旦コメントアウト
        # 予測値とラベルの取得
        # preds, labels = predict(model, dataloader_test, device)
        # mae, mse, rmse, r2 = evaluate_regression_model(preds, labels)
        # print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        # mlflow.log_metric("MAE", mae)
        # mlflow.log_metric("MSE", mse)
        # mlflow.log_metric("RMSE", rmse)
        # mlflow.log_metric("R2", r2)

        # if cfg.data.return_df:
        #     assert (
        #         len(preds) == len(labels) == len(test_df)
        #     ), f"予測値とラベルの長さが一致しません。 {len(preds)=}, {len(labels)=}, {len(test_df)=}"
        #     output_df = test_df.copy()
        #     output_df["output_preds"] = preds
        #     output_df["output_labels"] = labels
        #     output_df.to_pickle("output_test_df.pkl")
        #     mlflow.log_artifact("output_test_df.pkl")

        mlflow.pytorch.log_model(model, "model")

        # Hydraの成果物をArtifactに保存
        # log_dir = cfg.log_path
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/config.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/hydra.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/overrides.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, "main.log"))
