import itertools
import json
import os
import pickle
import sys
import unicodedata
from pathlib import Path
from pprint import pprint

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForTokenClassification, BertJapaneseTokenizer


def data_load():
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

    import pickle

    with open(exp_dir / "bert_traindata_pause_position_with_length.pkl", "rb") as f:
        df = pickle.load(f)

    # 奇数番号の要素をtextとして、偶数番号の要素をlabelとして取得
    texts = []
    labels = []
    for a in df["morp_pause_clip_no_pause"].values:
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
    # [PAUSE] を 数値, [NO_PAUSE] を 0 に変換

    import re

    # [PAUSE 0.5] などの文字列から、0.5 の部分を取得
    df["labels"] = df["labels_str"].apply(
        lambda x: [float(re.findall(r"\d+\.\d+", a)[0]) if a.startswith("[PAUSE") else 0 for a in x]
    )

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
):
    """データセットを作成する関数."""
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state
    )  # 0.25 x 0.8 = 0.2

    # Extracting texts and labels for each dataset
    train_texts = train_df["texts"].tolist()
    train_labels = train_df["labels"].tolist()
    val_texts = val_df["texts"].tolist()
    val_labels = val_df["labels"].tolist()
    test_texts = test_df["texts"].tolist()
    test_labels = test_df["labels"].tolist()

    dataset_train = CreateDataset(train_texts, train_labels, tokenizer, max_length)
    dataset_val = CreateDataset(val_texts, val_labels, tokenizer, max_length)
    dataset_test = CreateDataset(test_texts, test_labels, tokenizer, max_length)

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

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        morphs = self.texts[index]
        label = self.labels[index]
        input_ids = [self.tokenizer.cls_token_id]
        boundary = [False]
        label_ids = [-100]  # [CLS] はラベルなし

        for m, l in zip(morphs, label):
            morph_toks = self.tokenizer.encode(m, add_special_tokens=False)
            input_ids.extend(morph_toks)
            boundary.extend([False] * (len(morph_toks) - 1) + [True])
            label_ids.extend([l] * len(morph_toks))

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

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "boundary": torch.tensor(boundary, dtype=torch.bool),
            # 'labels': torch.tensor(label_ids, dtype=torch.long)
            "labels": torch.tensor(label_ids, dtype=torch.float),
        }


# 回帰のための損失関数 MSEloss
def calculate_loss(outputs, labels, boundaries):
    logits = outputs.logits
    loss_fct = MSELoss()

    # print(logits.shape, labels.shape, boundaries.shape)
    # logitsを2次元に変形
    # batch_size * max_length * 1 -> batch_size * max_length
    # logits = logits.view(-1, logits.size(-1))
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    # 形態素の最後のトークンのみを損失計算に使用
    # print(boundaries.view(-1))
    active_loss = boundaries.view(-1)
    active_logits = logits[active_loss]
    active_labels = labels[active_loss]

    # print(active_logits.shape, active_labels.shape)
    # print(active_logits, active_labels)
    # print("loss", loss_fct(active_logits, active_labels))

    return loss_fct(active_logits, active_labels)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(preds, labels):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    return mae, mse, rmse, r2


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


def train(model, dataloader, optimizer, device):
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
        loss = calculate_loss(outputs, labels, boundaries)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, optimizer, device):
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
            loss = calculate_loss(outputs, labels, boundaries)
            total_loss += loss.item()

            # 予測値とラベルの取得
            preds = outputs.logits.argmax(dim=-1)[boundaries]
            active_labels = labels[boundaries]
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(active_labels.cpu().numpy().flatten())

    return total_loss / len(dataloader)


def predict(model, dataloader, device):
    """予測値とラベルを取得する関数."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            boundaries = batch["boundary"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            # 予測値とラベルの取得
            logits = outputs.logits.float().squeeze().cpu().numpy()
            labels = labels.float().squeeze().cpu().numpy()
            boundaries = boundaries.cpu().numpy()
            for i, (logit, label, boundary) in enumerate(zip(logits, labels, boundaries)):
                active_logit = logit[boundary]
                active_label = label[boundary]
                all_preds.append(active_logit)
                all_labels.append(active_label)

    return all_preds, all_labels


def run_training(cfg):
    """トレーニングを実行する関数."""
    # モデルとトークナイザの初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained(cfg.model.name)
    model = BertForTokenClassification.from_pretrained(cfg.model.name, num_labels=1)
    device = torch.device(cfg.device, cfg.gpu_id)
    model.to(device)

    # オプティマイザの設定
    optimizer_grouped_parameters = set_optimizer_params(model, cfg.training.encoder_lr, cfg.training.decoder_lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.encoder_lr, eps=cfg.training.eps)

    # データセットの作成
    df = data_load()
    dataloader_train, dataloader_val, dataloader_test, train_df, val_df, test_df = create_dataset(
        df, tokenizer, cfg.training.max_length, cfg.training.batch_size, return_df=True
    )

    # トレーニングと評価のメトリクスを保存するリスト
    training_losses = []
    validation_losses = []

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_param("epochs", cfg.training.epochs)
        mlflow.log_param("max_length", cfg.training.max_length)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("encoder_lr", cfg.training.encoder_lr)
        mlflow.log_param("decoder_lr", cfg.training.decoder_lr)
        mlflow.log_param("eps", cfg.training.eps)
        mlflow.log_param("optimizer", optimizer)  # AdamW
        mlflow.log_param("MODEL_NAME", cfg.model.name)  # cl-tohoku/bert-base-japanese-whole-word-masking
        mlflow.log_param("device", device)  # cuda or cpu

        for epoch in range(cfg.training.epochs):
            train_loss = train(model, dataloader_train, optimizer, device)
            val_loss = evaluate(model, dataloader_val, optimizer, device)

            # メトリクスの保存
            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{cfg.training.epochs} || Training loss: {train_loss} || Validation loss: {val_loss}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # テストデータでの評価
        # test_loss = evaluate(model, dataloader_test, optimizer, device)
        # print(f"Test loss: {test_loss}")
        # mlflow.log_metric("test_loss", test_loss)
        # 予測値とラベルの取得
        preds, labels = predict(model, dataloader_test, device)
        mae, mse, rmse, r2 = evaluate_regression_model(preds, labels)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        if cfg.data.return_df:
            assert (
                len(preds) == len(labels) == len(test_df)
            ), f"予測値とラベルの長さが一致しません。 {len(preds)=}, {len(labels)=}, {len(test_df)=}"
            output_df = test_df.copy()
            output_df["output_preds"] = preds
            output_df["output_labels"] = labels
            output_df.to_pickle("output_test_df.pkl")
            mlflow.log_artifact("output_test_df.pkl")

        mlflow.pytorch.log_model(model, "model")

        # Hydraの成果物をArtifactに保存
        # log_dir = cfg.log_path
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/config.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/hydra.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, ".hydra/overrides.yaml"))
        # mlflow.log_artifact(os.path.join(log_dir, "main.log"))
