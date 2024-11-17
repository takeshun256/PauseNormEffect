# BERTモデルの訓練

## 概要
BERTモデルの訓練を行うための実験ディレクトリです。
実験には、hydraとmlflowを使用しています。

> [!WARNING]
> 多くの実験を行っており、それぞれの実験でコードの重複が発生しています。現在、コードの共通化を行っています。 @2024/11/18
> 論文の実験には、`bert_wo_lstm`, `bert_lstm`, `embbedding`のディレクトリを主に使用しています。


## ディレクトリ構成

```bash
├── bert_finetuning_many_models_and_many_preprocesses # モデルx前処理の組み合わせでの実験（未整理）
├── bert_lstm # BERTモデルにLSTMを追加したモデルの標準化の実験
├── bert_wo_lstm # BERTモデルのみの標準化の実験
└── embbedding # BERT/BERT-LSTMモデルの埋め込みの実験
```