

## 開発について

### コード構成

- `src/`
  - メインのコード
  - `pausenormeffect`パッケージを作成
- `research/`
  - 研究用のコード
  - 主にノートブック形式で記述
  - ほぼ未整理, 古いコードも含まれる
- `scripts/`
  - 実験用のスクリプト置き場
- `makefile`
  - 環境構築や管理用のスクリプト


### 環境

- 環境の有効化
  - 依存関係は`environment.yml`に記述
```bash
conda acitvate pause_norm_effect
```

- 環境設定ファイルの更新
```bash
make conda-env-export
(conda env update --file environment.yml) # これは試していない
```

- format & lint
  - `black`と`ruff`を使用
  - makeコマンドで実行
  - 設定は`pyproject.toml`に記述
```bash
make format
make lint
```

- vscode設定
  - `settings.json`に、`format`と `lint` について記述
  - `extensions.json`に、使用する拡張機能を記述

- submoduleの更新
  - `git submodule update --init --recursive`
  - （submoduleはPrivateで以下を追加して実験していました. @20241118）
    - git@github.com:Hiroshiba/jvs_hiho.git
    - git@github.com:Syuparn/TextGridConverter.git
    - git@github.com:julius-speech/segmentation-kit.git
    - git@github.com:espnet/espnet.git


### 使用データ

Shinnosuke Takamichi, Wataru Nakata, Naoko Tanji, and Hiroshi Saruwatari. J-MAC: Japanese multispeaker audiobook corpus for speech synthesis. In Proc. Interspeech 2022, pp. 2358–2362, 2022.

