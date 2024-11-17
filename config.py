import os

from dotenv import load_dotenv

# .env ファイルをロード
load_dotenv()

# 環境変数を読み込む
ROOT = os.getenv("ROOT")

# データセットのルートディレクトリ
DATA_DIR = os.getenv("DATA_DIR", "/data2")

# コード類のルートディレクトリ
SCRIPT_DIR = os.getenv("SCRIPT_DIR", os.path.join(ROOT, "scripts"))
SRC_DIR = os.getenv("SRC_DIR", os.path.join(ROOT, "src"))
RESEARCH_DIR = os.getenv("RESEARCH_DIR", os.path.join(ROOT, "research"))

# 出力ファイルのルートディレクトリ
LOG_DIR = os.getenv("LOG_DIR", os.path.join(ROOT, "logs"))
