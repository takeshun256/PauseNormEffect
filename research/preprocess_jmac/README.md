# J-MACコーパスの前処理

## 概要

```shell
.
├── README.md
│── サブトークン数が95%の長さを計測.ipynb # サブトークン数の統計
├── get_pause_ranges.ipynb # old, runcodeの修正前
├── get_pause_ranges_fix_runcode.ipynb # 無音区間の情報を保存, runcodeの修正後
├── mapping_morphone_phoneme.ipynb # 形態素labとポーズ情報のマッピングして、形態素単位ポーズ情報の作成
├── text_data_convert.ipynb # wakatiの変換(未使用)
├── vallidate_fix_align.ipynb # fix_alignの検証
├── z_normalize_pause_length_平均分散追加.ipynb # 文中ポーズ長の正規化(平均分散を追加で出力)
├── 音声データを正規化.ipynb # 未使用
├── 学習データの作成.ipynb # 文中ポーズの学習用データの作成, ラベル未処理 => ラベル処理済み
└── 文間ポーズの学習データ作成.ipynb # 文間ポーズの学習用データの作成 # 未標準化 => 標準化 + ラベル処理済み
```


## 学習データのパイプライン(記憶)

1. ポーズ情報の取得 get_pause_ranges_fix_runcode.ipynb
2. 形態素labファイルの作成 ()
3. 形態素labファイルとポーズ情報のマッピング mapping_morphone_phoneme.ipynb
4. ポーズ長の正規化 z_normalize_pause_length_平均分散追加.ipynb
5. (文中)学習データの作成 学習データの作成.ipynb
6. (文間)学習データの作成 文間ポーズの学習データ作成.ipynb
