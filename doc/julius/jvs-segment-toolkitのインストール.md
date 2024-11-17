# jvs_segment_toolkitのインストール（研究室サーバー）

## jvs_segment_toolkitのインストール

- [Praatに変換する](https://qiita.com/Syuparn/items/d86e77c39527539b16c5)
  - Juliusのツールキットの保存先の変更方法が書かれている
- [Julius公式レポジトリ](https://github.com/julius-speech/julius)


### Juliusのインストール

```bash
# 作業ディレクトリを用意
mkdir Julius-Test
cd Julius-Test

# 必要なpackageをチェックする
# https://github.com/julius-speech/juliusに記載のライブラリを参考
vi check_packages.sh
# vscodeの場合、touch check_packages.sh; code check_packages.sh
-----------------------------------
#!/bin/bash

packages=("build-essential" "zlib1g-dev" "libsdl2-dev" "libasound2-dev")

for package in "${packages[@]}"; do
  if dpkg-query -W -f='${Status}' $package 2>/dev/null | grep -q "ok installed"; then
    echo "$package is installed."
  else
    echo "$package is NOT installed."
  fi
done
------------------------------------

chmod +x check_packages.sh
./check_packages.sh

# 最新版ダウンロード

# wget https://github.com/julius-speech/julius/archive/v4.6.tar.gz
wget https://github.com/julius-speech/julius/archive/refs/tags/v4.6.tar.gz

# 解答
# x:extract, z: gzip, f: ファイルを指定, v:詳細表示
tar xvzf ./v4.6.tar.gz
rm v4.6.tar.gz

# インストール
cd ./julius-4.6/
./configure --prefix=$HOME/local --enable-words-int 2>&1 | tee configure_run.log # やり直したい場合は、make clean or make distclean -> 定義見た感じ、完全に取り消したい場合は、distcleanの方実行
make -j4 2>&1 | tee make_run.log

# 研究室のサーバーのためインストールはしない
# sudo make install # Makefileにmake installの記載あり、install.txtにはoption指定あり、システム下にbinを作成するので研究室サーバーでやるには危険かも
make install 2>&1 | tee make_install_run.log

ls -l julius/julius

# インストール確認
julius/julius -help # 実行可能=正常出力,正し、cd julius; julius -helpは不可だった
```

- HOME/localのパスを追加
	- bash_profileに追加 or bashrcに追加
	- bash_profileなかったのでbashrcに書く
```python
export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```
- 反映
```python
source ~/.bashrc
#source ~/.bash_profile
```


### ディクテーションキットをインストール

- 現在最新は、4.5らしい
	- [Julius音声認識パッケージ](https://julius.osdn.jp/index.php?q=dictation-kit.html)
- 任意の場所に解凍する：
```python
cd ..
ls check_packages.sh  julius-4.6  v4.6.tar.gz


# ディクテーションキットをダウンロードする : 以下から最新版(4.5)を確認してダウンロードした
# https://julius.osdn.jp/index.php?q=dictation-kit.htmlからリンクのアドレスをコピーする
wget https://osdn.net/dl/julius/dictation-kit-4.5.zip
# wget https://osdn.net/frs/redir.php?m=rwthaachen&f=julius%2F71011%2Fdictation-kit-4.5.zip HTTPの要求エラー
# おそらく、リダイレクトのリンクなので、正しいリンクではなかったため


# ファイルサイズは 478MB 程度だった。解凍する
$ unzip ./dictation-kit-4.5.zip

# ディクテーションキットのディレクトリ内に移動する
$ cd ./dictation-kit-4.5/

ls model/
```


### セグメンテーションの導入


リンク
- GitHub - julius-speech/segmentation-kit: Speech Segmentation Toolkit using Julius](https://github.com/julius-speech/segmentation-kit)
- [公式ドキュメント](https://julius.osdn.jp/index.php?q=ouyoukit.html)
- 導入記事
	- [Juliusによる音素アライメント(音素セグメンテーション) on MacOSX(Yosemite) - よーぐるとのブログ](https://yoghurt1131.hatenablog.com/entry/2016/01/01/212528)
- 
##### clone

- perlが入っているか確認:`perl -v`


```python
cd Julius-Test
git clone git@github.com:julius-speech/segmentation-kit.git
cd segmentation-kit 

# 実行パスの修正
which julius #/home/takeshun256/local/bin/julius
code segment_julius.pl
# l52のパスをjuliusのパスに修正
```

### 簡単な実行確認

- segmentation-kit /wav下にある、sampleで動作確認
```python
cat wav/sample.txt 
perl segment_julius.pl wav # plファイルを確認したところ、wavとtxtの入ったフォルダを指定すると全てのwavについてlabを出力するらしい
```
