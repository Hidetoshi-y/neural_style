# neural_style

add
## how to use
###　環境構築
pythonの仮想環境にて作成　仮想環境になっているものとして必要なパッケージをインストール

1. パッケージのインストール
`./setup.sh`
or
`pip install -r requirements.txt`

※GPUあり環境でKerasを用いているので、GPU用が使えるならばtensorflowをGPU版にする。 setup.shのコメントアウトを通常のtensorflow
にするとCPU版のtensorflowになる。

2. 生成画像を保存するディレクトリを作る
`mkdir output`

3. 同じディレクトリに「ターゲット画像」「スタイルリファレンス画像」を置く
`python main.py target.jpg reference.jpg`





## 内容物 
- README.md 
- setup.sh
- main.py
- requirement.txt GPU用のtensorflowが入っている。
- NOTE.md　参考文献を自分なりに解釈した。

## やりたいこと
Neural_styleを使って画像を新海誠風に変換する。

## 参考文献

PythonとKerasによるディープラーニング
https://www.amazon.co.jp/Python%E3%81%A8Keras%E3%81%AB%E3%82%88%E3%82%8B%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0-Francois-Chollet-ebook/dp/B07D498RJK

ｐ303より


## Web Pageについて
Web
ページで