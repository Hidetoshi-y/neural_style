# neural_style

## 内容物 
- README.md 

## やりたいこと
Neural_styleを使って画像を新海誠風に変換する。

## 参考文献

PythonとKerasによるディープラーニング
https://www.amazon.co.jp/Python%E3%81%A8Keras%E3%81%AB%E3%82%88%E3%82%8B%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0-Francois-Chollet-ebook/dp/B07D498RJK

ｐ303より

## 参考文献からの解釈

DeepLearningベースの画像加工における開発の１つ

２０１５年夏 LeonGatys他によって提唱されたのがneural style transfer(=ニューラルスタイル変換)

neural style transferは様々なバージョンを経て派生している。

今回は２０１５年夏に投稿された論文にしたがって実施する。

neural style transferはターゲット画像のコンテンツを維持したうえで、リファレンス画像の
styleをターゲット画像に適用するというもの。

スタイルはゴッホ調だの　という感じ

コンテンツは画像の俯瞰的なマクロ構造(?)を意味する。

従来のコンピュータビジョンの手法によるスタイル変換よりDLベースのスタイル変換のほうが良い

スタイル変換実装の考え方は　「何を達成したいのかを指定するための損失関数を定義」「この損失関数を最小化」

何を達成するのか　→　元の画像のコンテンツを維持した上で、リファレンス画像のスタイルを取り入れる。

コンテンツとスタイルを数学的に定義することが可能である場合に最小化の対象にすべき損失関数は
次のように定義できる。

loss = distance(style(リファレンス） - ｓｔｙle(作られた画像))　＋　distance(content(元画像) - content(作られた画像))

が損失関数となる。

作られた画像のスタイルがリファレンスのスタイルに対して差が小さく、元画像のコンテンツと作られた画像のコンテンツの
差が小さければlossは小さくなるので誤差が最小となる。　このようになるように誤差を小さくしていく。

元画像のコンテンツとリファレンス画像のスタイルとの差が小さい画像を作ることができれば、リファレンス画像のスタイル調の画像を生成することができる。

登場する関数が３つあり
- distance L２ノルムなどのノルム関数
- content 画像からそのコンテンツの表現を計算する関数
- style 画像からそのスタイルの表現を計算する関数

LeonGatysらが提案したのは、「style関数とcontent関数を数学的に定義する手段がDLのCNNによって提供されるということである。」

### Contentの損失関数を数学的に表現できる理由   

CNNでは、入力層に近い活性化では入力画像の局所的な情報を持っており、出力層に近い活性化では、入力画像の
大域的で抽象的な情報を持っている。

つまりCNNの出力層付近の表現は入力画像のコンテンツを補足したものになることが期待されている。

L２ノルムは、CNNの学習済みモデルで　ターゲット画像で計算された出力層の活性化　と　生成した画像で計算された
同じ層の活性化との距離を表すものである。

このため、CNNを出力層側から見たとき生成画像がターゲット画像と同じように見えることが保証されている。

CNNの出力側の層が見えるものが入力画像のコンテンツであるならば、画像のコンテンツを維持するための手段として
コンテンツの損失関数の候補としてL2ノルムが有力である。

CNNの出力側の層ではコンテンツが維持されるであろうというところから画像のコンテンツを導き出すのに
L2ノルムが使えるよ　と発見したのである。

### styleの損失関数を数学的に表現できる理由

ターゲット画像のコンテンツを補足するには、出力側の層のL2ノルムによって損失関数を定義した。

リファレンス画像のスタイルを補足するために、CNNの複数の層で損失関数を定義する。

リファレンス画像からスタイルを補足するには、CNNによってすべての空間規模でリファレンス画像のスタイルを
補足する必要がある。

このスタイルの損失関数をGatysらは、層の活性化のグラム行列を使用してた。

グラム行列は、与えられた層の特徴マップ同士の内政である。

この内積は、その層の特徴量同士の相関係数を表すマップとして考えられている。

特徴量同士の相関関係には特定の空間規模での統計パターンが反映されている。

特定の空間規模での統計パターンは、空間規模で抽出されるテクスチャの外観に対応している。

グラム行列によって相関関係がわかることで、テクスチャのようなものが理解できる。

スタイルの損失関数の目的は　リファレンス画像　と　生成画像　とで様々な層の活性化に含まれる
相関関係を同じに保つことである。

特定の空間規模で抽出されたテクスチャが、リファレンス画像でも生成画像でも同じように見えることが保証される。

学習済みCNNを使うことでこれら２つの損失関数(contentとstyle)が定義できる。

### neural style transferに必要な損失関数

- コンテンツを反映させたいターゲット画像　と　生成画像　との間で出力側の層の活性化を
同様に保つことで、コンテンツを維持する。　CNNからはターゲット画像と生成画像が同じものを含んでいるように「見えるはず」である。

- 入力側の層と出力側の層の両方で、活性化の「相関係数」を同じに保つことで、スタイルを維持する。
特徴量の相関係数はテクスチャを補足できる　リファレンス画像と生成画像のスタイル(テクスチャ)は空間規模で
同じになるはずである。

### 8.3.3
2015年のニューラルスタイル変換アルゴリズムをKerasで実装する。
