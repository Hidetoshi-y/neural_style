from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np

#定数の設定
img_height = 400


"""
VGG19ネットワークでやり取りされる画像の「読み込み」「前処理」「後処理」を行う補助関数
"""

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_img(x):
    # ImageNetから平均ピクセル値を取り除くことにより、中心を0に設定
    # これにより、vgg19.process_inputによって実行される変換が逆になる。
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    #画像を'BGR'から'RGB'に変換
    #これもvgg19.preprocess_inpuitの変換を逆にするための処置
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    """コンテンツの損失関数"""
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    """補助関数　元の特徴量行列から抽出された相関関係のマップのためにグラム行列にする。"""
    features = K.batch_flatten(k.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    """スタイルの損失関数を定義"""
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    """３つめの損失関数　全変動損失の定義　生成画像のピクセルを操作し、生成画像を空間的に連続させることで過度の画素化を回避　正則化損失関数である"""
    a = K.square(x[:, :img_height - 1, :img_width -1, ;] - x[:, 1:, :img_width -1, :])
    b = K.square(x[:, :img_height -1, :img_width -1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))



if __name__ == "__main__":
    
    #変換したいターゲット画像へのパス
    target_image_path = 'input/portrait.jpg'

    #ターゲット画像に適用させたいスタイル画像へのパス
    style_reference_img_path = 'input/transfer_style_reference.jpg'

    #生成する画像のサイズ
    width, height = load_img(target_image_path).size
    #print("変換するベース画像の横幅{0}, 縦幅{1}".format(width, height))
    img_width = int(width * img_height / height)

    """
    VGG19ネットワークを定義　
    リファレンス画像　ターゲット画像　生成画像　を保持するプレースホルダの３つを入力として使用
    プレースホルダは記号的なテンソルで、プレースホルダの値はNumPy配列を通じて外部から提供
    リファレンス画像とターゲット画像は静的な画像なので、K.constantを使ってプレースホルダを定義
    前者２つと違い生成画像のプレースホルダに含まれる値は除々に変化する。
    """

    #ターゲット画像とリファレンス画像のプレースホルダ
    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_img_path))

    #生成画像のプレースホルダ
    combination_image = K.placeholder((1, img_height, img_width, 3))

    #３つの画像を１つのバッチにまとめる
    input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

    #3つの画像からなるバッチを入力として使用するVGG19モデルを構築
    #このモデルには、学習済みImageNetの重みが読み込まれる

    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    print('Model loaded')

    """
    最小化の対象となる最終的な損失関数を定義
    """

    #層の名前を活性化テンソルにマッピングするディクショナリ
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    content_layer = 'block5_conv2' # コンテンツの損失関数に使用する層の名前

    style_layers = ['block1_conv1',
                    'blobk2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    
    #損失関数の荷重平均の重み
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    #すべてのコンポーネントをこのスカラー変数に追加することで、損失関数を定義
    loss = K.variable(0.)

    #コンテンツの損失関数を追加
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features, combination_features)

    #各ターゲット層のスタイルの損失関数を追加
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        lodd += (style_weight / len(style_layers)) * sl
    
    #全変動損失関数を追加
    loss += total_variation_weight * total_variation_loss(combination_image)

    """勾配降下のプロセスを定義
    """


    


    