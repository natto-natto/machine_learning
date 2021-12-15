"""
scikit-learnは機械学習ライブラリ
scikit-learnはNumPy,pandasなどの他のライブラリとの連携もしやすくなっている
機械学習においてデータを使ってモデルを訓練するまでに以下の5つのステップがよく共通して現れる
step1:データセットの準備
step2:モデルを決める
step3:目的関数を決める
step4:最適化手法を選択する
step5:モデルを訓練する
"""

"""
step1:データセットの準備
データセットの準備では訓練を行っていくデータをセットすることを行う
初学者はまずscikit-learnによって用意されているデータセットを利用して学習すると良い。
そのデータセットは二つに大きく分けることが出来る。
1,scikit-learnに同梱されていてすぐにインポート出来るサイズの小さなデータ(Toy Datasets)
2,ダウンロードすることで使えるサイズの大きいデータ(Real World Datasets)
である。
今回はToy Datasetsの「load_diabetes()」を使用する。
(load_boston()で勉強したため別のデータセットを使用する)
さて、実際にコードを書いていく
load_diabetes()のインポートは次の文で可能
"""

from sklearn.datasets import load_diabetes

"""
また、今回はpandasをインポートしてpandasのDataFrameを利用します。
そうすることによってデータの可視化がされて見やすいです。
以下の文はその他必要なライブラリを呼び出している
"""


