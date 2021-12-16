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
(load_boston()は1.2で削除されるため非推奨)
さて、実際にコードを書いていく
load_diabetes()のインポートは次の文で可能
"""

from sklearn.datasets import load_diabetes

"""
また、今回はpandasをインポートしてpandasのDataFrameを利用します。
そうすることによってデータの可視化がされて見やすいです。
以下の文はその他必要なライブラリを呼び出している
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame

import matplotlib.pyplot as plt

%matplotlib inline

"""
今データセットを呼び出した形になっています。
次にそれを変数に入れます。
"""

dataset = load_diabetes()

"""
上で定義したdatasetをprintを適用させると様々な情報が出てくる。
しかし、よくわからないと感じると思うのでデータセットの説明が欲しいと感じる時があると思う
そんなときはprint(dataset.DESCR)を使用するとよい
また、よくわからないと感じたので可視化が有効である。
ここでpandasのDataFrameを使用します
"""

#dataframeの引数にdataset.dataを入れる
diabetes_dataframe = DataFrame(dataset.data)
#上の文章だけだと何が何のデータを示しているのかわからない。そこでcolumnsを追加してあげる
diabetes_dataframe.columns = dataset.feature_names
#目的関数もここで追加するとよい
diabetes_dataframe['Diabetes'] = DataFrame(dataset.target)

#csvファイルを作りたい場合は.to_csv()を利用するとよい。(拡張子をcsvにすることに注意)
diabetes_dataframe.to_csv("diabetes.csv")

#csvの中身は以下の文で見ることが可能
diabetes_dataframe.head()

from sklearn.model_selection import train_test_split
x = dataset.data
t = dataset.target
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
# モデルの定義
reg_model = LinearRegression()
reg_model.fit(x_train, t_train)
# 訓練後のパラメータ w
reg_model.coef_
# 精度の検証
reg_model.score(x_train, t_train)
