#必要なライブラリのインポート
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#データセットの準備
wine = load_wine()
X_train, X_test, y_train, y_test=train_test_split(wine.data, wine.target,random_state=41)

#決定木の適用（木の深さの制限なし、分割基準をジニ不純度に設定）
tree =DecisionTreeClassifier(max_depth=None, criterion='gini', random_state=41)
tree.fit(X_train, y_train)

print(f'Train Accuracy:{tree.score(X_train, y_train):.3f}')
print(f'Test Accuracy:{tree.score(X_test, y_test):.3f}')

#ここから構築した決定木を可視化していく
#必要なライブラリのインポート
import graphviz
from sklearn.tree import export_graphviz

#graphviz形式で決定木をエクスポート
dot_data = export_graphviz(tree, out_file=None, impurity=False, filled=True,
                           feature_names= wine.feature_names,
                           class_names=wine.target_names)

#Graphviz形式の決定木を表示
graph = graphviz.Source(dot_data)
graph

"""
次に決定木の適用の時点でmax_depthを設定しておいて事前枝刈りを行っていく。
木が深くなりすぎないことによって過学習をおさえることが出来る。
"""
#決定木の適用(木の深さを3、分割基準をジニ不純度に設定)
tree=DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=41)
tree.fit(X_train, y_train)

#Accuracyの表示
print(f'Train Accuracy:{tree.score(X_train, y_train):.3f}')
print(f'Test Accuracy:{tree.score(X_test, y_test):.3f}')

"""
決定木は、訓練データで構築した決定木の構造から、各特徴量の重要度を定量化できる。
それをMatplotlibで可視化していく（むっちゃ便利だね～）
"""
#必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

#特徴量の重要度を可視化していくぜ
n_features = wine.data.shape[1]
plt.title('Feature Importances')
plt.bar(range(n_features), tree.feature_importances_, align='center')
plt.xticks(range(n_features), wine.feature_names, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
