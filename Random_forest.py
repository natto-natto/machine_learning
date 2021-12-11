#ランダムフォレストはアンサンブル手法
#（=複数の学習モデルを組み合わせてより強力な学習モデルを構築するアプローチ）の一つ
#引き続きここではwineのデータセットを用いて学習していく
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#データセットの準備
wine = load_wine()
X_train, X_test, Y_train, Y_test= train_test_split(wine.data, wine.target, random_state=41)

#ランダムフォレストの適用(決定木の数を7,特徴量の数を3,分割基準をジニ不純度に設定)
forest = RandomForestClassifier(n_estimators=7, max_features=3, max_depth=3,
                                criterion='gini', random_state=41)
forest.fit(X_train,Y_train)

#Accuracy(正確さ)の表示
print(f'Train Accuracy:{forest.score(X_train,Y_train):.3f}')
print(f'Test Accuracy:{forest.score(X_test, Y_test):.3f}')

"""
決定木の頃と比べると明らかだが過学習、つまりtrain dataからtest dataへの
Accuracy（正確さ）の数値の差が抑えられている。
決定木と比べて過学習は抑えられるがモデルの解釈性は低下してしまう
しかし、決定木の頃と同様に禿頭量の重要度を定量化させることは可能
"""

#以下、学習したランダムフォレストにおける特徴量の重要度の可視化
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

n_features = wine.data.shape[1]
plt.title('feature Importances')
plt.bar(range(n_features), forest.feature_importances_, align='center')
plt.xticks(range(n_features), wine.feature_names, rotation=90)
plt.xlim([-1, X_train.shape[-1]])
plt.show()
