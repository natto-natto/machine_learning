"""
分類課題をやりたい場合、最初は"ロジスティック回帰"もしくは"SVC"が一般的 ここでは"SVC"について扱っていく。 
"SVC"は訓練データをきれいに分離する境界線を学習でき、その上汎化性能が高いのが特徴。
"SVC"には線形に分離する"Liner SVC"と非線形に分離する"Kernel SVC"の2つある。
"Liner SVC"の中で訓練データに対して線形な境界線で誤分類なくきれいに訓練データを分離できるケースを"ハードマージンSVC"という。
逆に、誤分類を定めたペナルティの中である程度許すケースを"ソフトマージンSVC"という。
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

plt.style.use('seaborn-darkgrid')

#データの作成
x =np.zeros((20,2))
x[0:10, 1]= range(0,10)
x[10:20,1]= range(0,10)
x[0,0]=1.0
x[9,0]=1.0
x[1:9,0]=3.0
x[10:20,0]=range(-1,-11,-1)
x[9,0]=1
x[19,0]=-1
y=np.zeros((20))
y[10:20]=1.0
y= y.astype(np.int8)
print(x[:,0])
#描画
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
print(x)
print(y)
plt.show()

#結果を可視化させるためのモジュール
from mlxtend.plotting import plot_decision_regions

#変数xの標準化
sc = preprocessing.StandardScaler()
sc.fit(x)
x_std = sc.transform(x)
#Liner SVCの適用
from sklearn import svm
clf = svm.LinearSVC(random_state=0)
clf.fit(x_std, y)
plot_decision_regions(x_std,y,clf=clf,res=0.01)
plt.show()

x[19, 0]= 2
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
plt.show()

#変数xの標準化をする
sc = preprocessing.StandardScaler()
sc.fit(x)
x_std = sc.transform(x)

#LinerSVCの適用
clf = svm.LinearSVC(random_state=0)
clf.fit(x_std, y)
plot_decision_regions(x_std, y, clf=clf, res=0.01)

#ソフトマージンSVC(c=0.2)の適用
clf = svm.LinearSVC(C=0.2, random_state=0)
clf.fit(x_std,y)
plot_decision_regions(x_std,y,clf=clf,res=0.01)
