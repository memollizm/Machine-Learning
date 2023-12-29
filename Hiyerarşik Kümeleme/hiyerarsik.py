#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")
print(veriler)


#Yaş ve Hacim Kolonunu Alma İşlemi
X = veriler.iloc[:,2:4].values

#Hiyerarşik Kümeleme [Agglomerative]
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin == 0,0], X[Y_tahmin == 0,1], s = 100, c = 'red')
plt.scatter(X[Y_tahmin == 1,0], X[Y_tahmin == 1,1], s = 100, c = 'blue')
plt.scatter(X[Y_tahmin == 2,0], X[Y_tahmin == 2,1], s = 100, c = 'green')
plt.scatter(X[Y_tahmin == 3,0], X[Y_tahmin == 3,1], s = 100, c = 'yellow')
plt.show()


#Dendogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()
