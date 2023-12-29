#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")
print(veriler)


#Yaş ve Hacim Kolonunu Alma İşlemi
X = veriler.iloc[:,2:4].values


#K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X) #Train

print(kmeans.cluster_centers_) #Merkez Noktaları


#K-means'te küme sayısını görselleştirerek belirleme
sonuclar = [] #wcss değerlerini bir diziye atılması
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()