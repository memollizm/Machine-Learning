#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri Yükleme
veriler = pd.read_csv('maaslar.csv')
print(veriler)


#Linear Regression [Data Frame / Dilimleme]
x = veriler.iloc[:,1:2] #Eğitim seviyesi sütununu aldık
y = veriler.iloc[:,2:] #Maaş sütununu aldık

#Numpy dizi dönüşümü
X = x.values
Y = y.values

#Linear Regression [Doğrusal Model Oluşturma]
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X,Y)


#Polinomal Regresyon
#Doğrusal Olmayan Model Oluşturma [2. dereceden polinom]
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) 
x_poly = poly_reg.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly, y) #x_poly'ye göre y'yi öğren


#tahminler
print(le.predict([[11]]))
print(lin_reg3.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg.fit_transform([[11]])))


#Görselleştirme
plt.scatter(X,Y, color = 'red')
plt.plot(x, le.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X, lin_reg3.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()
