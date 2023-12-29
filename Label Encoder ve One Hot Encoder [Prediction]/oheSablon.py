#Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('veriler.csv')

Yas = veriler[['boy']]

ulke = veriler.iloc[:,0:1].values


#Label Encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

#OneHotEncoder
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


c = veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)

#OneHotEncoder
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c, index = range(22), columns = ['cinsiyet']) 
print(sonuc3)


s = pd.concat([sonuc, sonuc2], axis = 1)
print(s)

s2 = pd.concat([s, sonuc3], axis = 1)
print(s2)


from sklearn.linear_model import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 
#x_train'den y_traini öğren(Aralarında bir bağlantı)
#x_train input; y_train output

y_pred = regressor.predict(x_test) 
#x_test'e tahmin burada yapılıyor


#Veri manipülasyon
boy = s2.iloc[:3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol, sag], axis = 1)


#Modelin ve Modeldeki değişkenleri başarısı ile ilgili
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())
