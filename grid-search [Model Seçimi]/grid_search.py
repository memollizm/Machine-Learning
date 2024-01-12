# Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri Kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Eğitim ve Test Kümelerinin Bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#             *** GRID SEARCH *** 
# Parametremetre Optimizasyonu ve Algoritma Seçimi
from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear']},
     {'C':[1,2,3,4,5] ,'kernel':['rbf'],
      'gamma':[1,0.5,0.1,0.01,0.001]} ]

'''
Grid Search parametreleri
estimator : Optimize edilecek Sınıflandırma Algoritması
param_grid : Parametreler/ denenecekler
scoring: Hesaplanacak Skor (Accuracy)
cv : Kaç Katlamalı Olacağı
n_jobs : Aynı Anda Çalışacak İş
'''
gs = GridSearchCV(estimator= classifier, #SVM algoritması
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(X_train,y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_

print(eniyisonuc)
print(eniyiparametreler)

