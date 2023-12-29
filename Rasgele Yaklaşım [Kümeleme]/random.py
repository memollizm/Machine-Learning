#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Verinin Okunması
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection (Rasgele Seçim)
N = 10000  #İlana tıklama sayısı
d = 10     #Veri setindeki ilan sayısı
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()

