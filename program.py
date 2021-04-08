#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:21:53 2021

@author: grimaldo
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import Axes3D

#matplotlib inline

data = pd.read_csv("houses_to_rent_v2.csv")
#print(data.head())

plt.figure()
data['city'].value_counts().plot(kind='bar')
plt.title("Dados por cidade")
plt.xlabel("Cidade")
plt.ylabel("Qtd")
sns.despine()

plt.figure()
plt.scatter(data['rooms'], data['rent amount (R$)'])
plt.xlabel("Número de quartos")
plt.ylabel("Valor do aluguel (R$)")
plt.title("Exemplo de como relação entre as variáveis \n não obedece um modelo puramente linear")

cityname_cityids = np.unique(data['city'], return_inverse=True)
furnishedids     = np.unique(data['furniture'], return_inverse=True)
animalids = np.unique(data['animal'], return_inverse=True)

data['city'] = cityname_cityids[1]
data['furniture'] = furnishedids[1]
data['animal'] = animalids[1]

floor = [float(np.char.replace(f, '-', '0')) for f in data['floor']]
data['floor'] = floor

labels = data['total (R$)']
filtered_data = data.drop(['hoa (R$)', 'total (R$)', 'rent amount (R$)'], axis=1)

    
print(filtered_data.head(), filtered_data.columns)

xt, xtest, yt, ytest = train_test_split(filtered_data, labels, test_size = 0.15, random_state=42)

print(filtered_data)

start = timer()
gbr = GradientBoostingRegressor(n_estimators=290, max_depth=1, learning_rate=0.05, loss='ls').fit(filtered_data, labels).fit(xt, yt)
end = timer()
print(end - start)

predicted_totalprice = gbr.predict(xtest)
score = gbr.score(xtest, ytest)
print("R² = {:.1f}".format(score))

#score = r2_score(ytest, predicted_totalprice)
#print(score)

plt.figure()
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(filtered_data.columns)[sorted_idx])
plt.title('Importância de característica')

#[1, x, y, 'bathroom', 'parking spaces', 'floor',
#       'animal', 'furniture', 'property tax (R$)', 'fire insurance (R$)'


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
city = 0
X = np.linspace(1.0, 4, 20)
rooms = 3
Y = np.linspace(80, 400, 20)
bathroom = 2 
parking = 0
floor = 3 
animal = 0
furniture = 0
tax = 300 
fire = 50


X, Y = np.meshgrid(X, Y)
#print(X.shape)
#print(X[10, 10], Y)
Z = np.ndarray(shape=(20, 20))
for i in range(20):
    for j in range(20):
        single_sample = np.array([city, Y[j, i], rooms, X[j, i], parking, floor, animal, furniture, tax, fire])
        ss = [single_sample]
        #print(ss.shape)
        #print ("SS ", ss)
        #print([single_sample].shape)
        Z[j, i] = gbr.predict(ss)
   
        
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                       linewidth=0, antialiased=False)
ax.set_xlabel("Qtd. Banheiros")
ax.set_ylabel("Area da casa (m²)")
ax.set_zlabel("Aluguel (R$)")
ax.set_title("Relação entre qtd de banheiros, area de casa e o impacto no aluguel")
   

                               # result = permutation_importance(gbr, xtest, ytest, n_repeats=10,
#                                 random_state=42, n_jobs=2)
# sorted_idx = result.importances_mean.argsort()
# plt.subplot(1, 2, 2)
# plt.boxplot(result.importances[sorted_idx].T,
#             vert=False, labels=np.array(filtered_data.columns)[sorted_idx])
# plt.title("Importância de permutação (base de testes)")
# fig.tight_layout()
# plt.show()
#print(d)
    
# plt.figure()
# plt.scatter(data['area'], data['total (R$)'])
# plt.xlabel("Area")
# plt.ylabel("Valor (R$)")

#reg = LinearRegression()