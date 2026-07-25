# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:02:36 2026

@author: vinay agrawal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\EDA-Project\Salary_Data.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
# %%
regressor = LinearRegression()
# %%


regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('yers of experience')
plt.ylabel('Salary')
plt.show()

m_coeff = regressor.coef_
print(m_coeff)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_coeff *12 + c_intercept
print(y_12)

y_20 = m_coeff *20 + c_intercept
print(y_20)

bias_score = regressor.score(x_train, y_train)
print(bias_score)

variance_score = regressor.score(x_test, y_test)
print(variance_score)


#statistics integration lets implement stats to this model
dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

dataset.corr()
dataset.skew()

dataset['Salary'].skew()

dataset.sem()

dataset['Salary'].sem()

import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])#this will give us Z-score of that particular column

# Annova # SSR , SSE, SST

y_mean=np.mean(y)
SSR= np.sum((y_pred-y_mean)**2)
print(SSR) 

y=y[0:6]
SSE= np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(dataset.values)
# Here df.to_numpy() will convert pandas dataframe to 
SST= np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 -(SSR/SST)
r_square
print(r_square)
print(bias_training)
print(variance_testing)