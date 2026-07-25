# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:40:34 2026

@author: vinay agrawal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\data\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# linear model  -- linear algorythem ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial model  ( by defeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2= LinearRegression()
lin_reg_2.fit(X_poly, y)

# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# poly nomial visualization 

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#svr mpdel

from sklearn.svm import SVR
svr_regressor = SVR(kernel= 'poly', degree= 4,gamma= 'auto')
svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)






# Knn Model
from sklearn.neighbors import KNeighborsRegressor

knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X, y)

knn_model_pred = knn_regressor.predict([[6.5]])
knn_model_pred

knn_regressor = KNeighborsRegressor(n_neighbors=4)
knn_regressor.fit(X, y)

knn_model_pred = knn_regressor.predict([[6.5]])
knn_model_pred

knn_regressor = KNeighborsRegressor(n_neighbors=2,algorithm='brute')
knn_regressor.fit(X, y)

knn_model_pred = knn_regressor.predict([[6.5]])
knn_model_pred

# Tree Algorithm

from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X,y)

dt_model_pred = dt_regressor.predict([[6.5]])
dt_model_pred


# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(random_state=0, n_estimators=8)
rf_regressor.fit(X,y)

rf_model_pred = rf_regressor.predict([[6.5]])
rf_model_pred