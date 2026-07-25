# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:39:07 2026

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

# svr model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree = 4, gamma = 'auto', C=10 )
svr_regressor.fit(X,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("SVR Pred ",svr_model_pred)

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, svr_regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 

# knn model 
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5,weights='distance', leaf_size=30 )
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print("Knn Pred ",knn_reg_pred)

# Visualising the KNN results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, knn_reg_model.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# decission tree 
from sklearn.tree import DecisionTreeRegressor
dtr_reg_model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=10, splitter='random')
dtr_reg_model.fit(X,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print("Decision Tree ",dtr_reg_pred)

# Visualising the Decision tree results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, dtr_reg_model.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Random forest 
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=20, random_state=0)
rfr_reg_model.fit(X,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print("Random forest ",rfr_reg_pred)


# Visualising the Random forest results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rfr_reg_model.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

