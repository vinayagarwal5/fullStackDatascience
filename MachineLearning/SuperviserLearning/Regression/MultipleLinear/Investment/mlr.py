#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:35:10 2026

Multiple Linear Regression

@author: rojarani
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

inv_data = pd.read_csv('Investment.csv')

inv_data.head()

x = inv_data.iloc[:,:-1]
y = inv_data.iloc[:,-1]

x = pd.get_dummies(x,dtype=int)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(arr=np.full((50,1),54343).astype(int), values = x ,axis =1)


x_opt = x[:,:-1]

regressor_ols = sm.OLS(endog = y, exog= x_opt).fit()

regressor_ols.summary()



x_opt = x[:,[0,1,2,3,5]]

regressor_ols = sm.OLS(endog = y, exog= x_opt).fit()

regressor_ols.summary()



x_opt = x[:,[0,1,2,3]]

regressor_ols = sm.OLS(endog = y, exog= x_opt).fit()

regressor_ols.summary()


x_opt = x[:,[0,1,3]]

regressor_ols = sm.OLS(endog = y, exog= x_opt).fit()

regressor_ols.summary()


x_opt = x[:,[0,1]]

regressor_ols = sm.OLS(endog = y, exog= x_opt).fit()

regressor_ols.summary()


bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance