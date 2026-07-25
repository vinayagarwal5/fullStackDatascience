#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:00:31 2026
Phase 2 : Simple Linear Regression
@author: rojarani
"""

import pandas as pd

sal_data = pd.read_csv(r'/Users/rojarani/Documents/AIML/GitAIML/Full-Stack-Data-Science-WIth-Gen-AI-and-Agentic-AI/ML/Simple Linear Regression/Salary Prediction/dataset/Salary_Data.csv')

x = sal_data.iloc[:,:-1]
y = sal_data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary Predection")
plt.xlabel("Year of experience")
plt.ylabel("salary")
plt.show()

m_coef = regressor.coef_
c_intercept = regressor.intercept_

y_12 = m_coef*12+c_intercept
print(y_12)

bias_score = regressor.score(x_train,y_train)

variance_score = regressor.score(x_test,y_test)
print(bias_score)
print(variance_score)


# Statistics integration to ML Model

sal_data.mean()
sal_data['Salary'].mean()
sal_data['YearsExperience'].mean()

sal_data.median()
sal_data['Salary'].median()
sal_data['YearsExperience'].median()


sal_data.var()
sal_data['Salary'].var()
sal_data['YearsExperience'].var()

sal_data.var()
sal_data['Salary'].var()
sal_data['YearsExperience'].var()

sal_data.std()
sal_data['Salary'].std()
sal_data['YearsExperience'].std()

# Correlations

from scipy.stats import variation

variation(sal_data.values)
variation(sal_data['Salary'])
variation(sal_data['YearsExperience'])

sal_data.corr()
sal_data['Salary'].corr(sal_data['YearsExperience'])
sal_data['Salary'].corr(sal_data['Salary'])


sal_data.skew()
sal_data['Salary'].skew()
sal_data['YearsExperience'].skew()

sal_data.sem()

from scipy.stats import stats
sal_data.apply(stats.zscore)

stats.zscore(sal_data['Salary'])
stats.zscore(sal_data['YearsExperience'])

#ANOVA
import numpy as np
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
SSR

y=y[0:6]
SSE = np.sum((y-y_pred)**2)
SSE

mean_total = np.mean(sal_data.values)
SST = np.sum((sal_data.values - mean_total)**2)

rsquare = 1 - (SSR/(SSR+SST))
rsquare

