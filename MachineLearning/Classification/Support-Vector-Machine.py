# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:08:18 2026

@author: vinay agrawal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. LOAD TRAINING DATA
dataset = pd.read_csv(r'C:\Users\vinay agrawal\FSDS_PROJECT\data\logit classification.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Feature Scaling

sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Training the svm model on the training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)



cm = confusion_matrix(y_test, y_pred)
print("Confusion_matrix:\n",cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


cr = classification_report(y_test, y_pred)
print("classifier report:\n",cr)

bias = classifier.score(x_train,y_train)
print("bias:",bias)

variance= classifier.score(x_test, y_test)
print("variance:",variance)

