# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:08:18 2026

@author: vinay agrawal
"""
#pip install xgboost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.compose import ColumnTransformer 


# 1. LOAD TRAINING DATA
dataset = pd.read_csv(r'C:\Users\vinay agrawal\FSDS_PROJECT\data\Churn_Modelling.csv')

# Select independent features and target variable

X = dataset.iloc[:, 3: -1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#One Hot Encoding the "Geography" column

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Splitting dataset into the training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.20, random_state=0)

# Training XGBoost on training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

ac = accuracy_score(y_test, y_pred)
# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy after kfold : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

