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

# FIX 1: Drop missing values from your training dataset
dataset = dataset.dropna(subset=[dataset.columns[2], dataset.columns[3], dataset.columns[-1]])

# Select independent features and target variable
X_raw = dataset.iloc[:, [2,3]]
y = dataset.iloc[:, -1].values

# Convert text variables to numbers consistently
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Splitting dataset into the training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.20, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Training 
classifier = LogisticRegression(penalty='l2', solver='lbfgs')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))


# ------------ Future Prediction ----------------------

# 2. LOAD FUTURE PREDICTION DATA
dataset1 = pd.read_csv(r'C:\Users\vinay agrawal\FSDS_PROJECT\15. Logistic regression with future prediction\final.csv')
d2 = dataset1.copy()

# Select the same columns used for training
dataset1_features = dataset1.iloc[:, [3,4]]

# One-Hot Encode future prediction data
encoded_future_data = pd.get_dummies(dataset1_features, drop_first=True)

# Ensure columns align perfectly with training features
encoded_future_data = encoded_future_data.reindex(columns=X_encoded.columns, fill_value=0)

# FIX 2: Handle any missing values in your future prediction columns by filling them with 0
encoded_future_data = encoded_future_data.fillna(0)

# Scale using the existing scaler instance 'sc'
M = sc.transform(encoded_future_data)

y_pred1 = pd.DataFrame()
# Predict and save output
d2['y_pred1'] = classifier.predict(M)

# Save result to file
d2.to_csv(r'C:\Users\vinay agrawal\FSDS_PROJECT\data\final1.csv', index=False)
print("\nPredictions saved successfully to final1.csv!")

# to get the file loc
import os
os.getcwd()

#********************************************************************************

from sklearn.metrics import roc_auc_score,roc_curve

y_pred_prob = classifier.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(y_test, y_pred_prob)
print(auc_score)

fpr,tpr,thresolds = roc_curve(y_test,y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'Logistic Regression (AUC= {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show() 
#============

#training the Naive Bayes