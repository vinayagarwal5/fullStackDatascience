# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:31:41 2026

@author: vinay agrawal
"""

# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\21st - Navie Bayes\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Feature Scaling
# from sklearn.preprocessing import Normalizer
# sc = Normalizer()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Multinomial CM",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Multinomial AC",ac)

bias = classifier.score(X_train, y_train)
print("bias ",bias)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB() 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("BernouliNB CM ",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("BernouliNB AC ",ac)

bias = classifier.score(X_train, y_train)
print("bias ",bias)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("GaussianNB CM",cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("GaussianNB CM",ac)

bias = classifier.score(X_train, y_train)
print("bias ",bias)


