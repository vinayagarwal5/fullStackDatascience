#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:39:55 2026
phase 1 :  Data Processing
@author: rojarani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data.csv')

data.head()

X = data.iloc[:,:-1].values
y = data.iloc[:,3].values

imputer = SimpleImputer()
#imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])

X[:,1:3] = imputer.fit_transform(X[:,1:3])

le = LabelEncoder()

X[:,0] = le.fit_transform(X[:,0])

y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
