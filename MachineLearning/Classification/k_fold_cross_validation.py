# k-Fold Cross Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\29th - Cross validation\30th - Cross validation\0.ML MODEL TUNNING _ RANDOMSEARCH CV & GRID SEARCH CV\Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train, y_train)
print("Bias ",bias)

variance = classifier.score(X_test, y_test)
print("variance ",variance)

# you can add implement auc & roc 

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1,2, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1,2, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           ) 
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# Randomsearch cv  

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
#from scipy.stats import randint

# Define parameter distributions
parameters = {
    'C': uniform(1, 1000),   # Random values between 1 and 1001
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.1, 0.9)  # Random values between 0.1 and 1.0
}

# Apply Randomized Search
random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters,
    n_iter=20,          # Number of random combinations
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=0,
    
)

# Fit model
random_search.fit(X_train, y_train)

# Best results
best_accuracy = random_search.best_score_
best_parameters = random_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Parameters:", best_parameters)
