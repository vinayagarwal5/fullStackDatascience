# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:32:51 2026

@author: vinay agrawal
"""
#pip install lightgbm
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier  # Artificial Neural Network (ANN)

# Feature Extraction & Evaluation Metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# 1. Load Dataset
dataset = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\data\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# 2. Advanced Text Cleaning (Preserving Sentiment Nuance)
corpus = []
ps = PorterStemmer()

# FIX: Do not strip out negative words that define low review scores!
custom_stopwords = set(stopwords.words('english'))
words_to_keep = {'not', 'no', 'never', 'neither', 'nor', 'but', 'against'}
custom_stopwords = custom_stopwords - words_to_keep

print("Cleaning reviews text...")
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['Review'][i]))
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in custom_stopwords]
    corpus.append(' '.join(review))

y = dataset.iloc[:, 1].values

# 3. Define Vectorization Techniques to Test
vectorizers = {
    "Bag of Words (BoW)": CountVectorizer(max_features=1500),
    "TF-IDF Vectorizer": TfidfVectorizer(max_features=1500)
}

# 4. Define All Target Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=0),
    "XGBoost": XGBClassifier(random_state=0, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=0, verbose=-1),
    "ANN Classifier": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=0)
}

# 5. Master Case Study Loop
results_list = []

for vec_name, vec_tool in vectorizers.items():
    print(f"\n--- Running Models using: {vec_name} ---")
    
    # Generate Numerical Features
    X = vec_tool.fit_transform(corpus).toarray()
    
    # OPTIMIZATION STEP: Adjustable split ratio (Try 0.15 or 0.25 here if needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    for clf_name, model in classifiers.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict Classes and Probabilities
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        except AttributeError:
            auc_score = "N/A"  # Fallback if a specific model setting doesn't output probability
            
        # Metrics Calculations
        ac = accuracy_score(y_test, y_pred)
        train_score = model.score(X_train, y_train)  # Higher is lower Bias
        test_score = model.score(X_test, y_test)     # Generalization check
        
        # Determine Fit State
        score_gap = train_score - test_score
        if score_gap > 0.15:
            fit_status = "Overfitting (High Var)"
        elif train_score < 0.70 and test_score < 0.70:
            fit_status = "Underfitting (High Bias)"
        else:
            fit_status = "Balanced (Equal Scale)"
            
        # Store for tabular comparison
        results_list.append({
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Accuracy": round(ac, 4),
            "Train Score (Bias Indicator)": round(train_score, 4),
            "Test Score (Variance Indicator)": round(test_score, 4),
            "Score Gap": round(score_gap, 4),
            "AUC Score": round(auc_score, 4) if isinstance(auc_score, float) else auc_score,
            "Fit Status": fit_status
        })

# 6. Print Final Benchmark Report
df_results = pd.DataFrame(results_list)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n========================= FINAL CASE STUDY BENCHMARK REPORT =========================")
print(df_results.to_string(index=False))
