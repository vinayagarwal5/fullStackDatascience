import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline #data leakages 
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset

data = pd.read_csv(r"C:\Users\vinay agrawal\FSDS_PROJECT\data\USA_Housing.csv")

# Preprocessing

X = data.drop(['Price', 'Address'], axis=1) 
y = data['Price']

# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#define model
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'SGD Regressor': SGDRegressor(),
    'Huber Regressor': HuberRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'Polynomial Regression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'MLP Regressor': MLPRegressor(),
    'KNN Regressor': KNeighborsRegressor(),
    'LightGBM Regressor': lgb.LGBMRegressor(),
    'XGBoost Regressor': xgb.XGBRegressor()
}


# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

#convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print("Model training and evaluation completed. Results saved to 'model_results.csv'.")