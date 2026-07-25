import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__, template_folder=os.path.dirname(__file__))

base_dir = r"c:\Users\vinay agrawal\FSDS_PROJECT\templates"

# Load models
model_names = [
    'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression',
    'Polynomial Regression', 'SGD Regressor', 'MLP Regressor', 'Random Forest Regressor',
    'Support Vector Regressor', 'LightGBM Regressor', 'XGBoost Regressor', 'KNN Regressor'
]
models = {}
for name in model_names:
    with open(fr'{base_dir}\{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

# Load evaluation results
results_df = pd.read_csv(fr'{base_dir}\model_results.csv')

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])
   
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)