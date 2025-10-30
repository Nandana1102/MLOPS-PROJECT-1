# src/app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.joblib')
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

@app.route('/')
def home():
    return "House Price Prediction API. POST JSON to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload received'}), 400

    # Expect data to be a dict of features matching training columns
    df = pd.DataFrame([data])
    m = load_model()
    pred = m.predict(df)[0]
    return jsonify({'predicted_price': float(pred)})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
