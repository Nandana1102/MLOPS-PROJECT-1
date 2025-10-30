# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_pipeline import load_data

def evaluate(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df = load_data(data_path)
    df = df[df['SalePrice'].notna()]
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    print(f"Eval RMSE: {rmse:.2f}, R2: {r2:.3f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../models/model.joblib')
    parser.add_argument('--data', default='../data/house_prices.csv')
    args = parser.parse_args()
    evaluate(args.model, args.data)
