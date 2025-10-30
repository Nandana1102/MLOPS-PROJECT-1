# src/train_model.py
import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

try:
    # when run as a package (tests) this import works
    from src.data_pipeline import load_data, build_preprocessing_pipeline
except Exception:
    # when running the script directly from the src folder, fallback to local import
    from data_pipeline import load_data, build_preprocessing_pipeline
from sklearn.pipeline import Pipeline

def train(data_path: str, model_out: str, experiment_name: str = 'house_price_experiment', target: str = 'SalePrice'):
    df = load_data(data_path)
    # If target column isn't present exactly, try to find a case-insensitive match
    if target not in df.columns:
        for c in df.columns:
            if c.lower() == target.lower():
                df = df.rename(columns={c: target})
                break
    # As a fallback, try common alternative names
    if target not in df.columns:
        for alt in ('price', 'saleprice'):
            for c in df.columns:
                if c.lower() == alt:
                    df = df.rename(columns={c: target})
                    break
            if target in df.columns:
                break
    # simple drop rows with NA in target column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

    df = df.copy()
    df = df[df[target].notna()]

    X = df.drop(columns=[target])
    y = df[target]

    # Build preprocessing pipeline using feature matrix (without target)
    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(X)

    # full pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10]
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        # Some scikit-learn versions may not support the 'squared' kwarg; compute
        # RMSE via sqrt of MSE for compatibility.
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # log best params and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric('rmse', float(rmse))
        mlflow.log_metric('r2', float(r2))

        # log model
        # Provide an input example so MLflow can capture a model signature and
        # avoid warnings in the UI. Use the first row of X_train if available.
        try:
            input_example = X_train.head(1)
        except Exception:
            input_example = None
        if input_example is not None and not input_example.empty:
            # Convert integer columns to float in the input example to avoid
            # MLflow schema enforcement warnings about integers with missing values.
            try:
                for c in input_example.columns:
                    if pd.api.types.is_integer_dtype(input_example[c].dtype):
                        input_example[c] = input_example[c].astype(float)
            except Exception:
                # If conversion fails for any reason, fall back to logging without
                # modified input_example to ensure the run still records.
                mlflow.sklearn.log_model(grid.best_estimator_, 'model')
            else:
                mlflow.sklearn.log_model(grid.best_estimator_, 'model', input_example=input_example)
        else:
            mlflow.sklearn.log_model(grid.best_estimator_, 'model')

        # also save locally
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(grid.best_estimator_, model_out)

    print(f"Training completed. RMSE: {rmse:.2f}, R2: {r2:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Resolve defaults relative to the repository root (one level above src)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_data = os.path.join(repo_root, 'data', 'house_prices.csv')
    default_model = os.path.join(repo_root, 'models', 'model.joblib')

    parser.add_argument('--data', default=default_data,
                        help='Path to CSV dataset (default: data/house_prices.csv in repo root)')
    parser.add_argument('--model-out', default=default_model,
                        help='Path to write trained model (default: models/model.joblib in repo root)')
    parser.add_argument('--target', default='SalePrice',
                        help="Name of the target column in the dataset (default: 'SalePrice'). Case-insensitive aliases 'price' and 'saleprice' are recognized.")
    parser.add_argument('--experiment', default='house_price_experiment')
    args = parser.parse_args()

    train(args.data, args.model_out, args.experiment, args.target)
