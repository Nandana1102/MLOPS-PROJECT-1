import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import argparse
import mlflow
import mlflow.sklearn


def train_model(data_path, model_out):
    # Start an MLflow run
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking
    mlflow.set_experiment("House_Price_Prediction")

    with mlflow.start_run():
        # Load dataset
        df = pd.read_csv(data_path)

        # Select only the six key features
        selected_features = [
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "TotalBsmtSF",
            "FullBath",
            "YearBuilt",
        ]

        X = df[selected_features]
        y = df["SalePrice"]

        # Fill missing values
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define model
        model = RandomForestRegressor(n_estimators=150, random_state=42)

        # Log parameters
        mlflow.log_param("features", selected_features)
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"✅ Model trained successfully. MAE: {mae:.2f}, R2: {r2:.2f}")

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Save model
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(model, model_out)
        print(f"✅ Model saved at {model_out}")

        # Log to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()
    train_model(args.data, args.model_out)
