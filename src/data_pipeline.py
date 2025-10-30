# src/data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path: str):
    """Load CSV dataset from path."""
    df = pd.read_csv(path)

    # Normalize target column name for compatibility across datasets.
    # If dataset uses 'price' or other casing, map it to 'SalePrice' which the rest
    # of the code expects.
    cols_lower = {c: c.lower() for c in df.columns}
    target_candidates = [orig for orig, lower in cols_lower.items() if lower in ('saleprice', 'price')]
    if 'SalePrice' not in df.columns and target_candidates:
        # pick the first candidate and rename it to 'SalePrice'
        df = df.rename(columns={target_candidates[0]: 'SalePrice'})

    return df

def build_preprocessing_pipeline(df: pd.DataFrame):
    """Return a sklearn ColumnTransformer that imputes, encodes and scales features."""
    # Simple heuristic: treat object dtype as categorical
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop target if present
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features
