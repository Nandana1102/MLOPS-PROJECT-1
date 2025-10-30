# House Price Prediction — E2E MLOps Project

This project demonstrates a small end-to-end MLOps pipeline for a regression task (predicting house prices).

## What you get
- Automated preprocessing pipeline (scikit-learn ColumnTransformer)
- Model training with hyperparameter search (GridSearchCV)
- Experiment logging with MLflow
- CI workflow (GitHub Actions) to train, evaluate, and build a Docker image
- Dockerized Flask app for inference

## How to run locally

1. Clone repo and place dataset

```bash
git clone <your-repo>
cd house_price_mlops
# Place the Kaggle 'train.csv' as data/house_prices.csv
```

2. (Optional) Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # linux/mac
venv\Scripts\activate       # windows
pip install -r requirements.txt
```

3. Train model

```bash
python src/train_model.py --data data/house_prices.csv --model-out models/model.joblib
```

This will log the run in MLflow (local) and save `models/model.joblib`.

Notes on defaults and CLI
- The training script resolves default paths relative to the repository root (the `src` folder's parent). If you run the command from the repo root you can simply run `python src/train_model.py` and it will use `data/house_prices.csv` by default.
- You can explicitly specify the target column name with `--target`. The script accepts `SalePrice` by default but will also recognize common aliases (case-insensitive) such as `price` or `saleprice`. Example:

```bash
python src/train_model.py --target price
```

This is helpful if your CSV uses a different column name for the target.

4. Run the Flask app

```bash
python src/app.py
```

Then POST JSON to `http://localhost:8080/predict` with a single example feature dict (column names must match training data). Example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"LotArea": 8450, "OverallQual": 7}' http://localhost:8080/predict
```

## CI/CD
The `.github/workflows/ml_pipeline.yml` runs on pushes to `main`: installs deps, trains, evaluates and builds a Docker image.

## Deploying the Docker image
Push the Docker image to a container registry (Docker Hub / GitHub Container Registry) and connect to Render/Heroku/AWS to deploy the web service.

## Notes on dataset ingestion
- For full automation, you can use Kaggle API to download the dataset in a script. Add your `KAGGLE_USERNAME` and `KAGGLE_KEY` as secrets in GitHub and implement a small helper to download the dataset during CI.

## Tips for demonstration to examiners
- Show MLflow UI (`mlflow ui`) with runs and metrics.
- Show GitHub Actions run logs (successful training + Docker build).
- Demo the Flask endpoint with example inputs.
- Include short report describing problem, pipeline, model selection and experiments.
