Presentation checklist and script

Goal: 5-minute live demo showing the end-to-end pipeline: test -> train -> MLflow -> serve.

Timing (5-6 minutes)
- 0:00-0:30: Intro & problem statement
- 0:30-1:00: Show repo structure and key files (`src/train_model.py`, `src/data_pipeline.py`, `src/app.py`)
- 1:00-2:00: Run tests to show CI-ready code
- 2:00-3:30: Train model and show MLflow UI with metrics and artifacts
- 3:30-4:30: Start app and run a prediction (POST) to show inference
- 4:30-5:00: Wrap up, next steps

Exact commands (PowerShell, run from repo root)

# Activate venv
venv\Scripts\Activate.ps1

# Run tests
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe -m pytest -q

# Train model (uses repo defaults)
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe src\train_model.py --target price

# Start MLflow UI (separate terminal)
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe -m mlflow ui
# open http://127.0.0.1:5000 in browser to inspect runs

# Start Flask app (separate terminal)
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe src\app.py

# POST a sample JSON (PowerShell)
$payload = @{ area=2600; bedrooms=3; bathrooms=2; stories=2; parking=1 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8080/predict -Method POST -Body $payload -ContentType 'application/json'

Notes
- If MLflow warnings appear about integer columns, they are harmless for the demo; we've adjusted the training script to log float input_examples.
- `models/` is in `.gitignore`. If you want a committed sample model for offline demo, remove it from .gitignore and commit the model.

Troubleshooting
- If Flask fails to load the model, ensure `models/model.joblib` exists or set `MODEL_PATH` env var before starting the app.
- If GitHub Actions is not enabled, ensure the repo is pushed to GitHub (we pushed your branch already).

Good luck with the presentation!