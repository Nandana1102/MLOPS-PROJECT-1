# demo/run_demo.ps1 - simple demo script for presenters
# Usage: run from repo root in PowerShell with venv activated

Write-Host "Running unit tests..."
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe -m pytest -q

Write-Host "Training a quick model (will log to MLflow and save models/model.joblib)..."
C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe src\train_model.py --target price

Write-Host "Starting Flask app in background (you may open the URL shown)..."
Start-Process -NoNewWindow -FilePath C:/Users/nanda/Downloads/house_price_mlops/venv/Scripts/python.exe -ArgumentList 'src\app.py'

Write-Host "To show MLflow UI, run in a separate terminal: mlflow ui"
Write-Host "Then POST a sample JSON to http://localhost:8080/predict to get a prediction."

Write-Host "Demo script complete."
