from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["GarageCars"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["FullBath"]),
            float(request.form["YearBuilt"]),
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]
        output = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Estimated House Price: ${output:,}"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
