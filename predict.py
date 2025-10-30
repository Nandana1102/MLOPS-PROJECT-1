import pickle
import pandas as pd

# Load the trained model
with open("models/house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example new data for prediction (adjust according to your dataset features)
new_data = pd.DataFrame({
    'area': [2500],
    'bedrooms': [4],
    'bathrooms': [3],
    'stories': [2],
    'parking': [1]
})

# Predict
pred = model.predict(new_data)
print(f"Predicted house price: {pred[0]:.2f}")
