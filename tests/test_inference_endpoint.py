import os
import tempfile
import joblib
import unittest
import pandas as pd

from src import app as flask_app_module

class DummyModel:
    def predict(self, X):
        # return the sum of area as a fake price for determinism
        return [float(42.0)] * len(X)

class TestInferenceEndpoint(unittest.TestCase):
    def test_predict_endpoint_returns_json(self):
        # create a temp model file
        fd, model_path = tempfile.mkstemp(suffix='.joblib')
        os.close(fd)
        try:
            joblib.dump(DummyModel(), model_path)

            # point the app at our temp model
            flask_app_module.MODEL_PATH = model_path
            flask_app_module.model = None  # ensure it reloads

            client = flask_app_module.app.test_client()

            payload = {'area': 2600, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2, 'parking': 1}
            resp = client.post('/predict', json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn('predicted_price', data)
            self.assertIsInstance(data['predicted_price'], float)
        finally:
            try:
                os.remove(model_path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
