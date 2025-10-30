import os
import tempfile
import unittest
import pandas as pd
import joblib
from unittest.mock import patch

import src.train_model as tm

class DummyGrid:
    """A lightweight replacement for GridSearchCV used in tests.
    It simply fits the provided estimator once and exposes predict and best_estimator_.
    """
    def __init__(self, estimator, param_grid, cv, scoring, n_jobs):
        self.estimator = estimator
        self.best_estimator_ = None

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.best_estimator_ = self.estimator
        # mimic GridSearchCV attribute
        self.best_params_ = {}

    def predict(self, X):
        return self.best_estimator_.predict(X)

class TestTrainFast(unittest.TestCase):
    def test_train_creates_model_file(self):
        # Create a tiny synthetic dataset
        df = pd.DataFrame({
            'area': [1000, 1500, 1200, 1400],
            'bedrooms': [2, 3, 2, 3],
            'bathrooms': [1, 2, 1, 2],
            'stories': [1, 2, 1, 2],
            'parking': [0, 1, 0, 1],
            'price': [200000, 300000, 210000, 290000]
        })

        fd, path = tempfile.mkstemp(suffix='.csv', text=True)
        os.close(fd)
        try:
            df.to_csv(path, index=False)
            # temporary model path
            model_fd, model_path = tempfile.mkstemp(suffix='.joblib')
            os.close(model_fd)
            os.remove(model_path)

            # Patch GridSearchCV in the train_model module so training is fast
            with patch('src.train_model.GridSearchCV', new=DummyGrid):
                tm.train(path, model_path, experiment_name='test_exp', target='price')

            # Assert model file created
            self.assertTrue(os.path.exists(model_path))
            # load and sanity-check
            m = joblib.load(model_path)
            preds = m.predict(df.drop(columns=['price']))
            self.assertEqual(len(preds), len(df))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
            try:
                os.remove(model_path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
