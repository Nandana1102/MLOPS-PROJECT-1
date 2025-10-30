import os
import tempfile
import unittest
import pandas as pd

from src.data_pipeline import load_data

class TestDataPipeline(unittest.TestCase):
    def test_load_data_renames_price_to_SalePrice(self):
        # create a temporary CSV with 'price' column
        fd, path = tempfile.mkstemp(suffix='.csv', text=True)
        os.close(fd)
        try:
            df = pd.DataFrame({
                'area': [1000],
                'bedrooms': [2],
                'price': [123456]
            })
            df.to_csv(path, index=False)

            loaded = load_data(path)
            # After loading, the pipeline should normalize/rename 'price' to 'SalePrice'
            self.assertIn('SalePrice', loaded.columns)
            self.assertEqual(loaded['SalePrice'].iloc[0], 123456)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
