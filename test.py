import unittest
import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report


class TestStockModel(unittest.TestCase):
    def setUp(self):
        # Load trained model
        self.model = joblib.load("stock_model.pkl")

        # Load and process data
        all_files = glob.glob(os.path.join("data", "*.csv"))
        dfs = []

        for file in all_files:
            df = pd.read_csv(file, parse_dates=["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df.set_index("timestamp", inplace=True)

            df["rolling_avg_10"] = df["close"].rolling(window=10).mean()
            df["volume_sum_10"] = df["volume"].rolling(window=10).sum()
            df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)

            dfs.append(df)

        df_all = pd.concat(dfs)
        df_all.dropna(subset=["rolling_avg_10", "volume_sum_10", "target"], inplace=True)

        # Take 5 random samples
        sample_df = df_all.sample(n=5, random_state=42)
        self.X_sample = sample_df[["rolling_avg_10", "volume_sum_10"]]
        self.y_true = sample_df["target"]

    def test_model_prediction_accuracy(self):
        y_pred = self.model.predict(self.X_sample)
        acc = accuracy_score(self.y_true, y_pred)
        print("Sample predictions:", y_pred)
        print("True values:", self.y_true.tolist())
        print("Accuracy on 5 random samples:", acc)
        print("\nClassification Report:\n", classification_report(self.y_true, y_pred))

        self.assertGreater(acc, 0.3, "Model accuracy on 5-sample test is below 30%")


if __name__ == "__main__":
    unittest.main()
