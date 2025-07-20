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

        self.df_all = pd.concat(dfs)
        self.df_all.dropna(subset=["rolling_avg_10", "volume_sum_10", "target"], inplace=True)

        # Take 5 random samples
        self.sample_df = self.df_all.sample(n=5, random_state=42)
        self.X_sample = self.sample_df[["rolling_avg_10", "volume_sum_10"]]
        self.y_true = self.sample_df["target"]

    def test_model_prediction_accuracy(self):
        y_pred = self.model.predict(self.X_sample)
        acc = accuracy_score(self.y_true, y_pred)

        print("Sample predictions:", y_pred)
        print("True values:", self.y_true.tolist())
        print("Accuracy on 5 random samples:", acc)
        print("\nClassification Report:\n", classification_report(self.y_true, y_pred))

        self.assertGreater(acc, 0.3, "Model accuracy on 5-sample test is below 30%")

    def test_rolling_avg_10_feature(self):
        for idx in self.sample_df.index:
            window = self.df_all.loc[:idx].tail(10)["close"]
            expected = window.mean()
            actual = self.df_all.loc[idx, "rolling_avg_10"]
            self.assertAlmostEqual(actual, expected, places=5, msg=f"Mismatch in rolling_avg_10 at {idx}")

    def test_volume_sum_10_feature(self):
        for idx in self.sample_df.index:
            window = self.df_all.loc[:idx].tail(10)["volume"]
            expected = window.sum()
            actual = self.df_all.loc[idx, "volume_sum_10"]
            self.assertAlmostEqual(actual, expected, places=5, msg=f"Mismatch in volume_sum_10 at {idx}")


if __name__ == "__main__":
    unittest.main()
