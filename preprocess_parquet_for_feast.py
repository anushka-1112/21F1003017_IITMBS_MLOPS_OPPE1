from datetime import datetime, timezone
import pandas as pd
import os
import glob

# Setup paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# Read and concatenate all CSVs
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Normalize column names
df.columns = [col.lower().strip() for col in df.columns]

# Add Feast-required event_timestamp column
df["event_timestamp"] = pd.Timestamp(datetime.now(timezone.utc)).round("s")

# Save to Parquet
output_path = os.path.join(base_dir, "clean_unicorn/feature_repo/data/stock_dataset.parquet")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_parquet(output_path, index=False)

print("Parquet file created:", output_path)

