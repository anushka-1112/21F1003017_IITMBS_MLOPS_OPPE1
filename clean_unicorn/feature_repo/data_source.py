from feast import FileSource
from datetime import datetime

stock_data_source = FileSource(
    name="stock_source",
    path="./data/stock_dataset.parquet",
    timestamp_field="event_timestamp",  # Ensure this column exists
)

