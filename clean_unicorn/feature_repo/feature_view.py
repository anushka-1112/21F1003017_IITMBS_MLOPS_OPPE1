from feast import FeatureView, Field, Entity
from feast.types import Float32
from data_source import stock_data_source

# ✅ Entity definition — assuming a single stock or stock_id is included
stock_entity = Entity(name="stock_id")

# ✅ Feature View
stock_features = FeatureView(
    name="stock_features",
    entities=[stock_entity],
    ttl=None,
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    online=True,
    source=stock_data_source,
)

