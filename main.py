import os
import glob
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

# === CONFIGURATION ===
DATA_DIR = "data"
PARQUET_OUTPUT = os.path.join(DATA_DIR, "stock_dataset.parquet")
MLFLOW_TRACKING_URI = "http://34.45.141.223:8100"
MLFLOW_EXPERIMENT_NAME = "stock_movement_experiment"
MODEL_FILENAME = "stock_model.pkl"
WEIGHTS_FILENAME = "stock_model_weights.pkl"
REGISTERED_MODEL_NAME = "stock_rf_model"
PROMOTION_THRESHOLD = 0.70

# === SETUP MLFLOW ===
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# === STEP 1: READ ALL CSVs AND GENERATE PARQUET ===
def process_csvs_to_parquet():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = []

    for file in all_files:
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)

        # Feature Engineering
        df["rolling_avg_10"] = df["close"].rolling(window=10).mean()
        df["volume_sum_10"] = df["volume"].rolling(window=10).sum()
        df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
        df["event_timestamp"] = df.index
        df["stock_id"] = "STOCK_XYZ"

        dfs.append(df)

    full_df = pd.concat(dfs)
    full_df.dropna(subset=["rolling_avg_10", "volume_sum_10", "target"], inplace=True)

    # Save only required columns
    full_df.reset_index(drop=True)[
        ["stock_id", "event_timestamp", "rolling_avg_10", "volume_sum_10", "target"]
    ].to_parquet(PARQUET_OUTPUT, index=False)

    print(f"✅ Parquet file written to: {PARQUET_OUTPUT}")
    return full_df


# === STEP 2: LOAD PARQUET, TRAIN & REGISTER MODEL ===
def train_and_log_model_from_parquet():
    entity_df = pd.read_parquet(PARQUET_OUTPUT)

    if "event_timestamp" not in entity_df.columns:
        entity_df["event_timestamp"] = pd.Timestamp(datetime.now(timezone.utc))

    if "stock_id" not in entity_df.columns:
        entity_df["stock_id"] = entity_df.index.astype(str)
    else:
        entity_df["stock_id"] = entity_df["stock_id"].astype(str)

    feature_df = entity_df.copy()

    if "target" not in feature_df.columns:
        raise ValueError("The target column is missing from the feature dataset.")

    X = feature_df[["rolling_avg_10", "volume_sum_10"]]
    y = feature_df["target"]

    print("\n==== Debug Info ====")
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("Missing values:\n", X.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and log to MLflow
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save locally
        joblib.dump(clf, MODEL_FILENAME)
        joblib.dump(clf.feature_importances_, WEIGHTS_FILENAME)

        # Log parameters, metrics, and artifacts
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(MODEL_FILENAME)
        mlflow.log_artifact(WEIGHTS_FILENAME)

        # Signature and input example
        signature = infer_signature(X_train, clf.predict(X_train))
        input_example = X_train.iloc[:1]

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"🎯 Run ID: {run.info.run_id}")

        # === OPTIONAL: Auto-promote to Production ===
        if accuracy >= PROMOTION_THRESHOLD:
            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[-1]
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(f"🚀 Model v{latest_version.version} promoted to Production.")
            except MlflowException as e:
                print("⚠️ Failed to promote model to Production:", e)


# === MAIN ===
if __name__ == "__main__":
    print("📥 Processing all CSVs into feature parquet...")
    process_csvs_to_parquet()

    print("🚀 Training and logging model to MLflow (with registry)...")
    train_and_log_model_from_parquet()
