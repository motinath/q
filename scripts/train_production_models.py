"""
VECTOR-Q Production Model Training Script
Trains Isolation Forest and LightGBM models on the official 9-class, 34-feature datasets
and serializes checkpoints to models/.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_governance import ALL_FEATURE_COLUMNS, FAULT_LABEL_TO_ID
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier

def train_and_save():
    print("Loading official benchmark datasets...")
    df_norm = pd.read_parquet(PROJECT_ROOT / "data" / "normal_dataset.parquet")
    df_single = pd.read_parquet(PROJECT_ROOT / "data" / "single_fault_dataset.parquet")

    # 1. Train Isolation Forest on normal telemetry
    print("Fitting IsolationForestAnomalyDetector on normal manifold...")
    X_norm = df_norm[ALL_FEATURE_COLUMNS].values
    ad = IsolationForestAnomalyDetector(n_estimators=100)
    ad.fit(X_norm)
    ad.save(
        str(PROJECT_ROOT / "models" / "isolation_forest.joblib"),
        str(PROJECT_ROOT / "models" / "isolation_scaler.joblib"),
    )
    print("  -> Saved models/isolation_forest.joblib and models/isolation_scaler.joblib")

    # 2. Train LightGBM Classifier on known classes
    print("Fitting LightGBMRootCauseClassifier on known fault classes...")
    known_df = df_single[df_single["fault_class"] != "Unknown Fault"].copy()
    norm_sample = df_norm.sample(500, random_state=42).copy()
    full_train = pd.concat([norm_sample, known_df], ignore_index=True)

    X_train = full_train[ALL_FEATURE_COLUMNS].values
    y_train = full_train["fault_id"].values

    clf = LightGBMRootCauseClassifier(n_estimators=150)
    clf.fit(X_train, y_train)
    clf.save(str(PROJECT_ROOT / "models" / "lightgbm_classifier.joblib"))
    print(f"  -> Saved models/lightgbm_classifier.joblib (Classes: {len(np.unique(y_train))})")

if __name__ == "__main__":
    train_and_save()
