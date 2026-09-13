"""
Model Training and Serialization Script for Q-SENTINEL
Enforces Rule 7: Zero Data Leakage / Non-overlapping Seeds & Run-Level Partitions
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.qkd_system_parameters import (
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_LABEL_TO_ID,
    ROOT_CAUSE_ID_TO_LABEL,
)
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from validation_framework.data_split_manifest import (
    load_or_create_data_split_manifest,
    generate_scenario_run,
)


def train_and_export_models(output_dir: str = "models") -> None:
    """
    Executes model training pipeline using clean run-level partitions from data_splits.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("Q-SENTINEL ML TRAINING PIPELINE (Zero Data Leakage Enforcement)")
    print("=================================================================")
    
    manifest_path = os.path.join(PROJECT_ROOT, "data_splits.json")
    manifest = load_or_create_data_split_manifest(manifest_path)
    
    train_runs = manifest["partitions"]["training_runs"]
    test_runs = manifest["partitions"]["test_runs"]
    
    # 1. Generate Training Data
    print(f"\n[1/4] Generating Training Telemetry Dataset ({len(train_runs)} independent runs)...")
    X_train_list, y_train_list = [], []
    X_healthy_list = []
    
    for r in train_runs:
        X_run, y_run, _ = generate_scenario_run(
            run_id=r["run_id"],
            random_seed=r["seed"],
            fault_type=r["fault_type"],
            fault_intensity=r["intensity"],
            n_samples=r["samples"],
        )
        X_train_list.append(X_run)
        y_train_list.append(y_run)
        if r["fault_type"] == "Normal":
            X_healthy_list.append(X_run)
            
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_healthy = np.vstack(X_healthy_list)
    print(f"      Training set generated: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    
    # 2. Generate Independent Test Data
    print(f"\n[2/4] Generating Independent Test Telemetry Dataset ({len(test_runs)} held-out runs)...")
    X_test_list, y_test_list = [], []
    for r in test_runs:
        X_run, y_run, _ = generate_scenario_run(
            run_id=r["run_id"],
            random_seed=r["seed"],
            fault_type=r["fault_type"],
            fault_intensity=r["intensity"],
            n_samples=r["samples"],
        )
        X_test_list.append(X_run)
        y_test_list.append(y_run)
        
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    print(f"      Test set generated: {X_test.shape[0]} samples, {X_test.shape[1]} features.")
    
    # 3. Train Layer 2 Anomaly Detector on Healthy Baseline only
    print("\n[3/4] Training Layer 2 Isolation Forest Detector...")
    anomaly_detector = IsolationForestAnomalyDetector(contamination=0.04, n_estimators=60, random_state=42)
    anomaly_detector.fit_healthy_baseline(X_healthy)
    
    iso_model_path = os.path.join(output_dir, "isolation_forest.joblib")
    iso_scaler_path = os.path.join(output_dir, "isolation_scaler.joblib")
    anomaly_detector.save(iso_model_path, iso_scaler_path)
    print(f"      Saved Isolation Forest to: {iso_model_path}")
    
    # 4. Train Layer 4 Multi-Class LightGBM Classifier
    print("\n[4/4] Training Layer 4 LightGBM Multi-Class Classifier...")
    classifier = LightGBMRootCauseClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
    )
    classifier.fit(X_train, y_train)
    
    lgb_model_path = os.path.join(output_dir, "lightgbm_classifier.joblib")
    classifier.save(lgb_model_path)
    print(f"      Saved LightGBM Classifier to: {lgb_model_path}")
    
    # Evaluate on held-out test split
    test_preds = classifier.model.predict(X_test)
    test_accuracy = float(np.mean(test_preds == y_test))
    print(f"\n[Validation] Independent Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    train_and_export_models()
