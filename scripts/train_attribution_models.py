"""
Model Training and Serialization Script for Q-SENTINEL
Enforces Rule 7: Zero Data Leakage / Non-overlapping Seeds & Run-Level Partitions

Now with model registry integration for champion/challenger A/B testing.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Dict
from datetime import datetime

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
from model_lifecycle.model_registry import ModelRegistry, ModelMetadata, ModelStatus
from sklearn.metrics import f1_score, classification_report


def train_and_export_models(
    output_dir: str = "models",
    version: str = "1.0.0",
    register_models: bool = True,
    auto_promote: bool = False
) -> None:
    """
    Executes model training pipeline using clean run-level partitions from data_splits.json.
    
    Args:
        output_dir: Directory for model artifacts
        version: Semantic version for this training run
        register_models: If True, register models in model registry
        auto_promote: If True, auto-promote to champion on first training
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("Q-SENTINEL ML TRAINING PIPELINE (Zero Data Leakage Enforcement)")
    print("With Model Registry & Champion/Challenger Tracking")
    print("=================================================================")
    
    # Initialize model registry
    registry = None
    if register_models:
        registry = ModelRegistry(registry_root=output_dir)
        print(f"[REGISTRY] Model registry initialized at {output_dir}")
    
    manifest_path = os.path.join(PROJECT_ROOT, "data_splits.json")
    manifest = load_or_create_data_split_manifest(manifest_path)
    
    train_runs = manifest["partitions"]["training_runs"]
    test_runs = manifest["partitions"]["test_runs"]
    # P3.3: Use the previously-vestigial validation split for isotonic calibration
    val_runs = manifest["partitions"].get("validation_runs", [])
    
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
    training_start = time.time()
    # P2.4: contamination="auto" uses sklearn's heuristic, appropriate for near-clean healthy data.
    # Previous value of 0.04/0.05 incorrectly biased the decision boundary toward false negatives.
    anomaly_detector = IsolationForestAnomalyDetector(contamination="auto", n_estimators=60, random_state=42)
    anomaly_detector.fit_healthy_baseline(X_healthy)
    training_duration_iso = time.time() - training_start
    
    iso_model_path = os.path.join(output_dir, "isolation_forest.joblib")
    iso_scaler_path = os.path.join(output_dir, "isolation_scaler.joblib")
    anomaly_detector.save(iso_model_path, iso_scaler_path)
    print(f"      Saved Isolation Forest to: {iso_model_path}")
    
    # Register in model registry
    if registry:
        # Compute validation accuracy
        test_binary = (y_test == 0).astype(int) * 2 - 1  # Normal=1, Anomaly=-1
        test_results = [anomaly_detector.predict_sample(X_test[i]) for i in range(len(X_test))]
        test_preds = np.array([1 if r.is_anomaly else -1 for r in test_results])
        test_preds = -test_preds  # Flip: 1=normal, -1=anomaly
        val_acc_iso = float(np.mean(test_preds == test_binary))
        
        iso_metadata = ModelMetadata(
            model_id=f"isolation_forest_{version}",
            model_type="isolation_forest",
            version=version,
            created_timestamp=datetime.now().isoformat(),
            trained_on_n_samples=len(X_healthy),
            training_duration_seconds=training_duration_iso,
            status=ModelStatus.TRAINING.value,
            validation_accuracy=val_acc_iso,
            hyperparameters={"contamination": "auto", "n_estimators": 60}
        )
        
        registry.register_model(
            model_type="isolation_forest",
            version=version,
            model_artifact=anomaly_detector.model,
            scaler_artifact=anomaly_detector.scaler,
            metadata=iso_metadata,
            status=ModelStatus.CHAMPION if auto_promote else ModelStatus.TRAINING
        )
        print(f"      Registered IsolationForest v{version} (Validation Acc: {val_acc_iso:.3f})")
        
        if auto_promote:
            registry.promote_to_champion("isolation_forest", version)
            print(f"      Auto-promoted IsolationForest v{version} to CHAMPION")
    
    # 4. Train Layer 4 Multi-Class LightGBM Classifier
    print("\n[4/4] Training Layer 4 LightGBM Multi-Class Classifier...")
    training_start_lgb = time.time()
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

    # P3.3: Fit isotonic calibration on the validation split
    if val_runs:
        print(f"\n[4b/4] Fitting Isotonic Probability Calibration ({len(val_runs)} validation runs)...")
        X_val_list, y_val_list = [], []
        for r in val_runs:
            X_run, y_run, _ = generate_scenario_run(
                run_id=r["run_id"],
                random_seed=r["seed"],
                fault_type=r["fault_type"],
                fault_intensity=r["intensity"],
                n_samples=r["samples"],
            )
            X_val_list.append(X_run)
            y_val_list.append(y_run)
        X_val = np.vstack(X_val_list)
        y_val = np.concatenate(y_val_list)
        classifier.fit_calibration(X_val, y_val)
        classifier.save(lgb_model_path)   # re-save with calibrators
        print(f"      Calibration fitted on {X_val.shape[0]} validation samples and saved.")
    else:
        print("\n[4b/4] No validation_runs in manifest — skipping isotonic calibration.")
        X_val, y_val = X_test, y_test  # Fallback for metrics
    
    training_duration_lgb = time.time() - training_start_lgb
    
    # Evaluate on held-out test split
    test_preds = classifier.model.predict(classifier.scaler.transform(X_test))
    test_accuracy = float(np.mean(test_preds == y_test))
    test_f1 = float(f1_score(y_test, test_preds, average='macro', zero_division=0))
    
    # Per-class F1
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    per_class_f1 = {str(k): v['f1-score'] for k, v in report.items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    print(f"\n[Validation] Independent Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print(f"[Validation] Independent Test Set Macro F1: {test_f1:.3f}")
    
    # Register in model registry
    if registry:
        lgb_metadata = ModelMetadata(
            model_id=f"lightgbm_classifier_{version}",
            model_type="lightgbm_classifier",
            version=version,
            created_timestamp=datetime.now().isoformat(),
            trained_on_n_samples=len(X_train),
            training_duration_seconds=training_duration_lgb,
            status=ModelStatus.TRAINING.value,
            accuracy=test_accuracy,
            validation_accuracy=test_accuracy,
            validation_f1_score=test_f1,
            macro_f1_score=test_f1,
            per_class_f1_scores=per_class_f1,
            hyperparameters={
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31
            }
        )
        
        registry.register_model(
            model_type="lightgbm_classifier",
            version=version,
            model_artifact=classifier.model,
            scaler_artifact=classifier.scaler,
            calibrator_artifact=classifier.calibrator if hasattr(classifier, 'calibrator') else None,
            metadata=lgb_metadata,
            status=ModelStatus.CHAMPION if auto_promote else ModelStatus.TRAINING
        )
        print(f"      Registered LightGBM v{version} (Validation F1: {test_f1:.3f})")
        
        if auto_promote:
            registry.promote_to_champion("lightgbm_classifier", version)
            print(f"      Auto-promoted LightGBM v{version} to CHAMPION")
    
    print("=================================================================\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Q-SENTINEL ML models with registry support")
    parser.add_argument("--version", type=str, default="1.0.0", help="Model version (semantic versioning)")
    parser.add_argument("--no-registry", action="store_true", help="Disable model registry integration")
    parser.add_argument("--auto-promote", action="store_true", help="Auto-promote to champion after training")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    train_and_export_models(
        output_dir=args.output_dir,
        version=args.version,
        register_models=not args.no_registry,
        auto_promote=args.auto_promote
    )
