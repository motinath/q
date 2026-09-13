"""
Phase 4, 21, 22, 23: Independent Run-Level Test Evaluation, Parameter Shift & Ablation
Enforces zero data leakage by evaluating across whole independent scenario runs.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.qkd_system_parameters import (
    QKDPhysicsConfig,
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_LABEL_TO_ID,
    ROOT_CAUSE_ID_TO_LABEL,
)
from anomaly_detection.sliding_window_features import FEATURE_COLUMN_NAMES
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator
from validation_framework.data_split_manifest import (
    create_and_export_data_splits_manifest,
    generate_scenario_run,
)
from validation_framework.ablation_study import run_ablation_study


def train_models_on_manifest(manifest: Dict[str, Any], models_dir: str = "models") -> Tuple[IsolationForestAnomalyDetector, LightGBMRootCauseClassifier]:
    """
    Trains models using strictly the 70 training runs specified in manifest.
    """
    os.makedirs(models_dir, exist_ok=True)
    train_runs = manifest["partitions"]["training_runs"]
    
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
    
    # Phase 8: Train Anomaly Detector on Healthy Baseline only
    detector = IsolationForestAnomalyDetector(contamination=0.04, n_estimators=60, random_state=42)
    detector.fit_healthy_baseline(X_healthy)
    detector.save(os.path.join(models_dir, "isolation_forest.joblib"), os.path.join(models_dir, "isolation_scaler.joblib"))
    
    # Phase 9: Train Multi-Class Classifier
    classifier = LightGBMRootCauseClassifier(n_estimators=120, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42)
    classifier.fit(X_train, y_train)
    classifier.save(os.path.join(models_dir, "lightgbm_classifier.joblib"))
    
    return detector, classifier


def run_validation_set_b(models_dir: str = "models", manifest_path: str = "data_splits.json") -> Dict[str, Any]:
    """
    Executes rigorous evaluation on independent held-out test runs.
    """
    manifest = create_and_export_data_splits_manifest(manifest_path=manifest_path, total_runs=100)
    detector, classifier = train_models_on_manifest(manifest, models_dir=models_dir)
    validator = PhysicalInvariantValidator()
    
    # Load and evaluate Test Runs (Runs 086 to 100)
    test_runs = manifest["partitions"]["test_runs"]
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
    
    # Evaluate Test Set
    anomaly_preds = []
    ground_truth_anomaly = []
    y_pred_calibrated = []
    physics_agreed_count = 0
    
    for i in range(len(X_test)):
        feat_dict = {col: float(X_test[i, idx]) for idx, col in enumerate(FEATURE_COLUMN_NAMES)}
        gt_label_id = int(y_test[i])
        gt_label_name = ROOT_CAUSE_ID_TO_LABEL[gt_label_id]
        
        # Anomaly inference
        anom_res = detector.predict_sample(feat_dict)
        anomaly_preds.append(1 if anom_res.is_anomaly else 0)
        ground_truth_anomaly.append(0 if gt_label_name == "Normal" else 1)
        
        # Attribution inference + Physics Validation
        attr_raw = classifier.predict_sample(feat_dict, physics_validation=None)
        phys_res = validator.evaluate(feat_dict, ml_predicted_class=attr_raw.predicted_class, ml_confidence=attr_raw.confidence)
        
        attr_calib = classifier.predict_sample(feat_dict, physics_validation=phys_res)
        y_pred_calibrated.append(ROOT_CAUSE_LABEL_TO_ID[attr_calib.predicted_class])
        
        if phys_res.validation_status == "ML + Physics Agree":
            physics_agreed_count += 1
            
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred_calibrated)
    
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=list(range(len(ROOT_CAUSE_CLASSES))))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test_arr, y_pred_arr, average="macro", zero_division=0)
    p_anom, r_anom, f1_anom, _ = precision_recall_fscore_support(ground_truth_anomaly, anomaly_preds, average="binary", zero_division=0)
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_test_arr, y_pred_arr, labels=list(range(len(ROOT_CAUSE_CLASSES))), zero_division=0
    )
    class_metrics = {}
    for idx, cname in enumerate(ROOT_CAUSE_CLASSES):
        class_metrics[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(supp_cls[idx]),
        }
        
    # Phase 22 Test 2: Parameter Shift Evaluation (35 km link span, 250 Hz DCR)
    param_runs = manifest["partitions"]["parameter_shift_runs"]
    shift_X, shift_y = [], []
    shift_config = QKDPhysicsConfig(default_fiber_length_km=35.0, nominal_dark_count_rate_hz=250.0)
    
    for r in param_runs:
        X_run, y_run, _ = generate_scenario_run(
            run_id=r["run_id"],
            random_seed=r["seed"],
            fault_type=r["fault_type"],
            fault_intensity=r["intensity"],
            n_samples=30,
            config=shift_config,
        )
        shift_X.append(X_run)
        shift_y.append(y_run)
        
    X_shift = np.vstack(shift_X)
    y_shift = np.concatenate(shift_y)
    
    shift_preds = []
    for i in range(len(X_shift)):
        feat_dict = {col: float(X_shift[i, idx]) for idx, col in enumerate(FEATURE_COLUMN_NAMES)}
        attr = classifier.predict_sample(feat_dict)
        shift_preds.append(ROOT_CAUSE_LABEL_TO_ID[attr.predicted_class])
        
    shift_acc = float(np.mean(np.array(shift_preds) == y_shift))
    
    # Phase 23: Ablation Study
    ablation_res = run_ablation_study(classifier, validator, X_test, y_test)
    
    return {
        "suite_name": "Validation Set B (Whole Run-Level Split & Generalization)",
        "total_test_samples": len(y_test),
        "overall_accuracy": float(np.mean(y_test_arr == y_pred_arr)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "anomaly_detection": {
            "precision": float(p_anom),
            "recall": float(r_anom),
            "f1_score": float(f1_anom),
        },
        "physics_agreement_rate": float(physics_agreed_count / len(y_test)),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "parameter_shift_generalization_accuracy": float(shift_acc),
        "ablation_study": ablation_res,
    }


if __name__ == "__main__":
    res = run_validation_set_b()
    print("\n--- " + res["suite_name"] + " ---")
    print(f"Overall Accuracy: {res['overall_accuracy']*100:.2f}%")
    print(f"Macro Precision:  {res['macro_precision']*100:.2f}%")
    print(f"Macro Recall:     {res['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:   {res['macro_f1']*100:.2f}%")
    print(f"Physics Agreement Rate: {res['physics_agreement_rate']*100:.2f}%")
    print(f"Parameter Shift Generalization: {res['parameter_shift_generalization_accuracy']*100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cname, m in res["class_metrics"].items():
        print(f"  {cname:<28} | P: {m['precision']*100:5.1f}% | R: {m['recall']*100:5.1f}% | F1: {m['f1_score']*100:5.1f}% | Supp: {m['support']}")
    print("\nAblation Study Summary:")
    for row in res["ablation_study"]["ablation_table"]:
        print(f"  {row['Architecture']:<40} | Macro-F1: {row['Macro_F1']*100:5.1f}% | FPR: {row['False_Positive_Rate']*100:5.2f}%")
    print(f"  Hybrid Improvement over ML: {res['ablation_study']['f1_improvement_pct']:+.2f}%")
