"""
Automated Model Retraining Pipeline with Champion/Challenger A/B Testing

Monitors drift signals, triggers retraining, performs A/B testing,
and promotes challenger to champion if performance gains are validated.

Workflow:
1. Drift Detection: ModelDriftMonitor signals KL-divergence threshold breach
2. Auto-Trigger Retraining: Collect recent field data and retrain models
3. Offline Validation: Evaluate challenger on held-out test set
4. A/B Testing: Deploy challenger to 20% of traffic, compare metrics
5. Promotion Decision: If challenger outperforms champion by ≥3%, promote
6. Rollback Safety: Automatic rollback if production metrics degrade

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

from anomaly_detection.model_drift_monitor import ModelDriftMonitor, DriftDetectionResult
from model_lifecycle.model_registry import ModelRegistry, ModelMetadata, ModelStatus
from validation_framework.data_split_manifest import DataSplitManifest


@dataclass
class RetrainingTrigger:
    """Trigger event for automated retraining."""
    trigger_type: str  # "drift_detected", "scheduled", "manual"
    timestamp: str
    drift_result: Optional[DriftDetectionResult] = None
    reason: str = ""


@dataclass
class ABTestResult:
    """Result of champion vs challenger A/B test."""
    champion_version: str
    challenger_version: str
    champion_accuracy: float
    challenger_accuracy: float
    champion_f1_score: float
    challenger_f1_score: float
    champion_avg_latency_ms: float
    challenger_avg_latency_ms: float
    performance_delta: float  # (challenger - champion) / champion
    n_samples_tested: int
    statistical_significance: bool
    promotion_recommended: bool
    recommendation_reason: str


class AutomatedRetrainingPipeline:
    """
    End-to-end automated retraining pipeline with champion/challenger testing.
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        drift_monitor: ModelDriftMonitor,
        retraining_data_buffer_size: int = 5000,
        ab_test_sample_size: int = 500,
        promotion_threshold_pct: float = 3.0,  # Challenger must beat champion by 3%
        auto_promote: bool = False  # If True, auto-promote without manual approval
    ):
        """
        Args:
            registry: Model registry instance
            drift_monitor: Drift monitor instance
            retraining_data_buffer_size: Number of recent samples to use for retraining
            ab_test_sample_size: Number of samples for A/B test
            promotion_threshold_pct: Min % improvement required for promotion
            auto_promote: Enable automatic promotion without manual approval
        """
        self.registry = registry
        self.drift_monitor = drift_monitor
        self.retraining_buffer_size = retraining_data_buffer_size
        self.ab_test_sample_size = ab_test_sample_size
        self.promotion_threshold = promotion_threshold_pct / 100.0
        self.auto_promote = auto_promote
        
        # Recent telemetry buffer for retraining
        self.telemetry_buffer: List[Dict] = []
        
        # A/B test tracking
        self.ab_test_active = False
        self.challenger_version: Optional[str] = None
        self.ab_test_metrics = {"champion": [], "challenger": []}
    
    def add_telemetry_sample(self, sample: Dict) -> None:
        """
        Add a telemetry sample to the retraining buffer.
        
        Sample should contain:
            - features: engineered feature vector
            - label: ground truth class (if available)
            - timestamp: sample timestamp
        """
        self.telemetry_buffer.append(sample)
        
        # Keep buffer bounded
        if len(self.telemetry_buffer) > self.retraining_buffer_size:
            self.telemetry_buffer.pop(0)
    
    def check_and_trigger_retraining(self) -> Optional[RetrainingTrigger]:
        """
        Check if retraining should be triggered based on drift detection.
        
        Returns:
            RetrainingTrigger if retraining is needed, else None
        """
        drift_result = self.drift_monitor.check_drift()
        
        if drift_result is None:
            return None  # Insufficient data for drift check
        
        if drift_result.is_drift_detected:
            trigger = RetrainingTrigger(
                trigger_type="drift_detected",
                timestamp=datetime.now().isoformat(),
                drift_result=drift_result,
                reason=drift_result.recommendation
            )
            return trigger
        
        return None
    
    def execute_retraining(
        self,
        trigger: RetrainingTrigger,
        train_split: np.ndarray,
        val_split: np.ndarray,
        train_labels: np.ndarray,
        val_labels: np.ndarray
    ) -> Tuple[str, str]:
        """
        Execute retraining for both IsolationForest and LightGBM.
        
        Returns:
            Tuple of (isolation_forest_version, lightgbm_version)
        """
        print(f"[RETRAINING] Triggered: {trigger.reason}")
        print(f"[RETRAINING] Training on {len(train_split)} samples")
        
        start_time = time.time()
        
        # Generate new version numbers (increment minor version)
        champion_if = self.registry.get_champion("isolation_forest")
        champion_lgb = self.registry.get_champion("lightgbm_classifier")
        
        if_version = self._increment_version(
            champion_if.version if champion_if else "1.0.0"
        )
        lgb_version = self._increment_version(
            champion_lgb.version if champion_lgb else "1.0.0"
        )
        
        # ========== Train IsolationForest ==========
        print(f"[RETRAINING] Training IsolationForest v{if_version}...")
        
        scaler_if = StandardScaler()
        train_scaled = scaler_if.fit_transform(train_split)
        
        iso_forest = IsolationForest(
            contamination="auto",
            n_estimators=200,
            max_samples=512,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(train_scaled)
        
        # Validation metrics
        val_scaled = scaler_if.transform(val_split)
        val_scores = iso_forest.decision_function(val_scaled)
        val_preds = iso_forest.predict(val_scaled)
        
        # Convert to binary (1=normal, -1=anomaly)
        val_binary = (val_labels == 0).astype(int) * 2 - 1
        val_accuracy = accuracy_score(val_binary, val_preds)
        
        if_metadata = ModelMetadata(
            model_id=f"isolation_forest_{if_version}",
            model_type="isolation_forest",
            version=if_version,
            created_timestamp=datetime.now().isoformat(),
            trained_on_n_samples=len(train_split),
            training_duration_seconds=time.time() - start_time,
            status=ModelStatus.CHALLENGER.value,
            validation_accuracy=val_accuracy,
            hyperparameters={
                "contamination": "auto",
                "n_estimators": 200,
                "max_samples": 512
            }
        )
        
        self.registry.register_model(
            model_type="isolation_forest",
            version=if_version,
            model_artifact=iso_forest,
            scaler_artifact=scaler_if,
            metadata=if_metadata,
            status=ModelStatus.CHALLENGER
        )
        
        print(f"[RETRAINING] IsolationForest v{if_version} validation accuracy: {val_accuracy:.3f}")
        
        # ========== Train LightGBM ==========
        print(f"[RETRAINING] Training LightGBM v{lgb_version}...")
        
        lgb_start = time.time()
        
        scaler_lgb = StandardScaler()
        train_scaled_lgb = scaler_lgb.fit_transform(train_split)
        val_scaled_lgb = scaler_lgb.transform(val_split)
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(train_scaled_lgb, train_labels)
        
        # Isotonic calibration on validation set
        val_probs = lgb_model.predict_proba(val_scaled_lgb)
        calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Calibrate on validation set (use max probability as input)
        val_max_probs = np.max(val_probs, axis=1)
        val_correct = (lgb_model.predict(val_scaled_lgb) == val_labels).astype(float)
        calibrator.fit(val_max_probs, val_correct)
        
        # Validation metrics
        val_preds_lgb = lgb_model.predict(val_scaled_lgb)
        val_accuracy_lgb = accuracy_score(val_labels, val_preds_lgb)
        val_f1_lgb = f1_score(val_labels, val_preds_lgb, average='macro')
        
        # Per-class F1 scores
        from sklearn.metrics import classification_report
        report = classification_report(val_labels, val_preds_lgb, output_dict=True, zero_division=0)
        per_class_f1 = {str(k): v['f1-score'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        lgb_metadata = ModelMetadata(
            model_id=f"lightgbm_classifier_{lgb_version}",
            model_type="lightgbm_classifier",
            version=lgb_version,
            created_timestamp=datetime.now().isoformat(),
            trained_on_n_samples=len(train_split),
            training_duration_seconds=time.time() - lgb_start,
            status=ModelStatus.CHALLENGER.value,
            validation_accuracy=val_accuracy_lgb,
            validation_f1_score=val_f1_lgb,
            per_class_f1_scores=per_class_f1,
            hyperparameters={
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05
            }
        )
        
        self.registry.register_model(
            model_type="lightgbm_classifier",
            version=lgb_version,
            model_artifact=lgb_model,
            scaler_artifact=scaler_lgb,
            calibrator_artifact=calibrator,
            metadata=lgb_metadata,
            status=ModelStatus.CHALLENGER
        )
        
        print(f"[RETRAINING] LightGBM v{lgb_version} validation accuracy: {val_accuracy_lgb:.3f}, F1: {val_f1_lgb:.3f}")
        print(f"[RETRAINING] Total retraining time: {time.time() - start_time:.1f}s")
        
        return if_version, lgb_version
    
    def _increment_version(self, current_version: str) -> str:
        """Increment semantic version (minor version)."""
        parts = current_version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor + 1}.{patch}"
    
    def run_ab_test(
        self,
        model_type: str,
        champion_version: str,
        challenger_version: str,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> ABTestResult:
        """
        Run A/B test comparing champion vs challenger on test data.
        
        Metrics compared:
        - Accuracy
        - Macro F1 score
        - Inference latency
        """
        print(f"[A/B TEST] {model_type}: Champion v{champion_version} vs Challenger v{challenger_version}")
        
        # Load champion artifacts
        champion_artifacts = self.registry.load_version_artifacts(model_type, champion_version)
        challenger_artifacts = self.registry.load_version_artifacts(model_type, challenger_version)
        
        champion_model = champion_artifacts['model']
        champion_scaler = champion_artifacts.get('scaler')
        
        challenger_model = challenger_artifacts['model']
        challenger_scaler = challenger_artifacts.get('scaler')
        
        # Prepare data
        test_scaled_champion = champion_scaler.transform(test_data) if champion_scaler else test_data
        test_scaled_challenger = challenger_scaler.transform(test_data) if challenger_scaler else test_data
        
        # Champion predictions with latency
        start_champion = time.time()
        champion_preds = champion_model.predict(test_scaled_champion)
        champion_latency = (time.time() - start_champion) / len(test_data) * 1000  # ms per sample
        
        # Challenger predictions with latency
        start_challenger = time.time()
        challenger_preds = challenger_model.predict(test_scaled_challenger)
        challenger_latency = (time.time() - start_challenger) / len(test_data) * 1000
        
        # Compute metrics
        if model_type == "isolation_forest":
            # Binary classification: normal vs anomaly
            test_binary = (test_labels == 0).astype(int) * 2 - 1
            champion_acc = accuracy_score(test_binary, champion_preds)
            challenger_acc = accuracy_score(test_binary, challenger_preds)
            champion_f1 = f1_score(test_binary, champion_preds, average='binary')
            challenger_f1 = f1_score(test_binary, challenger_preds, average='binary')
        else:
            # Multi-class classification
            champion_acc = accuracy_score(test_labels, champion_preds)
            challenger_acc = accuracy_score(test_labels, challenger_preds)
            champion_f1 = f1_score(test_labels, champion_preds, average='macro', zero_division=0)
            challenger_f1 = f1_score(test_labels, challenger_preds, average='macro', zero_division=0)
        
        # Performance delta
        performance_delta = (challenger_acc - champion_acc) / champion_acc if champion_acc > 0 else 0.0
        
        # Statistical significance (simple bootstrap test)
        n_bootstrap = 100
        bootstrap_deltas = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(test_labels), size=len(test_labels), replace=True)
            if model_type == "isolation_forest":
                boot_labels = test_binary[indices]
            else:
                boot_labels = test_labels[indices]
            boot_champion = champion_preds[indices]
            boot_challenger = challenger_preds[indices]
            
            boot_acc_champ = accuracy_score(boot_labels, boot_champion)
            boot_acc_chall = accuracy_score(boot_labels, boot_challenger)
            bootstrap_deltas.append(boot_acc_chall - boot_acc_champ)
        
        # 95% confidence interval doesn't include zero → statistically significant
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        is_significant = ci_lower > 0 or ci_upper < 0
        
        # Promotion decision
        promote = (
            performance_delta >= self.promotion_threshold and
            is_significant and
            challenger_latency <= champion_latency * 1.1  # No more than 10% slower
        )
        
        if promote:
            reason = (
                f"Challenger outperforms champion by {performance_delta*100:.1f}% "
                f"(exceeds {self.promotion_threshold*100:.1f}% threshold) with statistical significance. "
                f"Latency impact: {(challenger_latency/champion_latency - 1)*100:+.1f}%"
            )
        elif not is_significant:
            reason = "Performance difference not statistically significant (95% CI includes zero)"
        elif performance_delta < self.promotion_threshold:
            reason = f"Performance gain {performance_delta*100:.1f}% below threshold {self.promotion_threshold*100:.1f}%"
        else:
            reason = f"Challenger latency {challenger_latency:.2f}ms exceeds champion {champion_latency:.2f}ms by >10%"
        
        result = ABTestResult(
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_accuracy=champion_acc,
            challenger_accuracy=challenger_acc,
            champion_f1_score=champion_f1,
            challenger_f1_score=challenger_f1,
            champion_avg_latency_ms=champion_latency,
            challenger_avg_latency_ms=challenger_latency,
            performance_delta=performance_delta,
            n_samples_tested=len(test_data),
            statistical_significance=is_significant,
            promotion_recommended=promote,
            recommendation_reason=reason
        )
        
        print(f"[A/B TEST] Champion Acc: {champion_acc:.4f}, Challenger Acc: {challenger_acc:.4f}")
        print(f"[A/B TEST] Delta: {performance_delta*100:+.2f}%, Significant: {is_significant}")
        print(f"[A/B TEST] Recommendation: {'PROMOTE' if promote else 'REJECT'}")
        print(f"[A/B TEST] Reason: {reason}")
        
        return result
    
    def promote_challenger_if_approved(
        self,
        model_type: str,
        ab_test_result: ABTestResult
    ) -> bool:
        """
        Promote challenger to champion if A/B test recommends it.
        
        Returns:
            True if promoted, False otherwise
        """
        if not ab_test_result.promotion_recommended:
            print(f"[PROMOTION] Rejected: {ab_test_result.recommendation_reason}")
            return False
        
        if not self.auto_promote:
            print(f"[PROMOTION] Awaiting manual approval for {model_type} v{ab_test_result.challenger_version}")
            return False
        
        print(f"[PROMOTION] Auto-promoting {model_type} v{ab_test_result.challenger_version} to champion")
        self.registry.promote_to_champion(model_type, ab_test_result.challenger_version)
        
        return True
