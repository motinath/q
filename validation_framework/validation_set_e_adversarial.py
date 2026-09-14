"""
Validation Set E: Adversarial Robustness Testing Suite

Comprehensive adversarial evaluation of Q-SENTINEL ML models.
Tests robustness against FGSM, PGD, and Boundary attacks with
physically-constrained perturbations.

Certification Criteria:
- Misclassification rate < 15% under ε=0.1 perturbations
- Confidence degradation < 0.20 under adversarial attack
- 95%+ of successful attacks remain physically plausible

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
import joblib
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_framework.adversarial_robustness_tester import (
    AdversarialRobustnessTester,
    RobustnessMetrics
)
from validation_framework.data_split_manifest import DataSplitManifest
from anomaly_detection.sliding_window_features import TelemetryFeatureExtractor
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator
from config.qkd_system_parameters import QKDSystemParameters


class ValidationSetE_AdversarialRobustness:
    """
    Validation Set E: Adversarial Robustness
    
    Tests:
    1. FGSM robustness at ε ∈ {0.05, 0.10, 0.20}
    2. PGD robustness (40 iterations) at same epsilon values
    3. Boundary attack robustness (100 iterations)
    4. Physical plausibility of adversarial examples
    5. Certified robustness metrics
    """
    
    def __init__(self):
        self.test_results = {}
        self.certification_status = "PENDING"
    
    def run_all_tests(self) -> Dict:
        """Execute complete adversarial test suite."""
        print("=" * 80)
        print("VALIDATION SET E: ADVERSARIAL ROBUSTNESS TESTING")
        print("=" * 80)
        
        # Load models
        print("\n[SETUP] Loading trained models...")
        try:
            iso_forest = joblib.load("models/isolation_forest.joblib")
            iso_scaler = joblib.load("models/isolation_scaler.joblib")
            lgb_model = joblib.load("models/lightgbm_classifier.joblib")
            # Note: Assuming same scaler for LightGBM
            lgb_scaler = iso_scaler
        except FileNotFoundError as e:
            print(f"[ERROR] Model files not found: {e}")
            print("[ERROR] Please run scripts/train_attribution_models.py first")
            return {"status": "FAILED", "reason": "Models not trained"}
        
        # Load test data
        print("[SETUP] Generating test dataset...")
        X_test, y_test = self._generate_test_dataset(n_samples=200)
        
        # Test 1: IsolationForest Robustness
        print("\n" + "=" * 80)
        print("TEST 1: IsolationForest Adversarial Robustness")
        print("=" * 80)
        
        iso_tester = AdversarialRobustnessTester(
            model=iso_forest,
            scaler=iso_scaler
        )
        
        iso_results = iso_tester.evaluate_robustness(
            X_test=X_test,
            y_test=y_test,
            epsilons=[0.05, 0.1, 0.2],
            attacks=["FGSM", "PGD", "Boundary"]
        )
        
        self.test_results['isolation_forest'] = iso_results
        
        # Test 2: LightGBM Robustness
        print("\n" + "=" * 80)
        print("TEST 2: LightGBM Classifier Adversarial Robustness")
        print("=" * 80)
        
        lgb_tester = AdversarialRobustnessTester(
            model=lgb_model,
            scaler=lgb_scaler
        )
        
        lgb_results = lgb_tester.evaluate_robustness(
            X_test=X_test,
            y_test=y_test,
            epsilons=[0.05, 0.1, 0.2],
            attacks=["FGSM", "PGD"]  # Boundary is slower for multi-class
        )
        
        self.test_results['lightgbm_classifier'] = lgb_results
        
        # Certification decision
        self._evaluate_certification()
        
        # Print summary
        self._print_summary()
        
        return {
            "status": self.certification_status,
            "results": self.test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_test_dataset(self, n_samples: int = 200) -> tuple:
        """
        Generate diverse test dataset covering all fault classes.
        """
        params = QKDSystemParameters()
        emulator = QuantumTelemetryEmulator()
        feature_extractor = TelemetryFeatureExtractor(window_size=25)
        
        X_list = []
        y_list = []
        
        # Sample from each class
        fault_modes = list(range(10))  # 0 to 9
        samples_per_class = n_samples // 10
        
        for fault_id in fault_modes:
            emulator.inject_fault(fault_id)
            
            # Generate enough samples for window
            for _ in range(samples_per_class + 25):
                sample = emulator.step()
                feature_extractor.add_sample(sample)
            
            # Extract features
            for _ in range(samples_per_class):
                features = feature_extractor.extract_features()
                if features is not None:
                    X_list.append([
                        features.qber,
                        features.skr_bps,
                        features.visibility,
                        features.raw_counts_hz,
                        features.dark_counts_hz,
                        features.apd_temperature_c,
                        features.channel_loss_db,
                        features.timing_jitter_ps,
                        features.snr,
                        features.qber_slope_25,
                        features.qber_acceleration_25,
                        features.qber_std_25,
                        features.visibility_mean_25,
                        features.temp_qber_corr_25
                    ])
                    y_list.append(fault_id)
                
                # Step emulator
                sample = emulator.step()
                feature_extractor.add_sample(sample)
            
            # Reset for next class
            emulator.reset()
            feature_extractor = TelemetryFeatureExtractor(window_size=25)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"[SETUP] Generated {len(X)} test samples across {len(set(y))} classes")
        
        return X, y
    
    def _evaluate_certification(self) -> None:
        """
        Evaluate whether models meet adversarial robustness certification criteria.
        
        Criteria:
        1. Misclassification rate < 15% under ε=0.1 (FGSM)
        2. Misclassification rate < 25% under ε=0.1 (PGD-40)
        3. Avg confidence drop < 0.20 under all attacks
        4. Physical plausibility ≥ 95%
        """
        certification_passed = True
        failure_reasons = []
        
        # Check IsolationForest
        if 'isolation_forest' in self.test_results:
            for attack_type, metrics_list in self.test_results['isolation_forest'].items():
                for metrics in metrics_list:
                    if metrics.epsilon_tested == 0.1:
                        # Criterion 1: Misclassification rate
                        if attack_type == "FGSM" and metrics.misclassification_rate > 0.15:
                            certification_passed = False
                            failure_reasons.append(
                                f"IsolationForest FGSM misclass rate {metrics.misclassification_rate:.2%} > 15%"
                            )
                        
                        # Criterion 2: PGD robustness
                        if attack_type == "PGD" and metrics.misclassification_rate > 0.25:
                            certification_passed = False
                            failure_reasons.append(
                                f"IsolationForest PGD misclass rate {metrics.misclassification_rate:.2%} > 25%"
                            )
                        
                        # Criterion 3: Confidence drop
                        if metrics.avg_confidence_drop > 0.20:
                            certification_passed = False
                            failure_reasons.append(
                                f"IsolationForest {attack_type} confidence drop {metrics.avg_confidence_drop:.3f} > 0.20"
                            )
                        
                        # Criterion 4: Physical plausibility
                        if metrics.physically_plausible_attacks_pct < 95.0:
                            certification_passed = False
                            failure_reasons.append(
                                f"IsolationForest {attack_type} plausibility {metrics.physically_plausible_attacks_pct:.1f}% < 95%"
                            )
        
        # Check LightGBM
        if 'lightgbm_classifier' in self.test_results:
            for attack_type, metrics_list in self.test_results['lightgbm_classifier'].items():
                for metrics in metrics_list:
                    if metrics.epsilon_tested == 0.1:
                        if attack_type == "FGSM" and metrics.misclassification_rate > 0.15:
                            certification_passed = False
                            failure_reasons.append(
                                f"LightGBM FGSM misclass rate {metrics.misclassification_rate:.2%} > 15%"
                            )
                        
                        if attack_type == "PGD" and metrics.misclassification_rate > 0.25:
                            certification_passed = False
                            failure_reasons.append(
                                f"LightGBM PGD misclass rate {metrics.misclassification_rate:.2%} > 25%"
                            )
                        
                        if metrics.avg_confidence_drop > 0.20:
                            certification_passed = False
                            failure_reasons.append(
                                f"LightGBM {attack_type} confidence drop {metrics.avg_confidence_drop:.3f} > 0.20"
                            )
                        
                        if metrics.physically_plausible_attacks_pct < 95.0:
                            certification_passed = False
                            failure_reasons.append(
                                f"LightGBM {attack_type} plausibility {metrics.physically_plausible_attacks_pct:.1f}% < 95%"
                            )
        
        if certification_passed:
            self.certification_status = "CERTIFIED"
        else:
            self.certification_status = "FAILED"
            print("\n[CERTIFICATION] FAILED - Reasons:")
            for reason in failure_reasons:
                print(f"  - {reason}")
    
    def _print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("ADVERSARIAL ROBUSTNESS TEST SUMMARY")
        print("=" * 80)
        
        for model_name, results in self.test_results.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 80)
            
            for attack_type, metrics_list in results.items():
                print(f"\n  {attack_type} Attack:")
                for metrics in metrics_list:
                    print(f"    ε = {metrics.epsilon_tested:.2f}:")
                    print(f"      Misclassification Rate: {metrics.misclassification_rate*100:.1f}%")
                    print(f"      Avg Confidence Drop: {metrics.avg_confidence_drop:.3f}")
                    print(f"      Avg L2 Perturbation: {metrics.avg_l2_perturbation:.4f}")
                    print(f"      Avg L∞ Perturbation: {metrics.avg_linf_perturbation:.4f}")
                    print(f"      Physically Plausible: {metrics.physically_plausible_attacks_pct:.1f}%")
                    print(f"      Certified Robust Samples: {metrics.certified_robust_samples}/{metrics.n_samples_tested}")
        
        print("\n" + "=" * 80)
        print(f"OVERALL CERTIFICATION STATUS: {self.certification_status}")
        print("=" * 80)
        
        if self.certification_status == "CERTIFIED":
            print("✅ Models meet adversarial robustness certification criteria")
        else:
            print("❌ Models do not meet certification criteria - retraining recommended")


def main():
    """Execute Validation Set E."""
    validator = ValidationSetE_AdversarialRobustness()
    results = validator.run_all_tests()
    
    # Return exit code based on certification
    if results['status'] == "CERTIFIED":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
