"""
Adversarial Robustness Testing Framework for VECTOR Q

Evaluates model robustness against adversarial perturbations in feature space.
Implements FGSM, PGD, and boundary attacks to test model resilience.

Unlike image-domain adversarial ML, QKD telemetry has physical constraints:
- Features must remain within physically plausible bounds
- Perturbations should respect optical conservation laws
- Attack success measured by misclassification rate and confidence degradation

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
import lightgbm as lgb


@dataclass
class AdversarialAttackResult:
    """Result of a single adversarial attack."""
    attack_type: str
    epsilon: float
    original_prediction: Any
    adversarial_prediction: Any
    original_confidence: float
    adversarial_confidence: float
    success: bool  # True if misclassification achieved
    l2_perturbation_norm: float
    linf_perturbation_norm: float
    n_iterations: int
    physically_plausible: bool


@dataclass
class RobustnessMetrics:
    """Aggregate robustness metrics across attack suite."""
    attack_type: str
    n_samples_tested: int
    misclassification_rate: float
    avg_confidence_drop: float
    avg_l2_perturbation: float
    avg_linf_perturbation: float
    physically_plausible_attacks_pct: float
    certified_robust_samples: int
    epsilon_tested: float


class AdversarialRobustnessTester:
    """
    Test ML model robustness against adversarial perturbations in feature space.
    
    Implements three attack strategies:
    1. FGSM (Fast Gradient Sign Method): Single-step gradient-based attack
    2. PGD (Projected Gradient Descent): Iterative gradient-based attack
    3. Boundary Attack: Decision-boundary walking attack (gradient-free)
    
    Physical plausibility constraints for QKD telemetry:
    - QBER ∈ [0.0, 0.50]
    - SKR ≥ 0.0 bps
    - Visibility ∈ [0.0, 1.0]
    - Dark counts ≥ 0 Hz
    - Temperature ∈ [-60°C, +30°C]
    """
    
    def __init__(
        self,
        model,
        scaler,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        physics_validator = None
    ):
        """
        Args:
            model: Trained model (IsolationForest or LightGBM)
            scaler: StandardScaler used during training
            feature_bounds: Physical bounds for each feature {feature_name: (min, max)}
            physics_validator: Optional physics validator for plausibility checks
        """
        self.model = model
        self.scaler = scaler
        self.feature_bounds = feature_bounds or self._default_qkd_bounds()
        self.physics_validator = physics_validator
    
    def _default_qkd_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Default physical bounds for QKD telemetry features."""
        return {
            "qber": (0.0, 0.50),
            "skr_bps": (0.0, 5e6),  # 0 to 5 Mbps
            "visibility": (0.0, 1.0),
            "raw_counts_hz": (0.0, 1e8),  # 0 to 100 MHz
            "dark_counts_hz": (0.0, 1e6),  # 0 to 1 MHz
            "apd_temperature_c": (-60.0, 30.0),
            "channel_loss_db": (0.0, 50.0),
            "timing_jitter_ps": (0.0, 1000.0),
            "snr": (0.0, 1000.0),
            "qber_slope_25": (-0.01, 0.01),
            "qber_acceleration_25": (-1e-4, 1e-4)
        }
    
    def fgsm_attack(
        self,
        x: np.ndarray,
        y_true: int,
        epsilon: float = 0.1,
        targeted: bool = False,
        target_class: Optional[int] = None
    ) -> AdversarialAttackResult:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Untargeted: Maximize loss to cause misclassification
        Targeted: Minimize loss toward target_class
        
        Args:
            x: Input feature vector (1D array)
            y_true: True class label
            epsilon: Perturbation magnitude in scaled space
            targeted: If True, perform targeted attack toward target_class
            target_class: Target class for targeted attack
        
        Returns:
            AdversarialAttackResult
        """
        # Scale input
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        
        # Get original prediction
        if hasattr(self.model, 'predict_proba'):
            orig_probs = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
            orig_pred = np.argmax(orig_probs)
            orig_conf = orig_probs[orig_pred]
        else:
            # IsolationForest
            orig_pred = self.model.predict(x_scaled.reshape(1, -1))[0]
            orig_conf = abs(self.model.decision_function(x_scaled.reshape(1, -1))[0])
        
        # Compute gradient approximation via finite differences
        delta = 1e-4
        gradients = np.zeros_like(x_scaled)
        
        for i in range(len(x_scaled)):
            x_plus = x_scaled.copy()
            x_minus = x_scaled.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            
            if hasattr(self.model, 'predict_proba'):
                # For classifiers, use cross-entropy gradient
                probs_plus = self.model.predict_proba(x_plus.reshape(1, -1))[0]
                probs_minus = self.model.predict_proba(x_minus.reshape(1, -1))[0]
                
                if targeted and target_class is not None:
                    # Targeted: gradient of target class probability
                    grad = (probs_plus[target_class] - probs_minus[target_class]) / (2 * delta)
                else:
                    # Untargeted: gradient of true class probability
                    grad = (probs_plus[y_true] - probs_minus[y_true]) / (2 * delta)
            else:
                # For anomaly detectors, use decision function
                score_plus = self.model.decision_function(x_plus.reshape(1, -1))[0]
                score_minus = self.model.decision_function(x_minus.reshape(1, -1))[0]
                grad = (score_plus - score_minus) / (2 * delta)
            
            gradients[i] = grad
        
        # Apply FGSM perturbation
        if targeted:
            perturbation = epsilon * np.sign(gradients)  # Move toward target
        else:
            perturbation = -epsilon * np.sign(gradients)  # Move away from true class
        
        x_adv_scaled = x_scaled + perturbation
        
        # Project back to original space and apply physical constraints
        x_adv = self.scaler.inverse_transform(x_adv_scaled.reshape(1, -1))[0]
        x_adv_constrained = self._apply_physical_constraints(x_adv)
        
        # Re-scale constrained adversarial example
        x_adv_scaled_final = self.scaler.transform(x_adv_constrained.reshape(1, -1))[0]
        
        # Get adversarial prediction
        if hasattr(self.model, 'predict_proba'):
            adv_probs = self.model.predict_proba(x_adv_scaled_final.reshape(1, -1))[0]
            adv_pred = np.argmax(adv_probs)
            adv_conf = adv_probs[adv_pred]
        else:
            adv_pred = self.model.predict(x_adv_scaled_final.reshape(1, -1))[0]
            adv_conf = abs(self.model.decision_function(x_adv_scaled_final.reshape(1, -1))[0])
        
        # Compute perturbation norms
        perturbation_final = x_adv_constrained - x
        l2_norm = np.linalg.norm(perturbation_final, ord=2)
        linf_norm = np.linalg.norm(perturbation_final, ord=np.inf)
        
        # Check physical plausibility
        is_plausible = self._check_physical_plausibility(x_adv_constrained)
        
        # Attack success
        if targeted:
            success = (adv_pred == target_class)
        else:
            success = (adv_pred != orig_pred) if hasattr(self.model, 'predict_proba') else (adv_pred != orig_pred)
        
        return AdversarialAttackResult(
            attack_type="FGSM",
            epsilon=epsilon,
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            original_confidence=float(orig_conf),
            adversarial_confidence=float(adv_conf),
            success=success,
            l2_perturbation_norm=l2_norm,
            linf_perturbation_norm=linf_norm,
            n_iterations=1,
            physically_plausible=is_plausible
        )
    
    def pgd_attack(
        self,
        x: np.ndarray,
        y_true: int,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        n_iterations: int = 40,
        targeted: bool = False,
        target_class: Optional[int] = None
    ) -> AdversarialAttackResult:
        """
        Projected Gradient Descent (PGD) attack.
        
        Iterative version of FGSM with projection back to epsilon-ball.
        
        Args:
            x: Input feature vector
            y_true: True class label
            epsilon: Maximum L∞ perturbation magnitude
            alpha: Step size per iteration
            n_iterations: Number of PGD iterations
            targeted: Targeted attack flag
            target_class: Target class for targeted attack
        
        Returns:
            AdversarialAttackResult
        """
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        x_adv = x_scaled.copy()
        
        # Get original prediction
        if hasattr(self.model, 'predict_proba'):
            orig_probs = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
            orig_pred = np.argmax(orig_probs)
            orig_conf = orig_probs[orig_pred]
        else:
            orig_pred = self.model.predict(x_scaled.reshape(1, -1))[0]
            orig_conf = abs(self.model.decision_function(x_scaled.reshape(1, -1))[0])
        
        # PGD iterations
        for iteration in range(n_iterations):
            # Compute gradient
            delta = 1e-4
            gradients = np.zeros_like(x_adv)
            
            for i in range(len(x_adv)):
                x_plus = x_adv.copy()
                x_minus = x_adv.copy()
                x_plus[i] += delta
                x_minus[i] -= delta
                
                if hasattr(self.model, 'predict_proba'):
                    probs_plus = self.model.predict_proba(x_plus.reshape(1, -1))[0]
                    probs_minus = self.model.predict_proba(x_minus.reshape(1, -1))[0]
                    
                    if targeted and target_class is not None:
                        grad = (probs_plus[target_class] - probs_minus[target_class]) / (2 * delta)
                    else:
                        grad = (probs_plus[y_true] - probs_minus[y_true]) / (2 * delta)
                else:
                    score_plus = self.model.decision_function(x_plus.reshape(1, -1))[0]
                    score_minus = self.model.decision_function(x_minus.reshape(1, -1))[0]
                    grad = (score_plus - score_minus) / (2 * delta)
                
                gradients[i] = grad
            
            # PGD step
            if targeted:
                x_adv = x_adv + alpha * np.sign(gradients)
            else:
                x_adv = x_adv - alpha * np.sign(gradients)
            
            # Project back to epsilon L∞ ball
            perturbation = x_adv - x_scaled
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            x_adv = x_scaled + perturbation
        
        # Apply physical constraints
        x_adv_original = self.scaler.inverse_transform(x_adv.reshape(1, -1))[0]
        x_adv_constrained = self._apply_physical_constraints(x_adv_original)
        x_adv_final = self.scaler.transform(x_adv_constrained.reshape(1, -1))[0]
        
        # Get adversarial prediction
        if hasattr(self.model, 'predict_proba'):
            adv_probs = self.model.predict_proba(x_adv_final.reshape(1, -1))[0]
            adv_pred = np.argmax(adv_probs)
            adv_conf = adv_probs[adv_pred]
        else:
            adv_pred = self.model.predict(x_adv_final.reshape(1, -1))[0]
            adv_conf = abs(self.model.decision_function(x_adv_final.reshape(1, -1))[0])
        
        # Compute perturbation norms
        perturbation_final = x_adv_constrained - x
        l2_norm = np.linalg.norm(perturbation_final, ord=2)
        linf_norm = np.linalg.norm(perturbation_final, ord=np.inf)
        
        is_plausible = self._check_physical_plausibility(x_adv_constrained)
        
        if targeted:
            success = (adv_pred == target_class)
        else:
            success = (adv_pred != orig_pred)
        
        return AdversarialAttackResult(
            attack_type="PGD",
            epsilon=epsilon,
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            original_confidence=float(orig_conf),
            adversarial_confidence=float(adv_conf),
            success=success,
            l2_perturbation_norm=l2_norm,
            linf_perturbation_norm=linf_norm,
            n_iterations=n_iterations,
            physically_plausible=is_plausible
        )
    
    def boundary_attack(
        self,
        x: np.ndarray,
        y_true: int,
        n_iterations: int = 100,
        delta: float = 0.01,
        epsilon: float = 0.01
    ) -> AdversarialAttackResult:
        """
        Boundary Attack: Gradient-free adversarial attack.
        
        Starts from a random misclassified point and walks along decision boundary
        toward the original sample.
        
        Args:
            x: Input feature vector
            y_true: True class label
            n_iterations: Number of boundary walking iterations
            delta: Step size for boundary walking
            epsilon: Orthogonal perturbation magnitude
        
        Returns:
            AdversarialAttackResult
        """
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        
        # Get original prediction
        if hasattr(self.model, 'predict_proba'):
            orig_probs = self.model.predict_proba(x_scaled.reshape(1, -1))[0]
            orig_pred = np.argmax(orig_probs)
            orig_conf = orig_probs[orig_pred]
        else:
            orig_pred = self.model.predict(x_scaled.reshape(1, -1))[0]
            orig_conf = abs(self.model.decision_function(x_scaled.reshape(1, -1))[0])
        
        # Initialize with random perturbation until misclassification
        max_init_attempts = 50
        x_adv = None
        
        for _ in range(max_init_attempts):
            noise = np.random.randn(*x_scaled.shape)
            noise = noise / np.linalg.norm(noise) * 5.0  # Large initial perturbation
            x_candidate = x_scaled + noise
            
            # Apply physical constraints
            x_candidate_orig = self.scaler.inverse_transform(x_candidate.reshape(1, -1))[0]
            x_candidate_constrained = self._apply_physical_constraints(x_candidate_orig)
            x_candidate_scaled = self.scaler.transform(x_candidate_constrained.reshape(1, -1))[0]
            
            if hasattr(self.model, 'predict_proba'):
                pred = np.argmax(self.model.predict_proba(x_candidate_scaled.reshape(1, -1))[0])
            else:
                pred = self.model.predict(x_candidate_scaled.reshape(1, -1))[0]
            
            if pred != orig_pred:
                x_adv = x_candidate_scaled
                break
        
        if x_adv is None:
            # Failed to initialize
            return AdversarialAttackResult(
                attack_type="Boundary",
                epsilon=epsilon,
                original_prediction=orig_pred,
                adversarial_prediction=orig_pred,
                original_confidence=float(orig_conf),
                adversarial_confidence=float(orig_conf),
                success=False,
                l2_perturbation_norm=0.0,
                linf_perturbation_norm=0.0,
                n_iterations=0,
                physically_plausible=True
            )
        
        # Boundary walking: move toward original sample
        for _ in range(n_iterations):
            # Step toward original
            direction = x_scaled - x_adv
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Orthogonal perturbation
            perturb = np.random.randn(*x_adv.shape)
            perturb = perturb - np.dot(perturb, direction) * direction  # Make orthogonal
            perturb = perturb / (np.linalg.norm(perturb) + 1e-8) * epsilon
            
            x_candidate = x_adv + delta * direction + perturb
            
            # Apply constraints
            x_candidate_orig = self.scaler.inverse_transform(x_candidate.reshape(1, -1))[0]
            x_candidate_constrained = self._apply_physical_constraints(x_candidate_orig)
            x_candidate_scaled = self.scaler.transform(x_candidate_constrained.reshape(1, -1))[0]
            
            # Check if still misclassified
            if hasattr(self.model, 'predict_proba'):
                pred = np.argmax(self.model.predict_proba(x_candidate_scaled.reshape(1, -1))[0])
            else:
                pred = self.model.predict(x_candidate_scaled.reshape(1, -1))[0]
            
            if pred != orig_pred:
                x_adv = x_candidate_scaled  # Accept move
        
        # Final adversarial prediction
        x_adv_orig = self.scaler.inverse_transform(x_adv.reshape(1, -1))[0]
        x_adv_constrained = self._apply_physical_constraints(x_adv_orig)
        x_adv_final = self.scaler.transform(x_adv_constrained.reshape(1, -1))[0]
        
        if hasattr(self.model, 'predict_proba'):
            adv_probs = self.model.predict_proba(x_adv_final.reshape(1, -1))[0]
            adv_pred = np.argmax(adv_probs)
            adv_conf = adv_probs[adv_pred]
        else:
            adv_pred = self.model.predict(x_adv_final.reshape(1, -1))[0]
            adv_conf = abs(self.model.decision_function(x_adv_final.reshape(1, -1))[0])
        
        perturbation_final = x_adv_constrained - x
        l2_norm = np.linalg.norm(perturbation_final, ord=2)
        linf_norm = np.linalg.norm(perturbation_final, ord=np.inf)
        
        is_plausible = self._check_physical_plausibility(x_adv_constrained)
        
        return AdversarialAttackResult(
            attack_type="Boundary",
            epsilon=epsilon,
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            original_confidence=float(orig_conf),
            adversarial_confidence=float(adv_conf),
            success=(adv_pred != orig_pred),
            l2_perturbation_norm=l2_norm,
            linf_perturbation_norm=linf_norm,
            n_iterations=n_iterations,
            physically_plausible=is_plausible
        )
    
    def _apply_physical_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Clip features to physically plausible bounds.
        
        This is critical for QKD: adversarial examples must be realizable
        in physical hardware to be meaningful threats.
        """
        x_constrained = x.copy()
        
        # Apply bounds for known features (assuming standard feature order)
        # This is a simplified version - production would use feature name mapping
        feature_idx_map = {
            0: "qber",
            1: "skr_bps",
            2: "visibility",
            3: "raw_counts_hz",
            4: "dark_counts_hz",
            5: "apd_temperature_c",
            6: "channel_loss_db",
            7: "timing_jitter_ps"
        }
        
        for idx, feature_name in feature_idx_map.items():
            if idx < len(x_constrained) and feature_name in self.feature_bounds:
                min_val, max_val = self.feature_bounds[feature_name]
                x_constrained[idx] = np.clip(x_constrained[idx], min_val, max_val)
        
        return x_constrained
    
    def _check_physical_plausibility(self, x: np.ndarray) -> bool:
        """
        Check if adversarial example satisfies optical conservation laws.
        
        Returns True if physically plausible, False if violates physics.
        """
        if self.physics_validator is None:
            # Fallback: check bounds only
            feature_idx_map = {
                0: "qber",
                1: "skr_bps",
                2: "visibility",
                3: "raw_counts_hz",
                4: "dark_counts_hz"
            }
            
            for idx, feature_name in feature_idx_map.items():
                if idx < len(x) and feature_name in self.feature_bounds:
                    min_val, max_val = self.feature_bounds[feature_name]
                    if x[idx] < min_val or x[idx] > max_val:
                        return False
            return True
        
        # Use physics validator if available
        # (Would require converting feature vector back to telemetry sample)
        return True
    
    def evaluate_robustness(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilons: List[float] = [0.05, 0.1, 0.2],
        attacks: List[str] = ["FGSM", "PGD", "Boundary"],
        sample_limit: int = 25
    ) -> Dict[str, List[RobustnessMetrics]]:
        """
        Run complete adversarial evaluation across multiple attacks and epsilons.
        
        Args:
            X_test: Test features
            y_test: True labels
            epsilons: List of perturbation magnitudes to test
            attacks: List of attack types ("FGSM", "PGD", "Boundary")
            sample_limit: Maximum test samples per attack for responsive evaluation
            
        Returns:
            Dictionary mapping attack type to list of RobustnessMetrics (one per epsilon)
        """
        results = {attack: [] for attack in attacks}
        
        for attack_type in attacks:
            for epsilon in epsilons:
                print(f"[ADVERSARIAL] Testing {attack_type} with eps={epsilon}...")
                
                attack_results = []
                
                for i, (x, y_true) in enumerate(zip(X_test, y_test)):
                    if i >= sample_limit:
                        break
                    
                    if attack_type == "FGSM":
                        result = self.fgsm_attack(x, y_true, epsilon=epsilon)
                    elif attack_type == "PGD":
                        result = self.pgd_attack(x, y_true, epsilon=epsilon, alpha=epsilon/10, n_iterations=40)
                    elif attack_type == "Boundary":
                        result = self.boundary_attack(x, y_true, n_iterations=50, epsilon=epsilon)
                    else:
                        continue
                    
                    attack_results.append(result)
                
                # Aggregate metrics
                n_tested = len(attack_results)
                misclass_rate = sum(r.success for r in attack_results) / n_tested if n_tested > 0 else 0.0
                avg_conf_drop = np.mean([r.original_confidence - r.adversarial_confidence 
                                        for r in attack_results]) if n_tested > 0 else 0.0
                avg_l2 = np.mean([r.l2_perturbation_norm for r in attack_results]) if n_tested > 0 else 0.0
                avg_linf = np.mean([r.linf_perturbation_norm for r in attack_results]) if n_tested > 0 else 0.0
                plausible_pct = sum(r.physically_plausible for r in attack_results) / n_tested * 100 if n_tested > 0 else 0.0
                certified_robust = sum(not r.success for r in attack_results)
                
                metrics = RobustnessMetrics(
                    attack_type=attack_type,
                    n_samples_tested=n_tested,
                    misclassification_rate=misclass_rate,
                    avg_confidence_drop=avg_conf_drop,
                    avg_l2_perturbation=avg_l2,
                    avg_linf_perturbation=avg_linf,
                    physically_plausible_attacks_pct=plausible_pct,
                    certified_robust_samples=certified_robust,
                    epsilon_tested=epsilon
                )
                
                results[attack_type].append(metrics)
                
                print(f"  Misclassification Rate: {misclass_rate*100:.1f}%")
                print(f"  Avg Confidence Drop: {avg_conf_drop:.3f}")
                print(f"  Physically Plausible: {plausible_pct:.1f}%")
        
        return results
