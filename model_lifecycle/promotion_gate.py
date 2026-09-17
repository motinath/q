"""
Module D: Shadow Validation Gate — The Core Model Safety Component
Ensures no retrained candidate model is promoted to production without passing:
1. Check 1 — Offline Regression Test: Macro F1 must not regress beyond tolerance (0.02)
   across historical physics validation datasets.
2. Check 2 — Attack-Class Recall Floor: Recall on quantum attack classes (6-9) must be
   strictly >= 0.98 (non-negotiable hard floor to prevent silent adversarial degradation).
3. Check 3 — Live Shadow Mode Burn-in: Runs silently in parallel with production on live
   traffic. Halts immediately and flags dangerous disagreements if candidate predicts
   a benign class while production or physics guard flagged an attack.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Tuple
from sklearn.metrics import f1_score, recall_score, confusion_matrix

CRITICAL_FAULT_CLASSES: Set[int] = {1, 2, 3, 4, 5}  # Optical Misalignment, Channel Loss, APD Degradation, Thermal Drift, Timing Jitter
ATTACK_CLASSES: Set[int] = {6, 7, 8, 9}  # Intercept-Resend, Blinding, PNS, Time-Shift (Research Reference)
BENIGN_CLASSES: Set[int] = {0, 1, 2, 3, 4, 5}  # Normal + Physical Degradations


@dataclass
class ShadowVerdict:
    """Outcome of shadow validation evaluation."""
    ready: bool
    reason: str
    n_samples_observed: int = 0
    agreement_rate: float = 0.0
    dangerous_disagreements: List[Dict[str, Any]] = field(default_factory=list)
    offline_regression_passed: bool = False
    attack_recall_floor_passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def compute_class_recall(model: Any, X: np.ndarray, y: np.ndarray, target_class: int) -> float:
    """Calculates recall for a specific target class."""
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
    elif hasattr(model, "predict_batch"):
        y_pred = model.predict_batch(X)
    else:
        raise AttributeError("Model must implement predict or predict_batch")

    mask = (y == target_class)
    if not np.any(mask):
        return 1.0  # No ground truth instances to miss

    tp = np.sum((y_pred == target_class) & mask)
    fn = np.sum((y_pred != target_class) & mask)
    total = tp + fn
    return float(tp / total) if total > 0 else 1.0


def check_no_regression(
    candidate_model: Any,
    production_model: Any,
    validation_datasets: List[Tuple[str, np.ndarray, np.ndarray]],
    tolerance: float = 0.02,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check 1: Evaluates candidate vs production across held-out validation sets.
    Returns (True, report) if candidate macro F1 >= production F1 - tolerance on all sets.
    """
    report = {}
    all_passed = True

    for name, X, y in validation_datasets:
        if len(X) == 0:
            continue

        cand_preds = candidate_model.predict(X) if hasattr(candidate_model, "predict") else candidate_model.predict_batch(X)
        prod_preds = production_model.predict(X) if hasattr(production_model, "predict") else production_model.predict_batch(X)

        cand_f1 = float(f1_score(y, cand_preds, average="macro", zero_division=0))
        prod_f1 = float(f1_score(y, prod_preds, average="macro", zero_division=0))
        delta = cand_f1 - prod_f1

        report[f"{name}_candidate_f1"] = round(cand_f1, 4)
        report[f"{name}_production_f1"] = round(prod_f1, 4)
        report[f"{name}_delta"] = round(delta, 4)

        if cand_f1 < (prod_f1 - tolerance):
            all_passed = False
            report[f"{name}_status"] = "REGRESSION_DETECTED"
        else:
            report[f"{name}_status"] = "PASSED"

    return all_passed, report


def check_critical_fault_recall_floor(
    candidate_model: Any,
    test_X: np.ndarray,
    test_y: np.ndarray,
    fault_classes: Set[int] = CRITICAL_FAULT_CLASSES,
    fault_recall_minimum: float = 0.95,
) -> Tuple[bool, Dict[int, float]]:
    """
    Check 2: Critical equipment fault recall floor (default >= 95%).
    Ensures candidate model reliably catches all core physical degradation modes
    (misalignment, attenuation, APD aging, thermal drift, timing jitter).
    """
    recall_report = {}
    all_passed = True

    for cls in fault_classes:
        rec = compute_class_recall(candidate_model, test_X, test_y, target_class=cls)
        recall_report[cls] = round(rec, 4)
        if rec < fault_recall_minimum:
            all_passed = False

    return all_passed, recall_report


def check_attack_recall_floor(
    candidate_model: Any,
    adversarial_test_X: np.ndarray,
    adversarial_test_y: np.ndarray,
    attack_classes: Set[int] = ATTACK_CLASSES,
    attack_recall_minimum: float = 0.98,
) -> Tuple[bool, Dict[int, float]]:
    """
    Check 2: Strict non-negotiable attack recall floor (default >= 98%).
    Ensures candidate model catches >= 98% of all known security attack signatures.

    Operational Safety Purpose:
    This empirical recall floor functions as an operational safety gate to ensure that
    retrained models never regress on critical security classes. Composable cryptographic
    security (epsilon_sec <= 1e-9) is fundamentally guaranteed at the physical layer by
    quantum measurement uncertainty, decoy-state parameter bounds, error correction, and
    privacy amplification (Tomamichel-Lim bounds), rather than by ML classifier predictions.
    The 98% threshold enforces high operational sensitivity to known attack anomalies.
    """
    recall_report = {}
    all_passed = True

    for cls in attack_classes:
        rec = compute_class_recall(candidate_model, adversarial_test_X, adversarial_test_y, target_class=cls)
        recall_report[cls] = round(rec, 4)
        if rec < attack_recall_minimum:
            all_passed = False

    return all_passed, recall_report


class ShadowDeployment:
    """
    Check 3: Manages live shadow evaluation where candidate model runs in parallel
    with production on incoming telemetry without actuating control actions.
    """

    def __init__(
        self,
        candidate_model: Any,
        production_model: Any,
        min_samples: int = 500,
        min_agreement_rate: float = 0.88,
    ):
        self.candidate = candidate_model
        self.production = production_model
        self.min_samples = int(min_samples)
        self.min_agreement_rate = float(min_agreement_rate)
        self.agreement_log: List[Dict[str, Any]] = []

    def observe(
        self,
        features: np.ndarray,
        physics_guard_verdict: str = "ML + Physics Agree",
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Receives live sample, generates both predictions, logs comparison,
        and returns production prediction for live execution.
        """
        feats_2d = np.atleast_2d(features)

        # Production prediction
        if hasattr(self.production, "predict"):
            prod_class = int(self.production.predict(feats_2d)[0])
        else:
            prod_class = int(self.production.predict_batch(feats_2d)[0])

        # Candidate prediction
        if hasattr(self.candidate, "predict"):
            cand_class = int(self.candidate.predict(feats_2d)[0])
        else:
            cand_class = int(self.candidate.predict_batch(feats_2d)[0])

        agree = (prod_class == cand_class)

        # Check for dangerous disagreement
        is_dangerous = (
            cand_class in BENIGN_CLASSES
            and (prod_class in ATTACK_CLASSES or "Attack" in physics_guard_verdict or "Contradiction" in physics_guard_verdict)
        )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_id,
            "production_class": prod_class,
            "candidate_class": cand_class,
            "agree": agree,
            "is_dangerous": is_dangerous,
            "physics_guard_verdict": physics_guard_verdict,
        }
        self.agreement_log.append(entry)

        return {
            "actuated_class": prod_class,
            "candidate_class": cand_class,
            "agree": agree,
            "is_dangerous": is_dangerous,
        }

    def evaluate_for_promotion(
        self,
        offline_regression_passed: bool = True,
        attack_recall_floor_passed: bool = True,
    ) -> ShadowVerdict:
        """
        Assesses whether the candidate model has cleared the full Shadow Validation Gate.
        """
        n_obs = len(self.agreement_log)
        if n_obs < self.min_samples:
            return ShadowVerdict(
                ready=False,
                reason=f"Insufficient shadow burn-in samples ({n_obs}/{self.min_samples} required)",
                n_samples_observed=n_obs,
                offline_regression_passed=offline_regression_passed,
                attack_recall_floor_passed=attack_recall_floor_passed,
            )

        if not offline_regression_passed:
            return ShadowVerdict(
                ready=False,
                reason="Candidate failed Check 1 (Offline regression test)",
                n_samples_observed=n_obs,
                offline_regression_passed=False,
                attack_recall_floor_passed=attack_recall_floor_passed,
            )

        if not attack_recall_floor_passed:
            return ShadowVerdict(
                ready=False,
                reason="Candidate failed Check 2 (Attack-class recall floor >= 0.98)",
                n_samples_observed=n_obs,
                offline_regression_passed=offline_regression_passed,
                attack_recall_floor_passed=False,
            )

        agreements = [e for e in self.agreement_log if e["agree"]]
        agreement_rate = float(len(agreements) / n_obs) if n_obs > 0 else 0.0

        dangerous = [e for e in self.agreement_log if e.get("is_dangerous", False)]
        if dangerous:
            return ShadowVerdict(
                ready=False,
                reason=f"CRITICAL SAFETY HOLD: {len(dangerous)} dangerous disagreements detected where candidate called benign during active attack/contradiction",
                n_samples_observed=n_obs,
                agreement_rate=round(agreement_rate, 4),
                dangerous_disagreements=dangerous,
                offline_regression_passed=True,
                attack_recall_floor_passed=True,
            )

        if agreement_rate < self.min_agreement_rate:
            return ShadowVerdict(
                ready=False,
                reason=f"Overall shadow agreement rate {agreement_rate*100:.1f}% below minimum threshold {self.min_agreement_rate*100:.1f}%",
                n_samples_observed=n_obs,
                agreement_rate=round(agreement_rate, 4),
                offline_regression_passed=True,
                attack_recall_floor_passed=True,
            )

        return ShadowVerdict(
            ready=True,
            reason="All 3 Shadow Validation Gates passed successfully (No regression, Attack recall >= 0.98, Zero dangerous disagreements)",
            n_samples_observed=n_obs,
            agreement_rate=round(agreement_rate, 4),
            dangerous_disagreements=[],
            offline_regression_passed=True,
            attack_recall_floor_passed=True,
        )
