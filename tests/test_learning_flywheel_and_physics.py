"""
Unit and Integration Tests for Continuous Learning Flywheel, Finite-Key Security, and Adaptive Decoy Defense
Validates:
1. OperatorFeedbackStore ground-truth capture and uncertainty filtering
2. ADWIN & Page-Hinkley Confidence Drift Monitoring
3. 3-stage Shadow Validation Gate (No regression, Attack recall >= 0.98, Dangerous disagreement)
4. ModelRegistry promotion and cryptographic audit rollback
5. Tomamichel-Lim-Curty-Lo Finite-Key Security Analysis & Invariant #8
6. Thompson Sampling Adaptive Decoy Optimizer

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import tempfile
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_lifecycle.operator_feedback_capture import OperatorFeedbackStore, OperatorLabeledEvent
from model_lifecycle.drift_monitor import ModelDriftMonitor, ADWIN, PageHinkley, should_trigger_retrain
from model_lifecycle.promotion_gate import (
    check_no_regression,
    check_attack_recall_floor,
    ShadowDeployment,
    ShadowVerdict,
    ATTACK_CLASSES,
)
from model_lifecycle.model_registry import ModelRegistry, ModelMetadata, ModelStatus
from physics_engine.finite_key_analysis import compute_finite_key_bound, binary_entropy
from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator
from remediation_engine.adaptive_decoy_optimizer import AdaptiveDecoyOptimizer, DISCRETE_DECOY_CONFIGURATIONS


# ==============================================================================
# TEST 1: OPERATOR FEEDBACK STORE & UNCERTAINTY FILTERING
# ==============================================================================
def test_operator_feedback_store_and_filtering():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        store = OperatorFeedbackStore(db_path=db_path)

        dummy_feat = [0.02] * 33

        # 1. Add Certain event
        ev1 = store.record_resolution(
            link_id="LINK_A",
            feature_vector=dummy_feat,
            model_predicted_class=1,
            model_confidence=0.72,
            physics_guard_verdict="Advisory",
            operator_assigned_class=1,
            operator_confidence="certain",
            operator_id="TECH_1",
            resolution_method="physical_test",
            corroborating_evidence="OTDR fiber splice check",
        )

        # 2. Add Probable event
        ev2 = store.record_resolution(
            link_id="LINK_A",
            feature_vector=dummy_feat,
            model_predicted_class=2,
            model_confidence=0.68,
            physics_guard_verdict="Advisory",
            operator_assigned_class=2,
            operator_confidence="probable",
            operator_id="TECH_2",
            resolution_method="hardware_log_crosscheck",
            corroborating_evidence="Cooling log confirmed temp rise",
        )

        # 3. Add Uncertain event (CRITICAL SAFEGUARD: must be filtered out of training)
        ev3 = store.record_resolution(
            link_id="LINK_A",
            feature_vector=dummy_feat,
            model_predicted_class=0,
            model_confidence=0.50,
            physics_guard_verdict="ML Prediction Uncertain",
            operator_assigned_class=3,
            operator_confidence="uncertain",
            operator_id="TECH_3",
            resolution_method="visual_inspection",
            corroborating_evidence="Unconfirmed guess",
        )

        stats = store.get_statistics()
        assert stats["total_labeled_events"] == 3
        assert stats["unused_events"] == 3
        assert stats["training_eligible_events"] == 2  # Only certain + probable

        # Test filter
        eligible = store.get_unused_labels(min_confidence="probable")
        assert len(eligible) == 2
        eligible_ids = {e.event_id for e in eligible}
        assert ev1.event_id in eligible_ids
        assert ev2.event_id in eligible_ids
        assert ev3.event_id not in eligible_ids

        # Mark used
        store.mark_used_in_training([ev1.event_id, ev2.event_id], training_run_id="run_101")
        stats_after = store.get_statistics()
        assert stats_after["unused_events"] == 1
        assert stats_after["training_eligible_events"] == 0


# ==============================================================================
# TEST 2: DRIFT MONITOR (ADWIN & PAGE-HINKLEY)
# ==============================================================================
def test_drift_monitor_and_triggers():
    monitor = ModelDriftMonitor(adwin_delta=0.01, ph_threshold=10.0, ph_alpha=0.01)

    # 1. Nominal stable high confidence (0.92 - 0.98)
    for _ in range(40):
        sig = monitor.update(0.95)
    assert not sig.drift_detected
    assert sig.window_mean_confidence > 0.90

    # 2. Sudden confidence collapse to 0.40 (abrupt shift)
    drift_triggered = False
    for _ in range(25):
        sig = monitor.update(0.35)
        if sig.drift_detected:
            drift_triggered = True
            break
    assert drift_triggered
    assert sig.page_hinkley_triggered or sig.adwin_triggered

    # 3. Retraining trigger logic
    run, reason = should_trigger_retrain(
        labeled_buffer_size=15,
        drift_signal=sig,
        days_since_last_train=2.0,
        min_labels_threshold=50
    )
    assert run
    assert "Drift alarm" in reason

    # Insufficient buffer test
    run_no_buf, _ = should_trigger_retrain(
        labeled_buffer_size=2,
        drift_signal=sig,
        days_since_last_train=2.0,
        min_labels_threshold=50
    )
    assert not run_no_buf


# ==============================================================================
# TEST 3: SHADOW VALIDATION GATE & DANGEROUS DISAGREEMENTS
# ==============================================================================
class MockModel:
    def __init__(self, pred_class: int):
        self.pred_class = pred_class

    def predict(self, X):
        return np.full(len(X), self.pred_class)


def test_shadow_validation_gate():
    prod_model = MockModel(pred_class=6)  # Intercept-Resend (Attack)
    cand_model_safe = MockModel(pred_class=6)  # Agree on attack
    cand_model_dangerous = MockModel(pred_class=0)  # Predicts Normal during attack!

    # 1. Test Dangerous Disagreement
    shadow_bad = ShadowDeployment(
        candidate_model=cand_model_dangerous,
        production_model=prod_model,
        min_samples=10,
    )
    for _ in range(10):
        obs = shadow_bad.observe(features=np.zeros(33), physics_guard_verdict="Physics Contradiction")
        assert obs["is_dangerous"]

    verdict_bad = shadow_bad.evaluate_for_promotion(
        offline_regression_passed=True,
        attack_recall_floor_passed=True,
    )
    assert not verdict_bad.ready
    assert "CRITICAL SAFETY HOLD" in verdict_bad.reason
    assert len(verdict_bad.dangerous_disagreements) == 10

    # 2. Test Safe Shadow Passing
    shadow_good = ShadowDeployment(
        candidate_model=cand_model_safe,
        production_model=prod_model,
        min_samples=10,
    )
    for _ in range(10):
        obs = shadow_good.observe(features=np.zeros(33), physics_guard_verdict="ML + Physics Agree")
        assert not obs["is_dangerous"]

    verdict_good = shadow_good.evaluate_for_promotion(
        offline_regression_passed=True,
        attack_recall_floor_passed=True,
    )
    assert verdict_good.ready
    assert verdict_good.agreement_rate == 1.0


def test_attack_recall_floor():
    # Test set where true class is 6 (Intercept-Resend)
    X_test = np.zeros((20, 33))
    y_test = np.full(20, 6)

    perfect_model = MockModel(pred_class=6)
    passed_perfect, rep_perfect = check_attack_recall_floor(perfect_model, X_test, y_test, attack_classes={6})
    assert passed_perfect
    assert rep_perfect[6] == 1.0

    failing_model = MockModel(pred_class=0)
    passed_fail, rep_fail = check_attack_recall_floor(failing_model, X_test, y_test, attack_classes={6})
    assert not passed_fail
    assert rep_fail[6] == 0.0


# ==============================================================================
# TEST 4: MODEL REGISTRY & CRYPTOGRAPHIC ROLLBACK
# ==============================================================================
def test_model_registry_promotion_and_rollback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        reg_dir = os.path.join(tmpdir, "models")
        db_path = os.path.join(tmpdir, "audit.db")
        registry = ModelRegistry(registry_root=reg_dir, db_path=db_path)

        model_v1 = MockModel(pred_class=0)
        model_v2 = MockModel(pred_class=1)

        # Register and promote initial champion
        registry.register_model("lightgbm_classifier", "1.0.0", model_v1, status=ModelStatus.CHAMPION)
        registry.promote_to_champion("lightgbm_classifier", "1.0.0")

        champ = registry.get_champion("lightgbm_classifier")
        assert champ.version == "1.0.0"

        # Promote candidate v2 after shadow validation pass
        verdict = ShadowVerdict(ready=True, reason="Passed shadow burn-in")
        registry.promote(
            candidate_model=model_v2,
            candidate_version="2.0.0",
            shadow_verdict=verdict,
            promoted_by="automated_pipeline",
            n_field_labels=15
        )

        champ_promoted = registry.get_champion("lightgbm_classifier")
        assert champ_promoted.version == "2.0.0"
        assert champ_promoted.trained_on_n_field_labels == 15

        # Check audit trail
        trail = registry.get_audit_trail("lightgbm_classifier")
        assert len(trail) >= 1
        assert trail[0]["event_type"] == "MODEL_PROMOTION"
        assert trail[0]["version"] == "2.0.0"

        # Test Rollback to 1.0.0
        success = registry.rollback("1.0.0", model_type="lightgbm_classifier")
        assert success
        champ_reverted = registry.get_champion("lightgbm_classifier")
        assert champ_reverted.version == "1.0.0"

        trail_after = registry.get_audit_trail("lightgbm_classifier")
        assert trail_after[0]["event_type"] == "MODEL_ROLLBACK"
        assert trail_after[0]["version"] == "1.0.0"


# ==============================================================================
# TEST 5: FINITE-KEY ANALYSIS & INVARIANT #8
# ==============================================================================
def test_finite_key_analysis_and_invariant_8():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(0.5) == 1.0
    assert 0.0 < binary_entropy(0.05) < 0.5

    # 1. Nominal 25km link test
    res_nom = compute_finite_key_bound(
        qber=0.02,
        raw_counts_hz=2.0e6,
        block_size_N=10_000_000,
        channel_loss_db=5.0,
    )
    assert res_nom.is_secure_block_positive
    assert res_nom.finite_key_rate_bps > 0.0
    assert res_nom.finite_key_rate_bps <= res_nom.asymptotic_key_rate_bps
    assert 0.0 <= res_nom.finite_overhead_penalty_pct <= 100.0

    # 2. Critical error test (QBER = 12% > 11% abort threshold)
    res_abort = compute_finite_key_bound(
        qber=0.12,
        raw_counts_hz=2.0e6,
        block_size_N=10_000_000,
    )
    assert not res_abort.is_secure_block_positive
    assert res_abort.finite_key_rate_bps == 0.0

    # 3. Small block size test (N=10,000 -> statistical fluctuations dominate)
    res_small = compute_finite_key_bound(
        qber=0.04,
        raw_counts_hz=5.0e4,
        block_size_N=10_000,
    )
    assert not res_small.is_secure_block_positive

    # 4. Invariant #8 in PhysicalInvariantValidator
    validator = PhysicalInvariantValidator()
    # High QBER sample with claimed Normal diagnosis
    bad_features = {
        "qber": 0.13,
        "skr_bps": 5000.0,
        "raw_counts_hz": 1.5e6,
        "dark_counts_hz": 500.0,
        "visibility": 0.98,
        "temperature_celsius": -40.0,
        "timing_jitter_ps": 65.0,
        "signal_to_noise_ratio": 3000.0,
        "channel_attenuation_db": 5.0,
        "block_size_N": 10_000_000,
    }
    val_res = validator.evaluate(bad_features, ml_predicted_class="Normal", ml_confidence=0.90)
    assert val_res.validation_status == "Physics Contradiction"
    assert "Finite-Key" in val_res.primary_physical_signature


# ==============================================================================
# TEST 6: ADAPTIVE DECOY OPTIMIZER (THOMPSON SAMPLING BANDIT)
# ==============================================================================
def test_adaptive_decoy_optimizer():
    optimizer = AdaptiveDecoyOptimizer(cooldown_steps=5, random_seed=42)

    # Initial state
    summary = optimizer.get_summary()
    assert summary["current_arm_id"] == 0
    assert len(summary["arms"]) == 5

    # Step through 15 cycles with nominal telemetry
    for _ in range(15):
        optimizer.step(
            qber=0.02,
            skr_bps=12000.0,
            is_anomaly=False,
            finite_key_rate_bps=11500.0,
        )

    summary_after = optimizer.get_summary()
    total_pulls = sum(a["pulls"] for a in summary_after["arms"])
    assert total_pulls == 15

    # Safety Retreat Test: sudden QBER spike to 9% triggers immediate reset to arm 0
    optimizer.current_arm_id = 2  # artificially move to arm 2
    step_info = optimizer.step(
        qber=0.09,
        skr_bps=1000.0,
        is_anomaly=True,
    )
    assert step_info["action_taken"] == "SAFETY_RESET_TO_NOMINAL"
    assert optimizer.current_arm_id == 0
    assert step_info["is_advisory"] is True
    assert step_info["operational_mode"] == "ADVISORY_SHADOW"

    # Advisory safety lock test: hardware actuation must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        optimizer.request_hardware_actuation()
    assert "strictly prohibited in ADVISORY_SHADOW mode" in str(exc_info.value)


# ==============================================================================
# TEST 7: ADVANCED ORCHESTRATOR FLYWHEEL & SECURITY INTEGRATION
# ==============================================================================
def test_advanced_orchestrator_flywheel_and_security_integration():
    from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator

    orchestrator = AdvancedQKDOrchestrator(
        enable_causal_inference=False,
        enable_survival_analysis=False,
        enable_conformal_prediction=False,
        enable_gnn=False,
        enable_counterfactual=False,
        enable_drift_monitoring=True,
        enable_finite_key_analysis=True,
        enable_adaptive_decoy_optimizer=True,
    )

    result = orchestrator.process_single_timestep(dt_seconds=1.0)
    assert result.base_result is not None
    assert result.drift_signal is not None
    assert result.drift_signal.current_confidence >= 0.0
    assert result.finite_key_result is not None
    assert result.finite_key_result.block_size_N == 10_000_000
    assert result.decoy_recommendation is not None
    assert result.decoy_recommendation["is_advisory"] is True
    assert result.decoy_recommendation["operational_mode"] == "ADVISORY_SHADOW"

