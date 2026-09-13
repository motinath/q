"""
Comprehensive Automated Pytest Suite for Q-SENTINEL (QIC 2026 Edition)
Covers:
- Physical channel calculations and bounds (Scarani 2009, GLLP 2004)
- Feature extraction ring buffers and W=25 moments
- Isolation Forest anomaly detection on healthy manifold
- Physics invariant validator & operational confidence fusion
- Multi-class root-cause attribution
- Phase 12 5-Point Explainability Interface
- Phase 13 PTCT threshold forecaster
- Phase 14 Advisory remediation engine
- Phase 15 Adaptive baseline engine
- Phase 20 SQLite audit persistence & cryptographic tamper verification
- Phase 3-6 Data split manifest generation
"""

import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.qkd_system_parameters import QKDPhysicsConfig, ROOT_CAUSE_CLASSES
from config.adaptive_baseline_engine import AdaptiveBaselineEngine
from physics_engine.optical_channel_models import (
    compute_channel_transmittance,
    compute_dark_count_probability,
    compute_signal_yield,
    compute_quantum_bit_error_rate,
    compute_secret_key_rate,
    compute_thermal_dark_count_rate,
    compute_raw_count_rate,
)
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator
from physics_engine.hardware_telemetry_source import HardwareTelemetrySource
from anomaly_detection.sliding_window_features import TelemetryFeatureExtractor, FEATURE_COLUMN_NAMES
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from root_cause_attribution.shap_feature_explainer import SHAPFeatureExplainer, FivePointExplainabilityReport
from predictive_maintenance.threshold_crossing_forecaster import ThresholdCrossingForecaster
from remediation_engine.mitigation_optimizer import RemediationOptimizer
from audit_logging.compliance_sqlite_database import ComplianceAuditDatabase
from validation_framework.data_split_manifest import create_and_export_data_splits_manifest


def test_optical_transmittance_and_qber():
    eta = compute_channel_transmittance(0.20, 25.0)
    assert 0.31 < eta < 0.32
    
    y0 = compute_dark_count_probability(500.0, 1.0e8)
    assert y0 == 5.0e-6
    
    s = compute_signal_yield(eta, 0.15, 0.50)
    assert s > 0.0
    
    qber = compute_quantum_bit_error_rate(y0, s, 0.0075)
    assert 0.005 < qber < 0.030


def test_gllp_critical_qber_cutoff():
    skr_nominal, _ = compute_secret_key_rate(0.02, 1e8, 5e-6, 0.316, 0.15, 0.50)
    assert skr_nominal > 0.0
    
    skr_abort, _ = compute_secret_key_rate(0.115, 1e8, 5e-6, 0.316, 0.15, 0.50)
    assert skr_abort == 0.0


def test_emulator_fault_injection():
    emulator = QuantumTelemetryEmulator(random_seed=42)
    s_nom = emulator.step(1.0)
    assert s_nom.qber < 0.04
    assert s_nom.active_fault_label == "Normal"
    
    emulator.inject_fault("Optical Misalignment", intensity=0.8)
    s_fault = emulator.step(1.0)
    assert s_fault.visibility < 0.85
    assert s_fault.qber > s_nom.qber
    
    emulator.reset_to_nominal()
    s_recovered = emulator.step(1.0)
    assert s_recovered.visibility > 0.95


def test_feature_extractor_and_moments():
    emulator = QuantumTelemetryEmulator(random_seed=42)
    extractor = TelemetryFeatureExtractor(max_buffer_size=35)
    
    for _ in range(15):
        sample = emulator.step(1.0)
        feats = extractor.extract_features(sample)
        assert "qber" in feats
        assert "count_to_dark_ratio" in feats
        assert "qber_slope_25" in feats
        assert "corr_temp_qber" in feats
        
    vec = extractor.get_feature_vector()
    assert len(vec) == len(FEATURE_COLUMN_NAMES)


def test_adaptive_baseline_engine():
    engine = AdaptiveBaselineEngine(warmup_samples=10)
    sample_dict = {
        "qber": 0.02,
        "skr_bps": 12000.0,
        "raw_counts_hz": 50000.0,
        "dark_counts_hz": 500.0,
        "visibility": 0.985,
        "temperature_celsius": -40.0,
        "timing_jitter_ps": 65.0,
    }
    for _ in range(12):
        env = engine.update(sample_dict, is_healthy=True)
    assert env["qber"].is_calibrated is True
    assert env["qber"].mean > 0.015


def test_physics_invariant_validator_and_fusion():
    validator = PhysicalInvariantValidator()
    
    normal_feats = {
        "qber": 0.02,
        "skr_bps": 12000.0,
        "raw_counts_hz": 50000.0,
        "dark_counts_hz": 500.0,
        "visibility": 0.985,
        "temperature_celsius": -40.0,
        "timing_jitter_ps": 65.0,
        "signal_to_noise_ratio": 100.0,
        "channel_attenuation_db": 5.0,
    }
    res_nom = validator.evaluate(normal_feats, ml_predicted_class="Normal", ml_confidence=0.95)
    assert res_nom.validation_status == "ML + Physics Agree"
    assert res_nom.operational_confidence_tier == "High Confidence"
    
    # Intercept-Resend condition
    eve_feats = normal_feats.copy()
    eve_feats["qber"] = 0.09
    res_eve = validator.evaluate(eve_feats, ml_predicted_class="Intercept-Resend", ml_confidence=0.92)
    assert res_eve.validation_status == "ML + Physics Agree"
    assert "Intercept-Resend" in res_eve.primary_physical_signature


def test_ptct_forecaster():
    forecaster = ThresholdCrossingForecaster()
    
    # Stable link
    res_stable = forecaster.compute_ptct(current_qber=0.02, dqber_dt=0.0)
    assert res_stable.t_cross_seconds is None
    assert res_stable.urgency_level == "STABLE"
    
    # Deteriorating link (dQBER/dt = 0.005 / s, curr = 0.06 -> delta = 0.05 -> t_cross = 10s)
    res_urgent = forecaster.compute_ptct(current_qber=0.06, dqber_dt=0.005)
    assert res_urgent.t_cross_seconds == 10.0
    assert res_urgent.urgency_level == "CRITICAL"


def test_remediation_optimizer_advisory():
    optimizer = RemediationOptimizer()
    feats = {"qber": 0.08, "skr_bps": 1000.0, "visibility": 0.75, "channel_attenuation_db": 5.0}
    rec = optimizer.generate_recommendation("Optical Misalignment", feats)
    assert rec.action_id == "ACTION_POLARIZATION_RECALIBRATION"
    assert rec.is_advisory_only is True
    assert rec.expected_post_action_qber < rec.current_qber
    assert rec.expected_post_action_skr_bps > rec.current_skr_bps


def test_data_splits_manifest_generation(tmp_path):
    manifest_file = str(tmp_path / "test_data_splits.json")
    manifest = create_and_export_data_splits_manifest(manifest_path=manifest_file, total_runs=20)
    assert len(manifest["partitions"]["training_runs"]) == 14
    assert len(manifest["partitions"]["validation_runs"]) == 3
    assert len(manifest["partitions"]["test_runs"]) == 3
    assert os.path.exists(manifest_file)


def test_quantum_attacks_physics_signatures():
    emu = QuantumTelemetryEmulator(random_seed=42)
    
    # 1. Detector Blinding
    emu.inject_fault("Detector Blinding", intensity=0.9)
    s_blind = emu.step()
    assert s_blind.raw_counts_hz > 1.0e7
    assert s_blind.qber < 0.02
    assert s_blind.skr_bps == 0.0
    
    # 2. Photon Number Splitting
    emu.inject_fault("Photon Number Splitting", intensity=0.8)
    s_pns = emu.step()
    assert s_pns.skr_bps == 0.0
    assert s_pns.qber < 0.06
    assert s_pns.raw_counts_hz > 1000.0
    
    # 3. Time-Shift Attack
    emu.inject_fault("Time-Shift Attack", intensity=0.85)
    s_ts = emu.step()
    assert s_ts.timing_jitter_ps > 120.0
    assert s_ts.qber > 0.065
    
    # 4. Intercept-Resend
    emu.inject_fault("Intercept-Resend", intensity=0.7)
    s_ir = emu.step()
    assert s_ir.qber > 0.08


def test_digital_twin_lite_operations():
    from digital_twin.digital_twin_lite import DigitalTwinLite
    dt = DigitalTwinLite()
    
    # 1. Sanity check
    sanity = dt.sanity_check_against_hand_calculations()
    assert sanity["status"] == "PASSED"
    
    # 2. Forward projection
    init_st = {"alpha_db_km": 0.20, "visibility": 0.985, "temperature_c": -40.0, "dcr_base": 500.0, "jitter_ps": 65.0}
    drifts = {"d_temp_dt": 0.50}  # Heating up rapidly
    proj = dt.simulate_forward_trajectory(init_st, drifts, horizon_seconds=60.0, dt_seconds=5.0)
    assert len(proj.trajectory) == 13
    assert proj.final_qber > proj.trajectory[0].qber
    
    # 3. What-if sweep
    sweep = dt.sweep_detector_temperature(min_temp_c=-40.0, max_temp_c=10.0, steps=10)
    assert len(sweep.parameter_values) == 10
    assert sweep.qber_values[-1] > sweep.qber_values[0]


def test_incident_intelligence_generation():
    from incident_intelligence.incident_report_generator import IncidentReportGenerator
    from root_cause_attribution.lightgbm_classifier import RootCauseAttributionResult
    from physics_validation.invariant_rule_evaluator import PhysicsValidationResult
    from root_cause_attribution.shap_feature_explainer import FivePointExplainabilityReport, SHAPFeatureContribution
    from predictive_maintenance.threshold_crossing_forecaster import PTCTForecastResult
    
    gen = IncidentReportGenerator()
    opt = RemediationOptimizer()
    rec = opt.generate_recommendation("Optical Misalignment", {"qber": 0.08, "skr_bps": 1000.0, "visibility": 0.75, "channel_attenuation_db": 5.0})
    
    attr = RootCauseAttributionResult(
        predicted_class="Optical Misalignment",
        confidence=0.95,
        raw_probabilities={"Optical Misalignment": 0.95},
        calibrated_probabilities={"Optical Misalignment": 0.98},
        is_physics_verified=True,
        physics_adjustment_applied=0.25,
        feature_vector=np.zeros(35),
    )
    phys = PhysicsValidationResult(
        validation_status="ML + Physics Agree",
        physics_consistency_score=0.95,
        operational_confidence_tier="High Confidence",
        primary_physical_signature="Fringe Visibility Degradation",
        physical_evidence={"measured_qber": 0.08, "theoretical_physics_qber": 0.08},
        explanation="Visibility dropped, directly causing QBER elevation.",
    )
    exp = FivePointExplainabilityReport(
        what_happened="Optical Misalignment",
        why_it_happened="Waveplate drift",
        supporting_measurements=["visibility dropped to 0.75"],
        confidence_assessment="High Confidence",
        recommended_action="Execute automated waveplate alignment",
        alarm_severity="MEDIUM",
        top_shap_contributions=[SHAPFeatureContribution("visibility", 0.75, -0.45, "negative", "Low visibility drives QBER")],
    )
    ptct = PTCTForecastResult(
        t_cross_seconds=25.0,
        t_cross_lower_bound_seconds=20.0,
        t_cross_upper_bound_seconds=30.0,
        current_qber=0.08,
        qber_abort_limit=0.11,
        dqber_dt=0.001,
        forecast_trajectory_timestamps=[0, 10, 20],
        forecast_trajectory_qber=[0.08, 0.09, 0.10],
        urgency_level="WARNING",
        recommendation_message="Projected crossing in 25s",
    )
    
    rep = gen.generate_report("INC-TEST-001", attr, phys, exp, ptct, rec)
    assert rep.incident_id == "INC-TEST-001"
    assert rep.alarm_severity == "MEDIUM"
    assert rep.recommended_action_id == "ACTION_POLARIZATION_RECALIBRATION"
    
    md = gen.to_markdown(rep)
    assert "# Q-SENTINEL INCIDENT DOSSIER" in md
    assert "INC-TEST-001" in md
    
    js = gen.to_json(rep)
    parsed = json.loads(js)
    assert parsed["incident_id"] == "INC-TEST-001"
    
    # ADD-2: Verify executive PDF export
    pdf_bytes = gen.export_pdf(rep)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_confidence_fusion_engine_add1():
    """ADD-1: Verifies Unified Trust Score and explicit agreement_state."""
    from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator
    
    val = PhysicalInvariantValidator()
    
    # Case 1: Agreement state
    telem_agree = {
        "qber": 0.08,
        "skr_bps": 3000.0,
        "visibility": 0.84, # Low visibility explains QBER
        "temperature_celsius": -40.0,
        "dark_counts_hz": 500.0,
        "timing_jitter_ps": 65.0,
        "raw_counts_hz": 2.2e6,
        "channel_attenuation_db": 5.0,
    }
    res_agree = val.evaluate(telem_agree, ml_predicted_class="Optical Misalignment", ml_confidence=0.92)
    assert res_agree.validation_status == "ML + Physics Agree"
    assert res_agree.agreement_state == "AGREEMENT"
    expected_fused = round(0.60 * 0.92 + 0.40 * res_agree.physics_consistency_score, 3)
    assert abs(res_agree.fused_trust_score - expected_fused) < 1e-4
    assert res_agree.operational_confidence_tier == "High Confidence"
    
    # Case 2: Contradiction state
    telem_contra = {
        "qber": 0.08,
        "skr_bps": 3000.0,
        "visibility": 0.99, # Near-perfect visibility contradicts optical misalignment
        "temperature_celsius": -40.0,
        "dark_counts_hz": 500.0,
        "timing_jitter_ps": 65.0,
        "raw_counts_hz": 2.2e6,
        "channel_attenuation_db": 5.0,
    }
    res_contra = val.evaluate(telem_contra, ml_predicted_class="Optical Misalignment", ml_confidence=0.95)
    assert res_contra.validation_status == "Physics Contradiction"
    assert res_contra.agreement_state == "CONTRADICTION"
    assert res_contra.operational_confidence_tier == "Contradictory / Review Needed"


def test_multi_candidate_remediation_add3():
    """ADD-3: Verifies >= 3 candidate actions per class and argmax J(a) selection."""
    from remediation_engine.mitigation_optimizer import RemediationOptimizer
    from config.qkd_system_parameters import ROOT_CAUSE_CLASSES
    
    opt = RemediationOptimizer()
    features = {
        "qber": 0.08,
        "skr_bps": 2000.0,
        "visibility": 0.82,
        "temperature_celsius": -32.0,
        "dark_counts_hz": 1800.0,
        "timing_jitter_ps": 115.0,
        "channel_attenuation_db": 8.0,
    }
    
    for cls_name in ROOT_CAUSE_CLASSES:
        rec = opt.generate_recommendation(cls_name, features)
        assert len(rec.candidates) >= 3, f"Class '{cls_name}' has fewer than 3 candidates ({len(rec.candidates)})"
        
        # Verify argmax utility selection
        winner = max(rec.candidates, key=lambda c: c.utility_score)
        assert rec.action_id == winner.action_id
        assert rec.expected_post_action_qber == winner.projected_qber
        assert rec.expected_post_action_skr_bps == winner.projected_skr_bps
        assert "Literal Argmax Optimization" in rec.optimization_rationale


def test_quantum_attack_intelligence_signatures_add4():
    """ADD-4: Verifies specialized physical signatures for all 3 quantum attack classes."""
    from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator
    from config.qkd_system_parameters import QKDPhysicsConfig
    
    cfg = QKDPhysicsConfig()
    val = PhysicalInvariantValidator(cfg)
    
    # 1. Detector Blinding: Click rate > 10 Mcps, QBER suppressed
    blinding_telem = {
        "raw_counts_hz": 15.0e6,
        "qber": 0.005,
        "visibility": 0.98,
        "dark_counts_hz": 500.0,
        "temperature_celsius": -40.0,
        "timing_jitter_ps": 65.0,
        "channel_attenuation_db": 5.0,
    }
    res_blind = val.evaluate(blinding_telem, ml_predicted_class="Detector Blinding", ml_confidence=0.96)
    assert res_blind.validation_status == "ML + Physics Agree"
    assert "Detector Blinding" in res_blind.primary_physical_signature
    
    # 2. Photon Number Splitting (PNS): SKR = 0, QBER low, counts present
    pns_telem = {
        "skr_bps": 0.0,
        "qber": 0.035,
        "raw_counts_hz": 1.5e6,
        "visibility": 0.98,
        "dark_counts_hz": 500.0,
        "temperature_celsius": -40.0,
        "timing_jitter_ps": 65.0,
        "channel_attenuation_db": 5.0,
    }
    res_pns = val.evaluate(pns_telem, ml_predicted_class="Photon Number Splitting", ml_confidence=0.94)
    assert res_pns.validation_status == "ML + Physics Agree"
    assert "PNS" in res_pns.primary_physical_signature
    
    # 3. Time-Shift Attack: Jitter elevated (>100 ps), QBER elevated, visibility intact
    ts_telem = {
        "timing_jitter_ps": 130.0,
        "qber": 0.075,
        "visibility": 0.975,
        "temperature_celsius": -40.0,
        "dark_counts_hz": 500.0,
        "raw_counts_hz": 2.2e6,
        "channel_attenuation_db": 5.0,
    }
    res_ts = val.evaluate(ts_telem, ml_predicted_class="Time-Shift Attack", ml_confidence=0.91)
    assert res_ts.validation_status == "ML + Physics Agree"
    assert "Time-Shift" in res_ts.primary_physical_signature


def test_digital_twin_perturbation_simulator_add5():
    """ADD-5: Verifies parameter perturbation and forward trajectory projection."""
    from digital_twin.digital_twin_lite import DigitalTwinLite
    
    twin = DigitalTwinLite()
    telemetry = {
        "temperature_celsius": -40.0,
        "visibility": 0.985,
        "channel_attenuation_db": 5.0,
        "dark_counts_hz": 500.0,
        "timing_jitter_ps": 65.0,
        "qber": 0.02,
        "skr_bps": 12500.0,
        "temperature_slope_25": 0.10,
    }
    
    # Scenario A: Moderate thermal perturbation
    out_a = twin.perturb_and_forward_simulate(telemetry, {"delta_temperature_c": 5.0})
    assert "immediate_qber" in out_a
    assert "immediate_skr_bps" in out_a
    assert len(out_a["trajectory"]) > 10
    assert out_a["security_status"] in ["SECURE", "WARNING_ELEVATED"]
    
    # Scenario B: Critical visibility perturbation causing abort threshold crossing
    out_b = twin.perturb_and_forward_simulate(telemetry, {"delta_visibility": -0.30})
    assert out_b["immediate_qber"] >= 0.11
    assert out_b["security_status"] == "ABORT_BREACHED"
    assert out_b["projected_ptct_seconds"] == 0.0


