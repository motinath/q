"""
Phase 10 & 11: Physics Consistency Validator & Operational Confidence Fusion
Evaluates deterministic quantum optical conservation laws and combines ML probability with physical plausibility.
States: 'ML + Physics Agree', 'ML Prediction Uncertain', 'Physics Contradiction'
Governing Standards: ETSI GS QKD 014 / GLLP / Makarov et al. / Zhao et al.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from config.qkd_system_parameters import QKDPhysicsConfig


@dataclass
class PhysicsValidationResult:
    """Represents the output of Phase 10 Physics Consistency Validation & Phase 11 Confidence Fusion."""
    validation_status: str              # 'ML + Physics Agree', 'ML Prediction Uncertain', 'Physics Contradiction'
    physics_consistency_score: float    # Plausibility score in [0.0, 1.0]
    operational_confidence_tier: str    # 'High Confidence', 'Moderate Confidence', 'Contradictory / Review Needed'
    primary_physical_signature: str     # Physical signature identified
    physical_evidence: Dict[str, Any]    # Numerical evidence metrics
    explanation: str                     # Plain language physical rationale
    physics_confidence_modifier: float = 0.0
    fused_trust_score: float = 0.0       # ADD-1: Fused Trust Score (0.60 * ML + 0.40 * Physics)
    agreement_state: str = "AGREEMENT"   # ADD-1: 'AGREEMENT', 'CONTRADICTION', 'UNCERTAIN'
    ml_confidence: float = 0.0           # ML attribution confidence score


class PhysicalInvariantValidator:
    """
    Deterministic Physics Invariant Validator & Confidence Fusion Engine.
    Acts as safety boundary, plausibility checker, and confidence modifier for ML predictions.
    Evaluates invariants for all 10 fault & quantum attack classes.
    """

    def __init__(self, config: Optional[QKDPhysicsConfig] = None):
        self.config = config or QKDPhysicsConfig()

    def evaluate(
        self,
        features: Dict[str, float],
        ml_predicted_class: str = "Normal",
        ml_confidence: float = 0.90,
    ) -> PhysicsValidationResult:
        """
        Cross-checks ML root-cause prediction against physical conservation laws and fuses confidence.
        """
        qber = float(features.get("qber", 0.02))
        skr = float(features.get("skr_bps", 10000.0))
        raw_cnt = float(features.get("raw_counts_hz", 2.3e6))
        dcr = float(features.get("dark_counts_hz", 500.0))
        vis = float(features.get("visibility", 0.985))
        temp = float(features.get("temperature_celsius", -40.0))
        jitter = float(features.get("timing_jitter_ps", 65.0))
        snr = float(features.get("signal_to_noise_ratio", 4500.0))
        loss_db = float(features.get("channel_attenuation_db", 5.0))
        
        # Physical Invariant Quantities:
        expected_e_opt = (1.0 - vis) / 2.0
        y0_approx = dcr / self.config.pulse_repetition_rate_hz
        total_yield = max(1e-15, raw_cnt / self.config.pulse_repetition_rate_hz)
        expected_dark_qber = (0.5 * y0_approx) / total_yield
        expected_physics_qber = expected_e_opt + expected_dark_qber
        unexplained_qber = qber - expected_physics_qber
        
        evidence: Dict[str, Any] = {
            "measured_qber": round(qber, 4),
            "expected_optical_error_e_opt": round(expected_e_opt, 4),
            "expected_dark_count_qber": round(expected_dark_qber, 5),
            "theoretical_physics_qber": round(expected_physics_qber, 4),
            "unexplained_qber_surplus": round(unexplained_qber, 4),
            "signal_to_noise_ratio": round(snr, 1),
            "visibility_measured": round(vis, 4),
            "detector_temp_celsius": round(temp, 1),
            "timing_jitter_ps": round(jitter, 1),
            "total_channel_loss_db": round(loss_db, 2),
            "raw_counts_hz": round(raw_cnt, 1),
            "secret_key_rate_bps": round(skr, 1),
        }
        
        # Check agreement with predicted ML class across 10 classes
        status = "ML Prediction Uncertain"
        consistency_score = 0.50
        signature = "General Operational Transition"
        explanation = "Measurements evaluated against baseline quantum optical invariants."
        
        if ml_predicted_class == "Intercept-Resend":
            if unexplained_qber > 0.020 and vis >= 0.960 and dcr <= 1500.0 and raw_cnt >= 20000.0:
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Quantum State Disturbance (Intercept-Resend Eavesdropping)"
                explanation = f"Physical Consistency Verified: QBER surplus of {unexplained_qber*100:.2f}% without optical misalignment (V={vis:.3f}) or detector noise surge matches active BB84 state disturbance."
            elif vis < 0.90 or dcr > 2500.0:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "Physical Noise Contradiction"
                explanation = "Contradiction: ML predicted Intercept-Resend, but observed QBER is fully explained by optical misalignment or detector noise."

        elif ml_predicted_class in ["Channel Attenuation Event", "Channel Attenuation"]:
            if raw_cnt < 1.2e6 and (loss_db > 6.5 or snr < 1000.0):
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Channel Attenuation & Signal Flux Collapse"
                explanation = f"Physical Consistency Verified: Raw photon flux collapsed to {raw_cnt:.0f} Hz (Loss {loss_db:.1f} dB), elevating QBER due to dark count fraction dominance."
            elif raw_cnt > 2.0e6 and loss_db < 6.0:
                status = "Physics Contradiction"
                consistency_score = 0.25
                signature = "Count Rate Contradiction"
                explanation = f"Contradiction: ML predicted Channel Attenuation Event, but raw photon flux is robust ({raw_cnt:.0f} Hz) and channel loss is nominal ({loss_db:.1f} dB)."

        elif ml_predicted_class == "Optical Misalignment":
            if vis < 0.960 and raw_cnt >= 1.0e6:
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Polarization / Phase Visibility Degradation"
                explanation = f"Physical Consistency Verified: Fringe visibility dropped to {vis:.3f} (optical error {expected_e_opt*100:.2f}%), directly accounting for QBER without photon loss."
            elif vis >= 0.980:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "Visibility Invariant Contradiction"
                explanation = f"Contradiction: ML predicted Optical Misalignment, but fringe visibility is near-ideal ({vis:.3f})."

        elif ml_predicted_class == "Thermal Drift":
            if temp > -32.0 and dcr > 1500.0:
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Arrhenius Detector Thermal Runaway"
                explanation = f"Physical Consistency Verified: APD temperature ({temp:.1f} C) is elevated above nominal (-40 C), driving exponential thermal dark carrier generation."
            elif temp <= -38.0:
                status = "Physics Contradiction"
                consistency_score = 0.25
                signature = "Temperature Invariant Contradiction"
                explanation = f"Contradiction: ML predicted Thermal Drift, but APD temperature ({temp:.1f} C) is nominal."

        elif ml_predicted_class in ["Detector APD Degradation", "APD Aging"]:
            if dcr > 1200.0 and temp <= -35.0:
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Aged SPAD Trap-Assisted Carrier Generation"
                explanation = f"Physical Consistency Verified: Dark count rate elevated ({dcr:.0f} Hz) despite nominal cold temperature ({temp:.1f} C), indicating semiconductor trap degradation."
            elif dcr < 800.0:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "Dark Count Contradiction"
                explanation = "Contradiction: ML predicted APD Degradation, but dark count rate is within nominal envelope."

        elif ml_predicted_class == "Timing Jitter":
            if jitter > 110.0 and qber < 0.065:
                status = "ML + Physics Agree"
                consistency_score = 0.90
                signature = "Receiver Sync Clock Phase Jitter"
                explanation = f"Physical Consistency Verified: Gating jitter ({jitter:.1f} ps) exceeds nominal envelope (65 ps) without attack error elevation."
            elif jitter <= 85.0:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "Jitter Contradiction"
                explanation = f"Contradiction: ML predicted Timing Jitter, but measured jitter ({jitter:.1f} ps) is nominal."

        elif ml_predicted_class == "Detector Blinding":
            # Invariant 5: Raw count rate explodes past saturation threshold while QBER is suppressed
            if raw_cnt > self.config.detector_blinding_counts_threshold_hz and qber < self.config.detector_blinding_max_qber:
                status = "ML + Physics Agree"
                consistency_score = 0.98
                signature = "Optical Saturation (Detector Blinding Attack)"
                explanation = f"Physical Consistency Verified: Raw photon clicks exploded to {raw_cnt:.0f} Hz (>10 Mcps) with near-zero QBER ({qber*100:.2f}%), indicating classical CW laser APD saturation (Makarov et al.)."
            elif raw_cnt < 5.0e6:
                status = "Physics Contradiction"
                consistency_score = 0.15
                signature = "Blinding Flux Contradiction"
                explanation = f"Contradiction: ML predicted Detector Blinding, but raw click rate ({raw_cnt:.0f} Hz) is far below optical saturation limits."

        elif ml_predicted_class == "Photon Number Splitting":
            # Invariant 6: Distilled SKR is 0 despite low QBER and healthy count rate
            if skr == 0.0 and qber < self.config.pns_max_qber_threshold and raw_cnt > 1.0e5:
                status = "ML + Physics Agree"
                consistency_score = 0.96
                signature = "Decoy-State Yield Collapse (PNS Attack)"
                explanation = f"Physical Consistency Verified: Distilled SKR collapsed to 0.0 bps while QBER is normal ({qber*100:.2f}%) and channel flux is present, confirming multi-photon pulse splitting detected by decoy bounds."
            elif skr > 500.0 or qber >= self.config.qber_abort_threshold:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "PNS Key Rate Contradiction"
                explanation = "Contradiction: ML predicted PNS attack, but positive secret key is actively distilling or QBER exceeded standard abort threshold."

        elif ml_predicted_class == "Time-Shift Attack":
            # Invariant 7: Timing jitter elevated AND QBER elevated without visibility loss or thermal drift
            if jitter > self.config.time_shift_jitter_threshold_ps and qber >= self.config.time_shift_min_qber and vis >= 0.960 and temp <= -35.0:
                status = "ML + Physics Agree"
                consistency_score = 0.95
                signature = "Gating Window Phase Asymmetry (Time-Shift Attack)"
                explanation = f"Physical Consistency Verified: Elevated QBER ({qber*100:.2f}%) coupled with timing jitter offset ({jitter:.1f} ps) without optical misalignment (V={vis:.3f}) matches detector efficiency mismatch attack (Zhao et al.)."
            elif jitter < 90.0 or vis < 0.92:
                status = "Physics Contradiction"
                consistency_score = 0.20
                signature = "Time-Shift Contradiction"
                explanation = f"Contradiction: ML predicted Time-Shift attack, but timing jitter ({jitter:.1f} ps) is nominal or error is explained by optical visibility drop."

        elif ml_predicted_class == "Normal":
            if qber < 0.040 and vis >= 0.970 and temp <= -37.0 and dcr <= 1000.0 and raw_cnt > 20000.0:
                status = "ML + Physics Agree"
                consistency_score = 0.98
                signature = "Nominal Optical Channel Baseline"
                explanation = "All quantum optical observables conform strictly to nominal ETSI GS QKD 014 bounds."

        # Phase 11 & ADD-1: Operational Confidence Fusion
        # Fused Trust Score = 0.60 * ML_Confidence + 0.40 * Physics_Consistency
        fused_score = 0.60 * ml_confidence + 0.40 * consistency_score
        
        # Explicit Agreement State (ADD-1)
        if status == "ML + Physics Agree":
            agreement_state = "AGREEMENT"
        elif status == "Physics Contradiction":
            agreement_state = "CONTRADICTION"
        else:
            agreement_state = "UNCERTAIN"
        
        if status == "ML + Physics Agree" and fused_score >= 0.80:
            confidence_tier = "High Confidence"
        elif status == "Physics Contradiction":
            confidence_tier = "Contradictory / Review Needed"
        else:
            confidence_tier = "Moderate Confidence"
            
        # Confidence modifier for Bayesian ML adjustment
        mod = 0.25 if status == "ML + Physics Agree" else (-0.35 if status == "Physics Contradiction" else 0.0)

        return PhysicsValidationResult(
            validation_status=status,
            physics_consistency_score=round(consistency_score, 3),
            operational_confidence_tier=confidence_tier,
            primary_physical_signature=signature,
            physical_evidence=evidence,
            explanation=explanation,
            physics_confidence_modifier=mod,
            fused_trust_score=round(fused_score, 3),
            agreement_state=agreement_state,
            ml_confidence=round(ml_confidence, 3),
        )
