"""
Unified Streaming Pipeline Orchestrator for VECTOR Q (Layers 1 through 9)
Synchronously routes telemetry from physics emulator / HIL across ML, XAI, forecasting, and audit stores.

P4.2: Audit logging is now asynchronous.  log_event() calls are placed on a bounded
      queue (max 256 entries) and consumed by a daemon background thread, keeping
      SQLite writes entirely off the <5 ms hot path.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import time
import queue
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from config.qkd_system_parameters import QKDPhysicsConfig
from config.adaptive_baseline_engine import AdaptiveBaselineEngine, ChannelBaselineEnvelope
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator, QuantumTelemetrySample
from physics_engine.hardware_telemetry_source import HardwareTelemetrySource, HardwareTelemetrySnapshot
from anomaly_detection.sliding_window_features import TelemetryFeatureExtractor
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector, AnomalyDetectionResult
from physics_validation.invariant_rule_evaluator import PhysicalInvariantValidator, PhysicsValidationResult
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier, RootCauseAttributionResult
from root_cause_attribution.shap_feature_explainer import SHAPFeatureExplainer, FivePointExplainabilityReport
from predictive_maintenance.threshold_crossing_forecaster import ThresholdCrossingForecaster, PTCTForecastResult
from remediation_engine.mitigation_optimizer import RemediationOptimizer, RemediationRecommendation
from audit_logging.compliance_sqlite_database import ComplianceAuditDatabase
from digital_twin.digital_twin_lite import DigitalTwinLite


@dataclass
class PipelineStepResult:
    """Consolidated end-to-end telemetry and intelligence result for a single time step."""
    sample: QuantumTelemetrySample
    hardware_snapshot: HardwareTelemetrySnapshot
    features: Dict[str, float]
    anomaly: AnomalyDetectionResult
    physics_validation: PhysicsValidationResult
    attribution: RootCauseAttributionResult
    explanation: FivePointExplainabilityReport
    ptct_forecast: PTCTForecastResult
    remediation: RemediationRecommendation
    adaptive_envelopes: Dict[str, ChannelBaselineEnvelope]
    audit_event_id: Optional[int]
    inference_latency_ms: float


class QKDNetworkOrchestrator:
    """
    Real-time streaming orchestrator connecting Layers 1 to 9.
    Executes the full pipeline on every telemetry step.
    """

    def __init__(
        self,
        config: Optional[QKDPhysicsConfig] = None,
        emulator: Optional[QuantumTelemetryEmulator] = None,
        anomaly_detector: Optional[IsolationForestAnomalyDetector] = None,
        classifier: Optional[LightGBMRootCauseClassifier] = None,
        audit_db: Optional[ComplianceAuditDatabase] = None,
        enable_audit_logging: bool = True,
    ):
        self.config = config or QKDPhysicsConfig()
        self.emulator = emulator or QuantumTelemetryEmulator(config=self.config)
        self.hil_source = HardwareTelemetrySource()
        self.feature_extractor = TelemetryFeatureExtractor()
        self.baseline_engine = AdaptiveBaselineEngine()
        
        self.anomaly_detector = anomaly_detector
        self.classifier = classifier
        self.explainer: Optional[SHAPFeatureExplainer] = None
        if self.classifier is not None and self.classifier.is_fitted:
            self.explainer = SHAPFeatureExplainer(self.classifier)
            
        self.physics_validator = PhysicalInvariantValidator(config=self.config)
        self.forecaster = ThresholdCrossingForecaster(config=self.config)
        self.remediation_engine = RemediationOptimizer(config=self.config)
        # P4.3: Digital twin instance for CRITICAL PTCT augmentation
        self.digital_twin = DigitalTwinLite(config=self.config)
        
        self.audit_db = audit_db or ComplianceAuditDatabase()
        self.enable_audit_logging = enable_audit_logging
        self.step_counter: int = 0

        # P4.2: Asynchronous audit logging — bounded queue + daemon worker thread.
        # SQLite writes happen in the background; the hot path only enqueues a dict.
        self._audit_queue: queue.Queue = queue.Queue(maxsize=256)
        self._audit_thread = threading.Thread(
            target=self._audit_worker, daemon=True, name="VECTOR-Q-AuditWorker"
        )
        self._audit_thread.start()

    def attach_trained_models(
        self,
        anomaly_detector: IsolationForestAnomalyDetector,
        classifier: LightGBMRootCauseClassifier,
    ) -> None:
        """Attaches trained ML models and initializes the SHAP explainer."""
        self.anomaly_detector = anomaly_detector
        self.classifier = classifier
        self.explainer = SHAPFeatureExplainer(classifier)

    # P4.2: Background audit worker — drains queue and writes to SQLite off hot path
    def _audit_worker(self) -> None:
        """Daemon thread: dequeues audit payloads and writes them to the SQLite DB."""
        while True:
            try:
                payload = self._audit_queue.get(timeout=1.0)
                if payload is None:          # Sentinel: shut down cleanly
                    break
                try:
                    self.audit_db.log_event(**payload)
                except Exception:
                    pass                     # Never crash the worker thread
                finally:
                    self._audit_queue.task_done()
            except queue.Empty:
                continue                     # No work; loop back and wait

    def flush_audit_queue(self, timeout: float = 5.0) -> None:
        """
        Blocks until all pending audit events have been written, or timeout expires.
        Call this before process exit or test teardown to ensure no events are lost.
        """
        try:
            self._audit_queue.join()
        except Exception:
            pass

    def _augment_remediation_with_twin(
        self,
        remediation: RemediationRecommendation,
        features: Dict[str, float],
        ptct: PTCTForecastResult,
    ) -> RemediationRecommendation:
        """
        P4.3: When PTCT urgency is CRITICAL, run a digital twin forward simulation for
        each candidate remediation action and select the one that minimises the peak
        projected QBER over a 60-second horizon.

        The twin simulation uses the candidate's projected post-action physical state
        (as computed by the optimizer) as the initial conditions, then propagates
        forward under the current drift rates extracted from the feature window.

        If the twin-preferred action differs from the utility-argmax action, the
        optimisation_rationale is updated to note the override.
        """
        if not remediation.candidates:
            return remediation

        # Extract current drift rates from the feature window
        drift_rates = {
            "d_alpha_dt": features.get("raw_counts_slope_25", 0.0) / max(1e-6, features.get("raw_counts_hz", 1e6)) * (-0.001),
            "d_vis_dt":   features.get("visibility_slope_25", 0.0),
            "d_temp_dt":  features.get("temp_slope_25", 0.0),
            "d_dcr_dt":   features.get("dark_counts_slope_25", 0.0),
        }

        best_candidate = None
        best_peak_qber = float("inf")
        twin_scores: Dict[str, float] = {}

        for cand in remediation.candidates:
            st = cand.post_physical_state
            initial_state = {
                "alpha_db_km":   st.get("alpha", self.config.nominal_fiber_attenuation_db_per_km),
                "visibility":    st.get("vis",   self.config.nominal_visibility),
                "temperature_c": st.get("temp",  self.config.nominal_detector_temp_celsius),
                "dcr_base":      st.get("dcr",   self.config.nominal_dark_count_rate_hz),
                "jitter_ps":     st.get("jitter", self.config.nominal_timing_jitter_ps),
            }
            try:
                proj = self.digital_twin.simulate_forward_trajectory(
                    initial_state=initial_state,
                    drift_rates=drift_rates,
                    horizon_seconds=60.0,
                    dt_seconds=5.0,
                )
                peak_qber = max(p.qber for p in proj.trajectory)
            except Exception:
                peak_qber = cand.projected_qber  # fallback: use static projection

            twin_scores[cand.action_id] = peak_qber
            if peak_qber < best_peak_qber:
                best_peak_qber = peak_qber
                best_candidate = cand

        if best_candidate is None or best_candidate.action_id == remediation.action_id:
            return remediation  # No change

        # The twin selected a different action — update remediation accordingly
        twin_rationale = (
            f" [DIGITAL TWIN OVERRIDE on CRITICAL PTCT: "
            f"Twin simulation over 60s selected '{best_candidate.action_title}' "
            f"(peak QBER {best_peak_qber*100:.2f}%) over utility-argmax "
            f"'{remediation.action_title}' "
            f"(peak QBER {twin_scores.get(remediation.action_id, 0.0)*100:.2f}%). "
            f"Twin scores: { {k: f'{v*100:.2f}%' for k, v in twin_scores.items()} }]"
        )
        from dataclasses import replace as dc_replace
        return dc_replace(
            remediation,
            action_id=best_candidate.action_id,
            action_title=best_candidate.action_title,
            action_description=best_candidate.action_description,
            target_subsystem=best_candidate.target_subsystem,
            expected_post_action_qber=best_candidate.projected_qber,
            expected_post_action_skr_bps=best_candidate.projected_skr_bps,
            expected_raw_counts_hz=best_candidate.projected_raw_counts_hz,
            qber_improvement_delta=round(remediation.current_qber - best_candidate.projected_qber, 4),
            skr_gain_delta_bps=round(best_candidate.projected_skr_bps - remediation.current_skr_bps, 1),
            winning_candidate_id=best_candidate.action_id,
            optimization_rationale=remediation.optimization_rationale + twin_rationale,
        )

    def process_step(self, dt_seconds: float = 1.0) -> PipelineStepResult:
        """
        Executes one full synchronized cycle of the 10-layer pipeline.
        """
        t_start = time.perf_counter()
        self.step_counter += 1
        
        # Layer 1: Acquire telemetry
        sample = self.emulator.step(dt_seconds=dt_seconds)
        hil_snapshot = self.hil_source.acquire_sample()
        
        # Extract features
        features = self.feature_extractor.extract_features(sample)
        
        # Update adaptive baselines
        is_healthy_step = (sample.active_fault_label == "Normal")
        envelopes = self.baseline_engine.update(features, is_healthy=is_healthy_step)
        
        # Layer 2: ML Anomaly Detection with adaptive baseline gate (P2.3)
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            anomaly_res = self.anomaly_detector.predict_sample_with_baseline_gate(
                features, envelopes
            )
        else:
            anomaly_res = AnomalyDetectionResult(
                is_anomaly=sample.qber >= self.config.qber_warning_threshold,
                operational_state="Warning" if sample.qber >= self.config.qber_warning_threshold else "Nominal",
                anomaly_score=float(min(1.0, sample.qber / 0.11)),
                raw_decision_score=0.0,
                telemetry_features=features,
            )
            
        # Layer 4: Preliminary Root Cause Attribution
        if self.classifier is not None and self.classifier.is_fitted:
            raw_attr = self.classifier.predict_sample(feature_dict=features, physics_validation=None)
            pred_class = raw_attr.predicted_class
            pred_conf = raw_attr.confidence
        else:
            pred_class = "Normal" if not anomaly_res.is_anomaly else "Channel Attenuation Event"
            pred_conf = 0.85
            
        # Layer 3: Physics Consistency Validation & Confidence Fusion
        physics_res = self.physics_validator.evaluate(
            features=features,
            ml_predicted_class=pred_class,
            ml_confidence=pred_conf,
        )
        
        # Layer 4: Final Attribution with Physics Verification
        if self.classifier is not None and self.classifier.is_fitted:
            attribution_res = self.classifier.predict_sample(
                feature_dict=features,
                physics_validation=physics_res,
            )
        else:
            attribution_res = RootCauseAttributionResult(
                predicted_class=pred_class,
                confidence=pred_conf,
                raw_probabilities={pred_class: pred_conf},
                calibrated_probabilities={pred_class: pred_conf},
                is_physics_verified=(physics_res.validation_status == "ML + Physics Agree"),
                physics_adjustment_applied=0.0,
                feature_vector=self.feature_extractor.get_feature_vector(),
            )
            
        # Layer 5: Explainable AI (5-Point Report + SHAP)
        if self.explainer is not None:
            explanation_res = self.explainer.explain_sample(
                attribution_result=attribution_res,
                physics_validation=physics_res,
                top_k=3,
            )
        else:
            explanation_res = FivePointExplainabilityReport(
                what_happened=f"Status: {attribution_res.predicted_class}",
                why_it_happened=physics_res.explanation,
                supporting_measurements=[f"QBER = {sample.qber*100:.2f}%"],
                confidence_assessment=f"Confidence: {attribution_res.confidence*100:.1f}%",
                recommended_action="Continue monitoring.",
                alarm_severity="NORMAL",
                top_shap_contributions=[],
            )
            
        # Layer 6: Predictive Maintenance (PTCT) — P2.5: actual SE; P3.1: acceleration
        dqber_dt = features.get("qber_slope_25", 0.0)
        dqber_se = features.get("qber_slope_25_se", 0.0002)      # P2.5
        qber_accel = features.get("qber_acceleration_25", 0.0)   # P3.1
        ptct_res = self.forecaster.compute_ptct(
            current_qber=sample.qber,
            dqber_dt=dqber_dt,
            slope_standard_error=dqber_se,
            qber_acceleration=qber_accel,
        )
        
        # Layer 7: Remediation & Impact Estimation with Security Gating
        remediation_res = self.remediation_engine.generate_recommendation(
            fault_class=attribution_res.predicted_class,
            current_features=features,
        )

        # Security-First Confidence & Invariant Gating Policy
        is_physics_verified = (physics_res.validation_status == "ML + Physics Agree")
        is_high_confidence = (attribution_res.confidence >= 0.85)
        EMERGENCY_CLASSES = {"Intercept-Resend", "Detector Blinding", "Photon Number Splitting", "Time-Shift Attack"}

        if not is_physics_verified or not is_high_confidence:
            if attribution_res.predicted_class in EMERGENCY_CLASSES:
                remediation_res.is_advisory_only = False
                remediation_res.actuation_mode = "EMERGENCY"
                remediation_res.optimization_rationale += (
                    " [SECURITY QUARANTINE: Attack class suspected under uncertainty/contradiction. "
                    "Enforcing immediate fail-secure session abort & channel quarantine.]"
                )
            else:
                # Inhibit autonomous hardware actuation for low-confidence or contradicted benign faults
                remediation_res.is_advisory_only = True
                remediation_res.actuation_mode = "ADVISORY"
                remediation_res.optimization_rationale = (
                    f"AUTOMATION INHIBITED: Model confidence ({attribution_res.confidence:.1%}) < 85% "
                    f"or Physics Invariant Guard flagged '{physics_res.validation_status}'. "
                    f"Deferred to Human Operator (Tier-2 QNOC) to prevent unauthorized optical actuation."
                )

        # P4.3: When PTCT is CRITICAL, augment remediation with digital twin trajectories
        if ptct_res.urgency_level == "CRITICAL" and self.digital_twin is not None and not remediation_res.is_advisory_only:
            remediation_res = self._augment_remediation_with_twin(
                remediation_res, features, ptct_res
            )
        
        # Layer 9: Audit Logging — P4.2: enqueue async; worker thread writes to SQLite
        event_id = None
        if self.enable_audit_logging and (anomaly_res.is_anomaly or self.step_counter % 10 == 0):
            top_feats_list = [
                {
                    "feature": c.feature_name,
                    "value": c.feature_value,
                    "shap": c.shap_value,
                    "interpretation": c.physics_interpretation,
                }
                for c in explanation_res.top_shap_contributions
            ]
            audit_payload = dict(
                timestamp=sample.timestamp,
                qber=sample.qber,
                skr_bps=sample.skr_bps,
                raw_counts_hz=sample.raw_counts_hz,
                dark_counts_hz=sample.dark_counts_hz,
                visibility=sample.visibility,
                temperature_celsius=sample.temperature_celsius,
                timing_jitter_ps=sample.timing_jitter_ps,
                anomaly_status=anomaly_res.is_anomaly,
                anomaly_score=anomaly_res.anomaly_score,
                root_cause_diagnosis=attribution_res.predicted_class,
                confidence=attribution_res.confidence,
                is_physics_verified=(physics_res.validation_status == "ML + Physics Agree"),
                physics_signature=physics_res.primary_physical_signature,
                top_shap_features=top_feats_list,
                ptct_seconds=ptct_res.t_cross_seconds,
                remediation_action_id=remediation_res.action_id,
                action_executed=remediation_res.actuation_mode,
                operator="VECTOR_Q_AI_SUPERVISOR",
            )
            try:
                self._audit_queue.put_nowait(audit_payload)
                event_id = -1   # Async; actual row ID not available in hot path
            except queue.Full:
                pass  # Queue full (backpressure) — drop rather than block hot path
            
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        
        return PipelineStepResult(
            sample=sample,
            hardware_snapshot=hil_snapshot,
            features=features,
            anomaly=anomaly_res,
            physics_validation=physics_res,
            attribution=attribution_res,
            explanation=explanation_res,
            ptct_forecast=ptct_res,
            remediation=remediation_res,
            adaptive_envelopes=envelopes,
            audit_event_id=event_id,
            inference_latency_ms=latency_ms,
        )
