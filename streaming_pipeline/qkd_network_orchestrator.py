"""
Unified Streaming Pipeline Orchestrator for Q-SENTINEL (Layers 1 through 9)
Synchronously routes telemetry from physics emulator / HIL across ML, XAI, forecasting, and audit stores.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import time
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
        
        self.audit_db = audit_db or ComplianceAuditDatabase()
        self.enable_audit_logging = enable_audit_logging
        self.step_counter: int = 0

    def attach_trained_models(
        self,
        anomaly_detector: IsolationForestAnomalyDetector,
        classifier: LightGBMRootCauseClassifier,
    ) -> None:
        """Attaches trained ML models and initializes the SHAP explainer."""
        self.anomaly_detector = anomaly_detector
        self.classifier = classifier
        self.explainer = SHAPFeatureExplainer(classifier)

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
        
        # Layer 2: ML Anomaly Detection (100% of samples)
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            anomaly_res = self.anomaly_detector.predict_sample(features)
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
            
        # Layer 6: Predictive Maintenance (PTCT)
        dqber_dt = features.get("qber_slope_25", 0.0)
        ptct_res = self.forecaster.compute_ptct(
            current_qber=sample.qber,
            dqber_dt=dqber_dt,
        )
        
        # Layer 7: Remediation & Impact Estimation
        remediation_res = self.remediation_engine.generate_recommendation(
            fault_class=attribution_res.predicted_class,
            current_features=features,
        )
        
        # Layer 9: Audit Logging (Log on anomaly or periodic every 10 steps)
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
            
            event_id = self.audit_db.log_event(
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
                action_executed="MONITORED",
                operator="Q_SENTINEL_AI_SUPERVISOR",
            )
            
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
