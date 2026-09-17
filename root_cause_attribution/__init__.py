"""
Root Cause Attribution and Explainable AI Module for VECTOR Q.
Contains LightGBM multi-class root-cause classifier and SHAP TreeExplainer.
"""

from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier, RootCauseAttributionResult
from root_cause_attribution.shap_feature_explainer import SHAPFeatureExplainer, FivePointExplainabilityReport

__all__ = [
    "LightGBMRootCauseClassifier",
    "RootCauseAttributionResult",
    "SHAPFeatureExplainer",
    "FivePointExplainabilityReport",
]
