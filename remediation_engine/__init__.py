"""
Remediation and Mitigation Engine for VECTOR Q.
Contains operational remediation mappings and closed-form physical impact estimator.
"""

from remediation_engine.mitigation_optimizer import RemediationOptimizer, RemediationRecommendation

__all__ = ["RemediationOptimizer", "RemediationRecommendation"]
