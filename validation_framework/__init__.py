"""
Validation Framework Module for VECTOR Q (Layer 10).
Contains comprehensive test suites for Validation Sets A, B, C, D and latency profiler.
"""

from validation_framework.validation_set_a_physics import run_validation_set_a
from validation_framework.validation_set_b_fault_matrix import run_validation_set_b
from validation_framework.validation_set_c_hil_interface import run_validation_set_c
from validation_framework.validation_set_d_cross_domain import run_validation_set_d
from validation_framework.latency_profiler import profile_pipeline_latencies

__all__ = [
    "run_validation_set_a",
    "run_validation_set_b",
    "run_validation_set_c",
    "run_validation_set_d",
    "profile_pipeline_latencies",
]
