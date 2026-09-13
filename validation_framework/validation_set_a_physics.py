"""
Layer 10 — Validation Set A: Analytical Physics Unit Tests
Validates BB84 / Decoy-State optical equations against known theoretical bounds.
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import math
import numpy as np
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.qkd_system_parameters import QKDPhysicsConfig
from physics_engine.optical_channel_models import (
    compute_channel_transmittance,
    compute_dark_count_probability,
    compute_signal_yield,
    compute_quantum_bit_error_rate,
    compute_secret_key_rate,
    compute_thermal_dark_count_rate,
    compute_raw_count_rate,
    binary_shannon_entropy,
)


def run_validation_set_a() -> Dict[str, Any]:
    """
    Executes analytical physics equation tests against theoretical physical limits.
    
    Returns:
        Dict summarizing test results, status, and verification metrics.
    """
    test_results = []
    
    # Test 1: Channel Transmittance at Zero Loss
    eta_zero = compute_channel_transmittance(0.0, 25.0)
    t1_pass = math.isclose(eta_zero, 1.0, rel_tol=1e-5)
    test_results.append({
        "test_name": "Zero Attenuation Transmittance (alpha=0 dB/km -> eta=1.0)",
        "passed": t1_pass,
        "value": eta_zero,
        "expected": 1.0
    })
    
    # Test 2: Standard SMF-28 Fiber Loss at 25 km (alpha = 0.20 dB/km -> Total Loss = 5.0 dB -> eta = 10^-0.5 ~ 0.3162)
    eta_std = compute_channel_transmittance(0.20, 25.0)
    expected_eta_std = 10.0 ** (-5.0 / 10.0)
    t2_pass = math.isclose(eta_std, expected_eta_std, rel_tol=1e-4)
    test_results.append({
        "test_name": "Standard SMF-28 25km Loss (alpha=0.20 dB/km -> eta=0.3162)",
        "passed": t2_pass,
        "value": eta_std,
        "expected": expected_eta_std
    })
    
    # Test 3: Dark Count Probability scaling (500 Hz at 100 MHz rep rate -> Y0 = 5e-6)
    y0_calc = compute_dark_count_probability(500.0, 1.0e8)
    t3_pass = math.isclose(y0_calc, 5.0e-6, rel_tol=1e-6)
    test_results.append({
        "test_name": "Dark Count Probability Scaling (500 Hz @ 100 MHz -> Y0=5e-6)",
        "passed": t3_pass,
        "value": y0_calc,
        "expected": 5.0e-6
    })
    
    # Test 4: Pure Noise Limit (S = 0 -> QBER must equal 0.50 exactly)
    qber_noise = compute_quantum_bit_error_rate(y0=5e-6, signal_yield=0.0, optical_error_rate=0.0075)
    t4_pass = math.isclose(qber_noise, 0.50, abs_tol=1e-5)
    test_results.append({
        "test_name": "Pure Noise Limit (Signal Yield S=0 -> QBER=0.50)",
        "passed": t4_pass,
        "value": qber_noise,
        "expected": 0.50
    })
    
    # Test 5: Perfect Visibility & Zero Dark Counts (V=1.0, e_opt=0, Y0=0 -> QBER=0.0)
    qber_ideal = compute_quantum_bit_error_rate(y0=0.0, signal_yield=0.05, optical_error_rate=0.0)
    t5_pass = math.isclose(qber_ideal, 0.0, abs_tol=1e-5)
    test_results.append({
        "test_name": "Ideal Optical Limit (V=1.0, Y0=0 -> QBER=0.00)",
        "passed": t5_pass,
        "value": qber_ideal,
        "expected": 0.0
    })
    
    # Test 6: Critical Security Abort Limit (QBER = 11.0% -> SKR = 0.0 strictly)
    skr_abort, _ = compute_secret_key_rate(
        qber=0.110,
        repetition_rate_hz=1.0e8,
        y0=5.0e-6,
        eta_channel=0.3162,
        eta_bob=0.15,
        mean_photon_number=0.50,
        critical_qber_threshold=0.110
    )
    t6_pass = (skr_abort == 0.0)
    test_results.append({
        "test_name": "Shor-Preskill / GLLP 11% Cutoff Bound (QBER=11% -> SKR=0.0 bps)",
        "passed": t6_pass,
        "value": skr_abort,
        "expected": 0.0
    })
    
    # Test 7: Intercept-Resend Error Injection (gamma=1.0 full intercept -> +25% error)
    qber_ir = compute_quantum_bit_error_rate(
        y0=0.0,
        signal_yield=0.05,
        optical_error_rate=0.0,
        eavesdropping_fraction=1.0
    )
    t7_pass = math.isclose(qber_ir, 0.25, abs_tol=1e-5)
    test_results.append({
        "test_name": "Full Intercept-Resend Attack (gamma=1.0 -> QBER=0.25)",
        "passed": t7_pass,
        "value": qber_ir,
        "expected": 0.25
    })
    
    # Test 8: Arrhenius APD Temperature Doubling (+10 deg C -> DCR doubles)
    dcr_warm = compute_thermal_dark_count_rate(
        nominal_dcr_hz=500.0,
        current_temp_celsius=-30.0,
        nominal_temp_celsius=-40.0,
        doubling_temp_delta=10.0
    )
    t8_pass = math.isclose(dcr_warm, 1000.0, rel_tol=1e-5)
    test_results.append({
        "test_name": "Arrhenius APD Dark Count Doubling (+10 C rise -> 2x DCR = 1000 Hz)",
        "passed": t8_pass,
        "value": dcr_warm,
        "expected": 1000.0
    })

    # Test 9: Detector Blinding Optical Saturation Bound (Makarov et al., 2009)
    from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator
    emu = QuantumTelemetryEmulator(random_seed=42)
    emu.inject_fault("Detector Blinding", intensity=1.0)
    blind_sample = emu.step()
    t9_pass = (blind_sample.raw_counts_hz > 1.0e7) and (blind_sample.qber < 0.020) and (blind_sample.skr_bps == 0.0)
    test_results.append({
        "test_name": "Detector Blinding Saturation (R_raw > 10 Mcps, QBER < 2%, SKR = 0)",
        "passed": t9_pass,
        "value": f"R={blind_sample.raw_counts_hz:.0f} cps, QBER={blind_sample.qber*100:.2f}%, SKR={blind_sample.skr_bps}",
        "expected": "R > 10,000,000 cps, QBER < 2.0%, SKR == 0.0"
    })

    # Test 10: Photon Number Splitting Attack Zero-SKR Bound (GLLP Security Limit)
    emu.inject_fault("Photon Number Splitting", intensity=0.8)
    pns_sample = emu.step()
    t10_pass = (pns_sample.skr_bps == 0.0) and (pns_sample.qber < 0.060) and (pns_sample.raw_counts_hz > 1000.0)
    test_results.append({
        "test_name": "PNS Attack Decoy-State Collapse (SKR = 0, QBER < 6%, R_raw > 1000)",
        "passed": t10_pass,
        "value": f"SKR={pns_sample.skr_bps}, QBER={pns_sample.qber*100:.2f}%, R={pns_sample.raw_counts_hz:.0f}",
        "expected": "SKR == 0.0, QBER < 6.0%, R > 1000"
    })

    # Test 11: Time-Shift Attack Gating Asymmetry (Zhao et al., 2008)
    emu.inject_fault("Time-Shift Attack", intensity=0.9)
    ts_sample = emu.step()
    t11_pass = (ts_sample.timing_jitter_ps > 120.0) and (ts_sample.qber > 0.065) and (ts_sample.fiber_loss_db_per_km < 0.25)
    test_results.append({
        "test_name": "Time-Shift Attack Gating Asymmetry (Jitter > 120 ps, QBER > 6.5%, Loss Nominal)",
        "passed": t11_pass,
        "value": f"Jitter={ts_sample.timing_jitter_ps:.1f} ps, QBER={ts_sample.qber*100:.2f}%, Loss={ts_sample.fiber_loss_db_per_km:.2f} dB/km",
        "expected": "Jitter > 120 ps, QBER > 6.5%, Loss < 0.25 dB/km"
    })
    
    all_passed = all(t["passed"] for t in test_results)
    
    return {
        "suite_name": "Validation Set A (Analytical Physics Verification)",
        "all_passed": all_passed,
        "total_tests": len(test_results),
        "passed_tests": sum(1 for t in test_results if t["passed"]),
        "details": test_results,
    }


if __name__ == "__main__":
    res = run_validation_set_a()
    print(f"\n--- {res['suite_name']} ---")
    print(f"Status: {'PASSED' if res['all_passed'] else 'FAILED'} ({res['passed_tests']}/{res['total_tests']})")
    for d in res["details"]:
        status = "[PASS]" if d["passed"] else "[FAIL]"
        print(f"  {status} {d['test_name']}: {d['value']} (Expected: {d['expected']})")
