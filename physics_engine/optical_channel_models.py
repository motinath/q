"""
Physical Optics and Quantum Key Distribution Mathematical Models
Governing Standards & Literature:
1. ETSI GS QKD 014 V1.1.1: Quantum Key Distribution (QKD); Protocol and data format
2. Scarani et al., "The security of practical quantum key distribution", Rev. Mod. Phys. 81, 1301 (2009)
3. Gottesman, Lo, Lütkenhaus, Preskill (GLLP), "Security of quantum key distribution with imperfect devices", Quant. Inf. Comput. 4, 325 (2004)
4. Ma, Qi, Zhao, Lo, "Practical decoy state quantum key distribution", Phys. Rev. A 72, 012326 (2005)
"""

import math
import numpy as np
from typing import Tuple, Dict, Any
from config.qkd_system_parameters import QKDPhysicsConfig


def compute_channel_transmittance(fiber_attenuation_db_per_km: float, fiber_length_km: float) -> float:
    """
    Computes optical fiber channel transmittance (eta_channel).
    Equation: eta_channel = 10^(-alpha * L / 10)
    
    References:
        - ITU-T G.652 standard single-mode optical fiber characteristics
        - ETSI GS QKD 014 optical transport specifications
        
    Args:
        fiber_attenuation_db_per_km: Fiber loss coefficient alpha (dB/km)
        fiber_length_km: Physical length of fiber link L (km)
        
    Returns:
        Transmittance fraction eta_channel in range (0.0, 1.0]
    """
    total_loss_db = max(0.0, fiber_attenuation_db_per_km * fiber_length_km)
    eta_channel = 10.0 ** (-total_loss_db / 10.0)
    return float(np.clip(eta_channel, 1e-15, 1.0))


def compute_dark_count_probability(dark_count_rate_hz: float, repetition_rate_hz: float) -> float:
    """
    Computes detector background / dark count probability per gate window (Y0).
    Equation: Y0 = DarkCounts / f_rep
    
    References:
        - Scarani et al., Rev. Mod. Phys. 81, 1301 (2009), Eq. (18)
        
    Args:
        dark_count_rate_hz: Detector dark count rate DCR (Hz / counts per second)
        repetition_rate_hz: Optical pulse transmission repetition frequency (Hz)
        
    Returns:
        Dark count probability per pulse Y0 in range [0.0, 1.0]
    """
    if repetition_rate_hz <= 0.0:
        raise ValueError("Repetition rate must be strictly positive.")
    y0 = dark_count_rate_hz / repetition_rate_hz
    return float(np.clip(y0, 1e-12, 1.0))


def compute_signal_yield(eta_channel: float, eta_bob: float, mean_photon_number: float) -> float:
    """
    Computes expected signal photon arrival yield (S) per pulse.
    Equation: S = eta_channel * eta_bob * mu
    
    References:
        - Ma et al., Phys. Rev. A 72, 012326 (2005), Section II.
        
    Args:
        eta_channel: Fiber transmission efficiency
        eta_bob: Bob optical transmission and detector quantum efficiency
        mean_photon_number: Mean photon intensity (mu) emitted by Alice
        
    Returns:
        Signal photon detection yield S
    """
    s = eta_channel * eta_bob * mean_photon_number
    return float(max(1e-15, s))


def compute_quantum_bit_error_rate(
    y0: float,
    signal_yield: float,
    optical_error_rate: float,
    eavesdropping_fraction: float = 0.0,
) -> float:
    """
    Computes the Quantum Bit Error Rate (QBER).
    Equation: QBER = (0.5 * Y0 + e_opt * S + 0.25 * gamma * S) / (Y0 + S)
    where:
        - 0.5 * Y0 accounts for random 50% errors from uncorrelated dark counts
        - e_opt * S accounts for optical misalignment (e_opt = (1 - V) / 2)
        - 0.25 * gamma * S accounts for Intercept-Resend eavesdropping errors
    
    References:
        - Scarani et al., Rev. Mod. Phys. 81, 1301 (2009), Eq. (20)
        - Gisin et al., Rev. Mod. Phys. 74, 145 (2002), Quantum Cryptography
        
    Args:
        y0: Dark count probability per pulse
        signal_yield: Signal yield S
        optical_error_rate: Intrinsic optical misalignment error rate e_opt
        eavesdropping_fraction: Fraction gamma of intercepted pulses (0.0 to 1.0)
        
    Returns:
        QBER value in range [0.0, 0.5]
    """
    total_yield = y0 + signal_yield
    if total_yield <= 0.0:
        return 0.50
    
    # 0.25 error contribution per intercepted pulse in BB84 intercept-resend attack
    effective_optical_error = optical_error_rate + 0.25 * eavesdropping_fraction
    error_clicks = 0.5 * y0 + effective_optical_error * signal_yield
    qber = error_clicks / total_yield
    return float(np.clip(qber, 0.0, 0.50))


def binary_shannon_entropy(p: float) -> float:
    """
    Computes binary Shannon entropy h2(p) = -p*log2(p) - (1-p)*log2(1-p).
    Strictly defined for p in [0, 1]. Returns 0.0 for p=0 or p=1.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def compute_secret_key_rate(
    qber: float,
    repetition_rate_hz: float,
    y0: float,
    eta_channel: float,
    eta_bob: float,
    mean_photon_number: float,
    error_correction_efficiency: float = 1.16,
    critical_qber_threshold: float = 0.110,
) -> Tuple[float, float]:
    """
    Computes asymptotic Secret Key Rate (SKR) under GLLP / decoy-state security bounds.
    
    Hard Security Constraint:
        If QBER >= critical_qber_threshold (11.0% Shor-Preskill bound for BB84),
        SKR is strictly set to 0.0 bps (No cryptographic key can be distilled).
        
    Equation:
        R_SKR = max(0, f_rep * [Q1 * (1 - h2(e1)) - Q_mu * f(QBER) * h2(QBER)])
        
    References:
        - Gottesman, Lo, Lütkenhaus, Preskill (GLLP), Quant. Inf. Comput. 4, 325 (2004)
        - Ma, Qi, Zhao, Lo, Phys. Rev. A 72, 012326 (2005)
        - Shor & Preskill, Phys. Rev. Lett. 85, 441 (2000)
        
    Args:
        qber: Evaluated Quantum Bit Error Rate
        repetition_rate_hz: Pulse repetition rate (Hz)
        y0: Dark count probability per pulse
        eta_channel: Fiber transmission efficiency
        eta_bob: Bob detector efficiency
        mean_photon_number: Mean photon intensity (mu)
        error_correction_efficiency: Cascade/LDPC reconciliation efficiency f(e) (default: 1.16)
        critical_qber_threshold: Maximum permissible QBER threshold (default: 0.110)
        
    Returns:
        Tuple (secret_key_rate_bps, secret_key_fraction_per_pulse)
    """
    # Hard physical security bound
    if qber >= critical_qber_threshold:
        return 0.0, 0.0
    
    overall_transmittance = eta_channel * eta_bob
    # Gain of signal state Q_mu
    q_mu = y0 + 1.0 - math.exp(-overall_transmittance * mean_photon_number)
    
    # Single photon yield Y1 and single photon gain Q1 (Poissonian weak coherent pulse)
    y1 = y0 + overall_transmittance
    q1 = mean_photon_number * math.exp(-mean_photon_number) * y1
    
    # Single photon phase error rate e1 estimate
    if y1 > 0:
        e1 = min(0.5, (0.5 * y0 + (qber * q_mu - 0.5 * y0)) / max(1e-12, y1))
    else:
        e1 = 0.5
    e1 = max(0.0, e1)
    
    # Privacy amplification term: 1 - h2(e1)
    privacy_amplification = 1.0 - binary_shannon_entropy(e1)
    
    # Error correction cost term: f(QBER) * h2(QBER)
    error_correction_cost = error_correction_efficiency * binary_shannon_entropy(qber)
    
    # Net secret key fraction per transmitted pulse
    key_fraction = q1 * privacy_amplification - q_mu * error_correction_cost
    
    if key_fraction <= 0.0:
        return 0.0, 0.0
        
    secret_key_rate_bps = float(repetition_rate_hz * key_fraction)
    return secret_key_rate_bps, float(key_fraction)


def compute_thermal_dark_count_rate(
    nominal_dcr_hz: float,
    current_temp_celsius: float,
    nominal_temp_celsius: float = -40.0,
    doubling_temp_delta: float = 10.0,
) -> float:
    """
    Computes temperature-dependent Dark Count Rate (DCR) for Avalanche Photodiodes (APDs).
    Thermal generation of dark carriers follows Arrhenius / Shockley-Read-Hall kinetics:
    DCR(T) = DCR_0 * 2^((T - T_0) / Delta_T_doubling)
    
    References:
        - Cova et al., "Avalanche photodiodes and quenching circuits for single-photon detection",
          Applied Optics 35, 1956-1976 (1996)
          
    Args:
        nominal_dcr_hz: DCR at nominal operating temperature
        current_temp_celsius: Current APD temperature in Celsius
        nominal_temp_celsius: Nominal calibrated temperature (default: -40 C)
        doubling_temp_delta: Temperature rise causing dark count doubling (default: 10 C)
        
    Returns:
        Expected thermal dark count rate (Hz)
    """
    temp_diff = current_temp_celsius - nominal_temp_celsius
    scaling_factor = 2.0 ** (temp_diff / doubling_temp_delta)
    return float(max(1.0, nominal_dcr_hz * scaling_factor))


def compute_raw_count_rate(repetition_rate_hz: float, y0: float, signal_yield: float) -> float:
    """
    Computes total raw photon detection count rate (clicks/second).
    Equation: R_raw = f_rep * (Y0 + S)
    
    Args:
        repetition_rate_hz: Repetition clock frequency (Hz)
        y0: Dark count probability per gate
        signal_yield: Signal arrival yield S
        
    Returns:
        Raw count rate (Hz / counts per second)
    """
    return float(repetition_rate_hz * (y0 + signal_yield))
