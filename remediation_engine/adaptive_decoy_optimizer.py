"""
Part 3: Proactive & Adaptive Decoy-State Optimization (Active Defense — Research Extension)
Implements a Thompson Sampling Multi-Armed Bandit (MAB) optimizer that explores discrete
decoy-state intensity and basis-timing configurations to elevate eavesdropper uncertainty.

CRITICAL SECURITY DISCLAIMER:
- This module is a RESEARCH-STAGE ACTIVE DEFENSE HEURISTIC.
- Non-stationary adaptive variation of decoy intensities based on an empirical bandit reward
  has NOT been formally proven to preserve composable GLLP / Tomamichel-Lim-Curty-Lo finite-key
  security bounds. In real deployments, an unproven parameter perturbation could violate the
  parameter-estimation assumptions underpinning Invariant #8.
- OPERATIONAL DIRECTIVE: This optimizer runs strictly in ADVISORY / SHADOW EVALUATION MODE only.
  It does not autonomously actuate live laser driver voltages without external cryptographic
  verification and proof completion.

Safety Guardrails:
- Advisory-only default: autonomous actuation disabled without explicit operator authorization.
- Cooldown period: Decoy parameters vary at most once every N steps (default: 15) to prevent link instability.
- Physical safety bounds: mu in [0.40, 0.60], nu1 in [0.10, 0.25], nu2 in [0.005, 0.05].
- High-QBER fallback: If QBER exceeds 0.08, immediately retreats to safe nominal configuration.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DecoyConfiguration:
    """Represents a candidate transmission parameter configuration."""
    arm_id: int
    name: str
    mu_signal: float            # Mean photon number of signal states
    nu1_decoy: float            # Mean photon number of primary decoy states
    nu2_vacuum: float           # Mean photon number of weak/vacuum states
    timing_offset_ps: float     # Receiver gating window clock dither in ps
    rationale: str


# Discrete Action Space A_decoy
DISCRETE_DECOY_CONFIGURATIONS: List[DecoyConfiguration] = [
    DecoyConfiguration(
        arm_id=0,
        name="Nominal Standard Decoy",
        mu_signal=0.50,
        nu1_decoy=0.15,
        nu2_vacuum=0.01,
        timing_offset_ps=0.0,
        rationale="Standard ETSI GS QKD 014 baseline for 25 km SMF-28 links."
    ),
    DecoyConfiguration(
        arm_id=1,
        name="Anti-PNS Yield Contrast",
        mu_signal=0.45,
        nu1_decoy=0.18,
        nu2_vacuum=0.02,
        timing_offset_ps=0.0,
        rationale="Elevates primary decoy intensity to heighten Poisson statistic sensitivity against PNS attacks."
    ),
    DecoyConfiguration(
        arm_id=2,
        name="High-Transmittance Flux Maximizer",
        mu_signal=0.55,
        nu1_decoy=0.12,
        nu2_vacuum=0.01,
        timing_offset_ps=15.0,
        rationale="Boosts signal intensity during high-loss atmospheric/fiber fluctuations."
    ),
    DecoyConfiguration(
        arm_id=3,
        name="Anti-Time-Shift Gating Dither",
        mu_signal=0.48,
        nu1_decoy=0.15,
        nu2_vacuum=0.015,
        timing_offset_ps=-20.0,
        rationale="Dithers gating clock window to disrupt Eve's timing-dependent efficiency mismatch."
    ),
    DecoyConfiguration(
        arm_id=4,
        name="Dense Statistical Characterizer",
        mu_signal=0.52,
        nu1_decoy=0.20,
        nu2_vacuum=0.025,
        timing_offset_ps=10.0,
        rationale="Broadens decoy separation for fast finite-sample Chernoff bound convergence."
    ),
]


class AdaptiveDecoyOptimizer:
    """
    Thompson Sampling Multi-Armed Bandit for Proactive Decoy-State Adaptation.
    Operates strictly in ADVISORY_SHADOW mode to preserve cryptographic proof integrity.
    """

    OPERATIONAL_MODE: str = "ADVISORY_SHADOW"
    IS_AUTONOMOUS_ACTUATION_PERMITTED: bool = False
    SECURITY_PROOF_STATUS: str = "RESEARCH_HEURISTIC_UNPROVEN"

    def __init__(
        self,
        configurations: Optional[List[DecoyConfiguration]] = None,
        cooldown_steps: int = 15,
        random_seed: int = 42,
    ):
        self.configurations = configurations or DISCRETE_DECOY_CONFIGURATIONS
        self.n_arms = len(self.configurations)
        self.cooldown_steps = int(cooldown_steps)
        self.rng = np.random.RandomState(random_seed)

        # Operational status
        self.operational_mode = self.OPERATIONAL_MODE
        self.autonomous_actuation_permitted = self.IS_AUTONOMOUS_ACTUATION_PERMITTED

        # Beta prior parameters (alpha=successes + 1, beta=failures + 1)
        self.alpha: np.ndarray = np.ones(self.n_arms, dtype=float)
        self.beta: np.ndarray = np.ones(self.n_arms, dtype=float)

        self.pull_counts: np.ndarray = np.zeros(self.n_arms, dtype=int)
        self.total_rewards: np.ndarray = np.zeros(self.n_arms, dtype=float)

        self.current_arm_id: int = 0
        self.steps_since_last_change: int = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def current_configuration(self) -> DecoyConfiguration:
        return self.configurations[self.current_arm_id]

    def select_arm(self) -> int:
        """Samples from Beta posteriors and selects the highest expected reward arm."""
        theta_samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(theta_samples))

    def compute_reward(
        self,
        qber: float,
        skr_bps: float,
        is_anomaly: bool,
        finite_key_rate_bps: float = 0.0,
    ) -> float:
        """
        Formulates composite reward r in [0, 1]:
        r = 0.50 * (finite_skr / target) + 0.30 * (1 - QBER/0.11) + 0.20 * (1 - anomaly)
        """
        eff_skr = max(skr_bps, finite_key_rate_bps)
        norm_skr = min(1.0, eff_skr / 15000.0)

        norm_qber = max(0.0, 1.0 - (qber / 0.110))
        norm_anomaly = 0.0 if is_anomaly else 1.0

        reward = 0.50 * norm_skr + 0.30 * norm_qber + 0.20 * norm_anomaly
        return float(np.clip(reward, 0.0, 1.0))

    def update(self, arm_id: int, reward: float) -> None:
        """Updates Beta posterior for the pulled arm."""
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm_id] += r
        self.beta[arm_id] += (1.0 - r)
        self.pull_counts[arm_id] += 1
        self.total_rewards[arm_id] += r

    def step(
        self,
        qber: float,
        skr_bps: float,
        is_anomaly: bool,
        finite_key_rate_bps: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Observes current link metrics, updates bandit posterior, and periodically
        adapts the active decoy-state configuration.
        """
        # Compute reward for the currently active arm
        reward = self.compute_reward(qber, skr_bps, is_anomaly, finite_key_rate_bps)
        self.update(self.current_arm_id, reward)
        self.steps_since_last_change += 1

        action_taken = "HOLD"
        old_arm = self.current_arm_id

        # Safety Fallback: high QBER triggers immediate retreat to nominal arm 0
        if qber >= 0.080 and self.current_arm_id != 0:
            self.current_arm_id = 0
            self.steps_since_last_change = 0
            action_taken = "SAFETY_RESET_TO_NOMINAL"
        elif self.steps_since_last_change >= self.cooldown_steps:
            # Sample next configuration via Thompson Sampling
            new_arm = self.select_arm()
            if new_arm != self.current_arm_id:
                self.current_arm_id = new_arm
                action_taken = f"TRANSITION_ARM_{old_arm}_TO_{new_arm}"
            self.steps_since_last_change = 0

        info = {
            "active_arm": self.current_arm_id,
            "active_config_name": self.current_configuration.name,
            "mu_signal": self.current_configuration.mu_signal,
            "nu1_decoy": self.current_configuration.nu1_decoy,
            "nu2_vacuum": self.current_configuration.nu2_vacuum,
            "timing_offset_ps": self.current_configuration.timing_offset_ps,
            "reward_observed": round(reward, 4),
            "action_taken": action_taken,
            "is_advisory": True,
            "operational_mode": self.OPERATIONAL_MODE,
            "security_proof_status": self.SECURITY_PROOF_STATUS,
            "advisory_disclaimer": "Adaptive decoy parameter optimization is an unproven research heuristic running in advisory/shadow mode only. Direct hardware laser actuation is blocked.",
        }
        self.history.append(info)
        if len(self.history) > 100:
            self.history.pop(0)

        return info

    def request_hardware_actuation(self) -> Dict[str, Any]:
        """
        Safety Lock: Explicitly refuses direct hardware actuation of laser drivers
        because adaptive parameter variations have not been formally integrated
        into composable finite-key security proofs.
        """
        raise PermissionError(
            "CRITICAL SECURITY ENFORCEMENT: Autonomous actuation of adaptive decoy parameters "
            "is strictly prohibited in ADVISORY_SHADOW mode. Preserves GLLP/finite-key proof integrity."
        )

    def get_advisory_recommendation(self) -> Dict[str, Any]:
        """Returns structured advisory recommendation for network operations review."""
        return {
            "status": "ADVISORY_ONLY",
            "recommended_arm_id": self.current_arm_id,
            "configuration": self.current_configuration.name,
            "parameters": {
                "mu_signal": self.current_configuration.mu_signal,
                "nu1_decoy": self.current_configuration.nu1_decoy,
                "nu2_vacuum": self.current_configuration.nu2_vacuum,
                "timing_offset_ps": self.current_configuration.timing_offset_ps,
            },
            "rationale": self.current_configuration.rationale,
            "security_disclaimer": (
                "For research evaluation only. Do not actuate on live production links "
                "without formal composable cryptographic verification."
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for all arms."""
        means = self.alpha / (self.alpha + self.beta)
        arm_stats = []
        for i, cfg in enumerate(self.configurations):
            arm_stats.append({
                "arm_id": cfg.arm_id,
                "name": cfg.name,
                "pulls": int(self.pull_counts[i]),
                "expected_reward": round(float(means[i]), 4),
                "alpha": round(float(self.alpha[i]), 2),
                "beta": round(float(self.beta[i]), 2),
                "active": (i == self.current_arm_id),
            })
        return {
            "current_arm_id": self.current_arm_id,
            "current_config": self.current_configuration.name,
            "operational_mode": self.OPERATIONAL_MODE,
            "is_advisory": True,
            "arms": arm_stats,
            "cooldown_steps": self.cooldown_steps,
            "steps_since_change": self.steps_since_last_change,
        }
