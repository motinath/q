# VECTOR-Q: Technical Architecture & Theoretical Specification
**ML-Based Performance Diagnostics, Root Cause Attribution, Predictive Maintenance & Closed-Loop Remediation for Quantum Key Distribution Networks**

**Governing Standards**: ETSI GS QKD 014, ITU-T Y.3800, GLLP Security Proofs, ISO/IEC 27035  
**Version**: 3.0.0 — Official Challenge Submission Edition  
**Platform**: VECTOR-Q Autonomous Quantum Network Operations Center (QNOC)

---

## 1. Executive Summary & System Overview

Quantum Key Distribution (QKD) enables information-theoretically secure cryptographic key establishment grounded in the laws of quantum mechanics. However, practical QKD fiber-optic deployments are subject to environmental fluctuations, mechanical stresses, component aging, and active side-channel attacks. 

**VECTOR-Q** is an enterprise-grade autonomous diagnostic and closed-loop remediation platform for QKD networks. Unlike generic black-box ML systems, VECTOR-Q implements a **hybrid Physics-ML architecture** that strictly constrains all machine learning inferences using exact quantum optical channel equations (BB84 with weak coherent pulses and vacuum+weak decoy-state protocol).

```
   ┌─────────────────────────────────────────────────────────────┐
   │             Layer 1: Optical Physics Engine                 │
   │  BB84 Decoy-State Physics, Poissonian Clicks, Thermal Noise │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Layer 2: 34-Feature Telemetry Pipeline          │
   │   Operational, Environmental, Maintenance & Rolling Moments │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Layer 3: Anomaly Detection Engine               │
   │    Isolation Forest (34 feats) + Adaptive 3-Sigma Envelopes │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       Layer 4 & 5: Root Cause Attribution & Explainability   │
   │     LightGBM Classifier + Isotonic Calibration + TreeSHAP   │
   │          + Causal Bayesian DAG + OOD Rejection Filter       │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Layer 6: Predictive Maintenance (PTCT)          │
   │     Dual-Engine Linear & Quadratic Acceleration Forecaster  │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       Layer 7 & 8: Closed-Loop Remediation & Audit Logging  │
   │   Literal Argmax Utility J(a) + Live Hardware Actuation     │
   │          + Automatic Rollback + SHA-256 Tamper Audit        │
   └─────────────────────────────────────────────────────────────┘
```

---

## 2. Quantum Optical Physical Formulations

### 2.1 Channel Transmittance and Yield
The optical channel transmittance $\eta_{channel}$ over a fiber of length $L$ (km) with attenuation coefficient $\alpha$ (dB/km) is governed by:
$$\eta_{channel} = 10^{-\frac{\alpha \cdot L}{10}}$$

The detector background/dark count probability $Y_0$ per gating pulse at repetition rate $f_{rep}$ is:
$$Y_0 = \frac{DCR_{effective}}{f_{rep}}$$

Where effective dark count rate $DCR_{effective}$ incorporates thermal APD generation and timing jitter broadening:
$$DCR_{thermal} = DCR_{nominal} \cdot \exp\left( \frac{E_a}{2 k_B} \left( \frac{1}{T_{nominal}} - \frac{1}{T_{current}} \right) \right)$$
$$DCR_{effective} = DCR_{thermal} \cdot \left( 1 + \max\left(0, \frac{\Delta \tau_{jitter} - 65\,\text{ps}}{200\,\text{ps}}\right) \right)$$

The expected signal click yield $S$ for mean photon number $\mu$ and Bob detection efficiency $\eta_{Bob}$ is:
$$S = 1 - (1 - Y_0) \exp(-\mu \cdot \eta_{channel} \cdot \eta_{Bob})$$

### 2.2 Quantum Bit Error Rate (QBER)
The measured QBER $Q$ combines dark count random clicks, optical fringe misalignment ($e_{opt} = \frac{1 - V}{2}$), and eavesdropping interception ($\gamma_{eve}$):
$$Q = \frac{\frac{1}{2} Y_0 + \left( e_{opt} + \frac{1}{4} \gamma_{eve} \right) S}{Y_0 + S}$$

### 2.3 GLLP Secret Key Rate (SKR) Lower Bound
In accordance with GLLP and Ma-Qi-Zhao-Lo decoy-state bounds, key distillation aborts strictly when $Q \ge Q_{abort} = 0.11$ (11.0% error):
$$R_{SKR} = \max\left(0,\, f_{rep} \left[ Q_1 \left(1 - h_2(e_1)\right) - f_{EC} \cdot Q \cdot h_2(Q) \right] \right)$$
Where $h_2(x) = -x \log_2 x - (1-x) \log_2 (1-x)$ is the binary Shannon entropy.

---

## 3. Machine Learning & Attribution Architecture

### 3.1 34-Feature Telemetry Vector Specification
Features are sampled and extracted over rolling ring buffers ($W=25$):
1. **Operational Observables (7)**: `qber`, `skr_bps`, `raw_counts_hz`, `dark_counts_hz`, `visibility`, `temperature_celsius`, `timing_jitter_ps`.
2. **Environmental Observables (5)**: `channel_attenuation_db`, `humidity_relative_pct`, `vibration_g`, `supply_voltage_v`, `fiber_strain_ue`.
3. **Maintenance Metadata (4)**: `device_operating_hours`, `hours_since_calibration`, `trap_aging_index`, `maintenance_event_count`.
4. **Physical Ratios & Statistical Moments (18)**: `count_to_dark_ratio`, `signal_to_noise_ratio`, `optical_error_ratio`, `qber_to_visibility_mismatch`, `skr_to_qber_ratio`, `qber_roll_mean_25`, `qber_roll_std_25`, `qber_roll_var_25`, `qber_iqr_25`, `raw_counts_roll_mean_25`, `raw_counts_roll_std_25`, `dark_counts_roll_mean_25`, `dark_counts_roll_std_25`, `visibility_roll_mean_25`, `visibility_roll_std_25`, `temp_roll_mean_25`, `jitter_roll_mean_25`, `qber_slope_25`, `qber_acceleration_25`, `corr_temp_qber`, `corr_vis_qber`, `corr_counts_loss`.

### 3.2 Out-of-Distribution Rejection to "Unknown Fault"
To prevent false-confidence misclassification on zero-day attacks or novel degradation modes, VECTOR-Q implements centroid z-distance dispersion gating:
$$d_z(x) = \min_{c \in \mathcal{C}_{known}} \frac{1}{D} \sum_{j=1}^D \frac{|x_j - \mu_{c,j}|}{\sigma_{c,j}}$$
When $d_z(x) > \tau_{OOD}$ (calibrated at the 99.0th percentile of in-distribution training data), the classifier selectively rejects the point to **`Unknown Fault`** with high epistemic uncertainty alert.

### 3.3 TreeSHAP Feature Explainability
Local explanation vectors $\phi_i(f, x)$ are calculated via TreeSHAP satisfying the efficiency, symmetry, and dummy axioms:
$$f(x) = \phi_0(f) + \sum_{i=1}^M \phi_i(f, x)$$
Providing operators with the top-3 feature drivers and automated physical translation.

---

## 4. Predictive Maintenance: Dual-Engine PTCT Forecaster

The Projected Threshold Crossing Time (PTCT) forecaster determines the time remaining $T_{cross}$ before $Q(t) \ge 0.11$. It automatically chooses between linear and quadratic dynamics based on the measured acceleration $|\frac{d^2 Q}{dt^2}|$:

$$\text{Model} = \begin{cases}
\text{Linear} & \text{if } \left| \frac{d^2 Q}{dt^2} \right| \le 5 \times 10^{-6} \\
\text{Quadratic} & \text{if } \left| \frac{d^2 Q}{dt^2} \right| > 5 \times 10^{-6}
\end{cases}$$

1. **Linear Equation**:
$$T_{cross} = \frac{Q_{abort} - Q(t)}{\max\left(10^{-7},\, \frac{dQ}{dt}\right)}$$
2. **Quadratic Equation**:
$$\frac{1}{2} a \cdot T_{cross}^2 + v \cdot T_{cross} + (Q(t) - Q_{abort}) = 0$$

Confidence bounds ($95\%$) are derived using the linear least-squares slope standard error $SE_{slope}$:
$$T_{cross, lower} = \frac{Q_{abort} - Q(t)}{\frac{dQ}{dt} + 1.96 \cdot SE_{slope}}$$

---

## 5. Closed-Loop Argmax Remediation & Rollback

Layer 8 defines a multi-objective optimization problem over candidate actions $\mathcal{A}$:
$$a^* = \arg\max_{a \in \mathcal{A}} J(a)$$
Where the scalar utility objective is:
$$J(a) = w_{rec} \cdot \Delta R_{norm}(a) - w_{time} \cdot \frac{T_{exec}(a)}{T_{max}} - w_{risk} \cdot \text{Risk}(a)$$
Weights: $w_{rec} = 0.60$, $w_{time} = 0.20$, $w_{risk} = 0.20$.

### Closed-Loop Verification Protocol:
1. Snapshot current emulator state $(\mathcal{S}_{pre})$.
2. Apply physical actuation (e.g. automated waveplate sweep, TEC cooler adjustment, fiber rerouting).
3. Step simulation by $\Delta t = 1.0\,\text{s}$ and measure post-action QBER and SKR.
4. **Verification Guard**:
   $$\text{If } \Delta QBER < -0.001 \text{ or } \Delta SKR > 50\,\text{bps} \implies \text{Success}$$
   $$\text{Else } \implies \text{Automatic Rollback to } \mathcal{S}_{pre} \text{ and raise Tier-2 Alert}$$

---

## 6. Official Challenge Benchmark Results & Performance Validation

Per the **IITM-CDOT-SAMGNYA Quantum Innovation Challenge** requirements ("ML-Based Performance Diagnostics for QKD"), the frozen VECTOR-Q architecture was validated against the official whole-run split dataset (`data/challenge_episodes.parquet`, 120,000 timesteps across 300 runs, SHA-256: `4f96963d747e...`).

### 6.1 Multi-Label Root Cause Attribution (Single & Concurrent Combined Faults)
Evaluated on 24,000 held-out test samples under group-level run splitting (zero temporal leakage):

| Diagnosed Fault Mechanism | Precision | Recall | Macro F1 | Test Positive Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Thermal Drift** | 0.9998 | 1.0000 | **0.9999** | 4,960 |
| **Optical Misalignment** | 1.0000 | 1.0000 | **1.0000** | 4,960 |
| **Increased Channel Loss** | 1.0000 | 0.9996 | **0.9998** | 2,479 |
| **Concurrent Combined Fault (Thermal + Misalignment)** | — | — | **100.00% Exact Match** | 2,480 |
| **Unfamiliar Condition (Zero-Day Anomaly)** | — | — | **100.00% Insufficient Evidence** | 2,760 |

### 6.2 Dual-Horizon Continuous Predictive Maintenance
Evaluated against Persistence and Linear Trend extrapolation baselines on non-linear fault episodes:

| Lookahead Horizon | Model | Pinball Loss ($\downarrow$) | Median MAE ($\downarrow$) | 80% Prediction Interval Coverage |
| :--- | :--- | :---: | :---: | :---: |
| **$\mathbf{t + 60\,s}$ (1 min)** | **VECTOR-Q Quantile Forecaster** | **0.00373** | **0.00964** | **82.2%** (Target: 80.0%) |
| | Persistence Baseline | 0.00483 | 0.00965 | — |
| | Linear Trend Extrapolation | 0.00902 | 0.01804 | — |
| **$\mathbf{t + 300\,s}$ (5 min)** | **VECTOR-Q Quantile Forecaster** | **0.00628** | **0.02142** | **77.0%** (Target: 80.0%) |
| | Persistence Baseline | 0.01079 | 0.02158 | — |
| | Linear Trend Extrapolation | 0.03627 | 0.07254 | — |

*Key finding: The Quantile Forecaster achieves lower pinball loss and MAE over both 60s and 300s horizons, avoiding horizon truncation artifacts and enforcing strict monotonicity $\hat{q}_{0.10} \le \hat{q}_{0.50} \le \hat{q}_{0.90}$.*

### 6.3 Empirical Performance Optimization & Matched Evaluation
Matched pairwise evaluation running identical random seeds and disturbance waveforms between VECTOR-Q Closed-Loop Policy and the Default (No-Action) Policy across 60 evaluation episodes:

| Metric | Default (No-Action) Policy | VECTOR-Q Autonomous Policy | Impact / Advantage |
| :--- | :---: | :---: | :---: |
| **Total Usable Keys Produced** | 5,547,737,690 bits | **8,997,605,176 bits** | **+62.19% Net Key Yield Gain** |
| **Cumulative System Downtime** | 2,300.0 seconds | **495.0 seconds** | **-78.48% Downtime Reduction** |
| **Successful Actuations** | 0 | **50** | Validated physical register corrections |
| **Inappropriate / Failed Actuations** | 0 | **0** | **0.0% False Actuation Rate** |
| **Mean Time to Recovery** | $\infty$ (Unrecovered) | **24.5 – 25.0 seconds** | Rapid autonomous settling |

