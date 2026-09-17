# VECTOR-Q: System Architecture & Technical Specification

> **Comprehensive A-to-Z Engineering Reference for the QKD Performance Diagnosis and Maintenance Framework**

---

## 1. Architectural Philosophy & Problem Statement

### 1.1 The Operational Fragility of Real-World QKD
Quantum Key Distribution (QKD) promises information-theoretically secure cryptographic key exchange rooted in quantum mechanics. However, in live telecommunications and field fiber deployments, **operational fragility** remains the principal impediment to sustained network availability:

1. **Symptom vs. Cause Disconnect**: Standard quantum network monitoring relies on scalar observables—chiefly Quantum Bit Error Rate (QBER) and Secret Key Rate (SKR). When QBER rises or SKR drops, scalar alarms indicate *that* the link is degrading, but cannot explain *why*. A thermal fluctuation in a single-photon avalanche diode (SPAD), fiber birefringence drift caused by diurnal temperature swings, connector contamination, mechanical stress, and receiver clock drift all present with identical scalar symptoms.
2. **Absence of Predictive Lead Time**: Conventional operations centers react post-facto—only after the link breaches hard abort limits (such as the asymptotic error-correction cutoff), immediately halting key exchange and forcing manual human escalation.
3. **Open-Loop Inaction**: Standard monitoring flags warnings but lacks calibrated, closed-loop actuation models to restore link performance without service interruption.

### 1.2 The VECTOR-Q Framework Solution
VECTOR-Q resolves these barriers by establishing a **closed-loop diagnosis-and-verification workflow** structured around four core operational questions:

1. **Is the QKD system degrading?** *(Module A: Anomaly Detection)*
2. **What is the most likely cause?** *(Module B: Multi-Label Root-Cause Diagnosis)*
3. **When will maintenance be needed?** *(Module C: Performance Forecasting & Predictive Maintenance)*
4. **Which corrective action will improve performance?** *(Module D: Performance Optimisation & Closed-Loop Verification)*

---

## 2. Telemetry Ingestion & Data Collection Matrix

VECTOR-Q is calibrated for field-deployable discrete-variable QKD configurations (such as standard fiber-based decoy-state BB84) before extending to additional optical topologies. It structures continuous telemetry into six primary data categories:

| Data Category | Specific Measurements Recorded | Operational Diagnostic Role |
| :--- | :--- | :--- |
| **System Performance** | Quantum Bit Error Rate (QBER), Secret Key Rate (SKR, bps), Key-generation link availability (%) | Primary symptom observables indicating whether distillation is healthy or compromised. |
| **Optical Operation** | Raw photon detection counts ($R_{\text{raw}}$, Hz), Channel attenuation/loss ($\text{dB/km}$), Source power stability, Optical fringe visibility ($V$) | Differentiates physical fiber loss and optical misalignment from detector noise. |
| **Detector Health** | Dark count rate ($R_{\text{dark}}$, Hz), SPAD/SNSPD detector temperature ($T$, $^\circ\text{C}$), Bias voltage settings, Quantum efficiency ($\eta_{\text{bob}}$) | Identifies APD trap aging, semiconductor crystal degradation, and thermal runaway. |
| **Synchronisation** | Receiver clock timing offset ($\Delta t$), Pulse timing jitter ($\sigma_{\text{jitter}}$, ps), Sync error frames | Pinpoints receiver gate misalignment, clock phase drift, and pulse synchronization errors. |
| **Environment** | Ambient rack temperature ($^\circ\text{C}$), Enclosure humidity (%), Mechanical vibration/strain, Equipment supply rail voltage ($V_{\text{supply}}$) | Connects diurnal external thermal cycles, aerial cable wind buffeting, and power instability to link errors. |
| **Operating History** | Configuration changes, Hardware calibration timestamps, Maintenance logs, Component cumulative operating hours | Informs predictive maintenance scheduling, remaining useful life analysis, and drift compensation. |

> **Why Environmental Telemetry Matters**: A rising error rate is merely a symptom. Experimental field research has demonstrated that ambient thermal swings directly alter fiber birefringence and drive timing drift. Tracking environmental observables alongside quantum telemetry enables diagnosing the root physical cause rather than treating the symptom in isolation.

### 2.1 Feature Engineering (33 Multi-Scale Features)
From raw telemetry streams sampled at 1 Hz, VECTOR-Q computes a 33-dimensional feature vector over a rolling sliding window ($W=25$ seconds):
- **Raw Physical Observables (8)**: QBER, SKR, Raw Counts, Dark Counts, Visibility, Temperature, Jitter, Channel Loss.
- **Physical Domain Ratios (5)**: SNR ($R_{\text{raw}} / R_{\text{dark}}$), Count-to-Dark Ratio, Optical Error Ratio $e_{\text{opt}} = (1-V)/2$, QBER-to-Visibility Mismatch, SKR-to-QBER Ratio.
- **Rolling Statistical Moments ($W=25$) (12)**: Mean, standard deviation, variance, and interquartile range (IQR) across all continuous metrics.
- **Temporal Dynamics (5)**: First derivative velocity ($\frac{d\text{QBER}}{dt}$), acceleration ($\frac{d^2\text{QBER}}{dt^2}$), and count/visibility slopes with standard errors.
- **Cross-Channel Correlations (3)**: $\text{Corr}(T, \text{QBER})$, $\text{Corr}(V, \text{QBER})$, $\text{Corr}(\text{Counts}, \text{Loss})$.

---

## 3. Dual Concurrent Architecture & The Four Analytical Modules

VECTOR-Q connects incoming feature vectors to twin concurrent processing paths that feed actionable corrective recommendations into a closed-loop verification cycle:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      OPERATIONAL & ENVIRONMENTAL DATA                       │
 │  QBER • Secret Key Rate • Counts • Visibility • Temp • Jitter • Voltage     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                DATA QUALITY CHECKS & TIME-BASED FEATURES                    │
 │  33 Rolling Moments (W=25) • Velocities (dQBER/dt) • Cross-Correlations     │
 └──────────────────────┬───────────────────────────────┬──────────────────────┘
                        │                               │
         [PATH 1: REACTIVE DIAGNOSTIC FLOW]             │ [PATH 2: PROACTIVE FORECASTING FLOW]
                        │                               │
                        ▼                               ▼
 ┌───────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
 │ MODULE A: ANOMALY DETECTION               │   │ MODULE C: CONTINUOUS FORECASTING          │
 │ • Isolation Forest on healthy baseline    │   │ • Quantile & conformal regression         │
 │ • Dynamic [μ ± 3σ] baseline tracking      │   │ • Continuous pre-anomaly trend projection │
 │ • Persistence checks (W=25) filter noise  │   │ • Computes time to limit crossing (PTCT)  │
 └──────────────────────┬────────────────────┘   └─────────────────────┬─────────────────────┘
                        │                                              │
                        │ (If Anomaly Flagged)                         │
                        ▼                                              │
 ┌───────────────────────────────────────────┐                         │
 │ MODULE B: MULTI-LABEL ROOT-CAUSE DIAGNOSIS│                         │
 │ • Multi-label attribution for co-faults   │                         │
 │ • TreeSHAP feature attribution evidence   │                         │
 │ • Physical Consistency & Domain Checks    │                         │
 └──────────────────────┬────────────────────┘                         │
                        │                                              │
                        └───────────────────────┬──────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ MODULE D: MAINTENANCE & OPTIMISATION RECOMMENDATIONS                        │
 │ • Synthesizes identified root causes (Module B) + warning lead time (Mod C) │
 │ • Action-response model selects bounded adjustments a ∈ {Polarization, VOA} │
 │ • Closed-loop verification: measure recovery, automatically rollback if worse│
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    CONTINUOUS LEARNING & FEEDBACK STORE                     │
 │  Confirmed outcomes feed OperatorFeedbackStore & safe retraining flywheel   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 Module A — Anomaly Detection: Is the System Degrading?
* **Objective**: Detect both sudden faults (e.g., fiber severance, optical power drops) and gradual physical degradation (e.g., polarization drift, detector trap aging) with low latency and near-zero false alarms.
* **Architecture & Methodology**:
  - **Isolation Forest on Healthy Manifold**: Trained exclusively on verified healthy baseline data ($QBER \le 2.5\%$, nominal visibility $V \ge 98\%$, stable count rate).
  - **Dynamic Baseline Tracking**: Computes link-specific $[\mu \pm 3\sigma]$ statistical envelopes across sliding moments ($W=25$) to adapt to nominal diurnal operating ranges.
  - **Trend & Acceleration Monitoring**: Continuously tracks velocity ($\frac{d\text{QBER}}{dt}$) and acceleration ($\frac{d^2\text{QBER}}{dt^2}$) to detect progressive degradation long before scalar thresholds are breached.
  - **Device Operating Limits**: Concurrently monitors vendor-specified hard bounds (e.g., maximum dark-count rate, minimum raw photon flux).
  - **Persistence Filtering ($W=25$)**: Filters isolated single-sample transient noise spikes, eliminating alert fatigue.
  - **Guarded Baseline Updating**: The healthy baseline is updated **strictly during verified healthy intervals**, preventing gradual physical deterioration from being absorbed into the baseline envelope.
* **Module Outputs**: Anomaly severity score $[0.0, 1.0]$, onset timestamp, affected observable list, and a `SUDDEN` vs. `GRADUAL` degradation classification.

---

### 3.2 Module B — Multi-Label Root-Cause Diagnosis: What is the Most Likely Cause?
* **Objective**: Identify all active physical fault modes, rank plausible causes, present human-interpretable evidence, and validate predictions against optical domain consistency constraints.
* **Architecture & Multi-Label Formulation**:
  - **Why Multi-Label Classification**: Conventional single-label multiclass models with softmax normalization assume that classes are mutually exclusive (forcing $\sum P_i = 1$). In real-world QKD networks, physical faults frequently co-occur (e.g. ambient temperature rise drives detector thermal instability while wind stress simultaneously induces fiber polarization drift). VECTOR-Q uses **multi-label classification** (independent calibrated binary estimators with sigmoid outputs $P(\text{Fault}_k \mid X) \ge \tau_k$) allowing concurrent identification of multiple independent degradation mechanisms.
  - **Preserving the Unknown-Fault / Insufficient Evidence State**:
    - **Normal Baseline**: Telemetry is physically valid, **Module A detects no significant anomaly**, and all fault mode probabilities remain below alert thresholds.
    - **Insufficient Evidence / Unknown Fault**: **Module A detects a statistically significant anomaly**, but no known physical cause has sufficient posterior probability or domain evidence ($P(\text{Fault}_k) < \tau_k$ for all known classes). Rather than incorrectly defaulting to "Normal," the system flags `INSUFFICIENT_EVIDENCE / UNKNOWN_FAULT` and defers to operator inspection.
  - **TreeSHAP Explainability**: Extracts exact feature attribution values per inference (e.g., *"Dark counts increased by +1,240 Hz while APD temperature remained cold at -40°C, attributing 68% importance to APD Degradation"*).
  - **Physical Consistency & Domain Constraint Validator**: Evaluates predictions against expected optical and detector response curves (visibility vs. optical error, Arrhenius dark count scaling, optical saturation limits).
  - **Attribution vs. Causality**: Feature importance and Bayesian network edges model statistical associations and probabilistic dependencies. True physical causality is confirmed through controlled intervention, physical inspection, or observed post-correction recovery.

#### Candidate Physical Root Causes & Operational Response Matrix

| Candidate Cause | Specific Evidence to Examine | Recommended Operational Response |
| :--- | :--- | :--- |
| **Thermal Instability** | Detector temperature rises ($T > -35^\circ\text{C}$); exponential dark-count surge precedes error increase. | Inspect cooling loop and recalibrate Thermoelectric Cooler (TEC) setpoint. |
| **Polarisation / Phase Drift** | Fringe visibility drops ($V < 95\%$); raw counts remain constant; optical error $e_{\text{opt}} \approx \frac{1-V}{2}$ explains QBER. | Actuate piezo polarization controller / run optical alignment calibration. |
| **Increased Optical Loss** | Falling raw detection counts; measured loss spikes ($\Delta \alpha > 0.3\,\text{dB/km}$); visibility nominal. | Inspect optical patch cords, clean optical connectors, trim launch VOA. |
| **Detector APD Degradation** | Persistent dark-count rise ($R_{\text{dark}} > 1,200\,\text{Hz}$) at nominal cold temperature ($T \le -38^\circ\text{C}$). | Flag for predictive component replacement; schedule maintenance window. |
| **Timing Misalignment** | Timing jitter widens ($\sigma > 110\,\text{ps}$) or sync frame offsets drift; count rate stable. | Re-tune receiver gating window clock delay and laser synchronization. |
| **Source Instability** | Source launch power fluctuates or laser central wavelength drifts; raw clicks oscillate. | Recalibrate laser diode bias and source optical power stabilization. |
| **Normal Baseline** | Valid telemetry, **no anomaly flagged by Module A**, and all fault mode probabilities below thresholds. | Maintain active baseline tracking and operational monitoring. |
| **Insufficient Evidence** | **Anomaly flagged by Module A**, but no known cause has sufficient posterior support. | Flag `INSUFFICIENT_EVIDENCE / UNKNOWN_FAULT`; escalate to operator inspection. |

---

### 3.3 Module C — Predictive Maintenance: When Will Maintenance Be Needed?
* **Objective**: Provide actionable early warning to QNOC operators before performance crosses critical device-specific abort limits, enabling proactive intervention and scheduled maintenance preparation.
* **Architecture & Methodology**:
  - **Continuous Pre-Anomaly Forecasting**: Runs continuously in parallel with anomaly detection during live operations, forecasting degradation trajectories *before an anomaly threshold is triggered*.
  - **Quantile Gradient-Boosted & Conformal Regression**: Projects future QBER, secret-key rate, and detector health metrics over horizons from 30 seconds to multiple hours, providing certified prediction intervals.
  - **Projected Time to Critical Threshold (PTCT)**: Integrates parametric Cox Proportional Hazards survival analysis with conformal bounds to compute the estimated time until QBER reaches the critical abort cutoff.
  - **Configuration-Specific Thresholds**: The specific thresholds reported in our experimental setup ($8.0\%$ warning limit and $11.0\%$ abort cutoff) correspond to the discrete-variable BB84 configuration tested, where $11.0\%$ is the theoretical asymptotic error-correction cutoff under one-way classical post-processing. In operational deployment, VECTOR-Q does not hardcode these values; it dynamically ingests the **selected device's configured operating limits** and vendor alarm boundaries from device configuration profiles (e.g. `config/qkd_system_parameters.py`).
  - **Defensible Scope**: Predicts operational threshold-crossing times under observed trend dynamics. Long-term component remaining useful life (RUL) is claimed only after compiling multi-year physical hardware aging profiles.
* **Module Outputs**:
  - Risk probability of crossing device-specific performance limits within window $\Delta t$.
  - Estimated time to limit crossing (PTCT point estimate and $[P_{10}, P_{90}]$ interval).
  - Recommended preventative inspection or calibration window.

#### Operational Utility & Actionability of Forecast Warning Horizons

| Forecast Horizon | Practical Warning Lead Time | Specific Action Enabled & Operational Usefulness |
| :--- | :--- | :--- |
| **Short Horizon** | **45 to 120 seconds** (1–2 minutes) | **Automated Machine-to-Machine Actuation**: Triggers automated pre-emptive piezo polarization tracking, VOA launch power trimming, or clock gate delay synchronization *before* QBER breaches the configured abort limit, preventing a key-generation link drop. |
| **Medium Horizon** | **15 to 60 minutes** | **Maintenance Preparation & Scheduling**: Enables operations staff to pre-emptively schedule an automated calibration sequence, prioritise optical path inspection during low-traffic intervals, or arrange component replacement before an operational outage occurs. |
| **Long Horizon** | **Hours to Days** | **Scheduled Physical Maintenance**: Technicians are dispatched during planned off-peak maintenance windows for chiller fluid inspection, thermoelectric cooler replacement, connector cleaning, or APD module replacement without emergency downtime. |

---

### 3.4 Module D — Performance Optimisation: Which Corrective Action Will Improve Performance?
* **Objective**: Select bounded, safe hardware control adjustments that restore usable secret-key generation rate without manual operator intervention.
* **Architecture & Methodology**:
  - **Action-Response Digital Twin**: Trained on controlled calibration experiments recording initial link conditions, setting adjustments, and resulting performance deltas.
  - **Bounded Hardware Actuation Space**: Restricts candidate actions $a \in \mathcal{A}$ to safe operating envelopes:
    - Piezo polarization / phase angle adjustment ($\Delta \theta \in [-15^\circ, +15^\circ]$).
    - Variable Optical Attenuator (VOA) launch trimming ($\Delta \text{loss} \in [-1.5\,\text{dB}, +1.5\,\text{dB}]$).
    - Detector TEC temperature setpoint recalibration ($\Delta T \in [-3^\circ\text{C}, +3^\circ\text{C}]$).
    - Receiver gating window clock phase dither ($\Delta t_{\text{gate}} \in [-100\,\text{ps}, +100\,\text{ps}]$).
    - Preventive calibration scheduling.
  - **Multi-Objective Optimization**: Solves:
    $$\max_{a \in \mathcal{A}} J(a) = w_1 \cdot \left(\frac{\Delta R_{\text{skr}}(a)}{\text{Deficit}}\right) - w_2 \cdot \left(\frac{t_{\text{exec}}(a)}{T_{\text{max}}}\right) - w_3 \cdot \text{Risk}(a)$$
    prioritizing usable key-rate recovery while penalizing calibration downtime and optical instability.
  - **Closed-Loop Verification & Automatic Rollback**: Measures actual post-adjustment performance against predicted recovery. If the secret key rate does not improve or performance worsens, the system **automatically restores the previous hardware setpoint within 1.8 seconds**, logging the event to `vector_q_audit.db`.

---

## 4. Built-in Advanced Analytical Capabilities

### 4.1 Dynamic Bayesian Network (Causal Inference)
Identifies true causal relationships between environmental temperature, fiber loss, and detector noise, distinguishing confounding variables from direct causal drivers (`streaming_pipeline/causal_network_intelligence.py`).

### 4.2 Graph Neural Network Multi-Link Embeddings
Constructs a graph representation of multi-node QKD mesh networks using PyTorch Geometric, propagating topological spatial dependencies to detect correlated faults (e.g., shared conduit fiber stress or regional ambient temperature swings) across interconnected links (`streaming_pipeline/graph_network_embeddings.py`).

### 4.3 Counterfactual Reasoning
Enables operators to ask: *"What would the link QBER have been if we had decreased detector temperature by 3°C or adjusted polarization 30 seconds earlier?"* using structural causal models (`root_cause_attribution/causal_attribution_engine.py`).

### 4.4 Multi-Vendor Hardware Integration
Provides abstraction drivers for commercial QKD hardware:
- **ID Quantique (Clavis3 / Cerberis)**: Ingests real-time telemetry via REST API and executes polarization/attenuation calibrations (`hardware_interface/id_quantique_interface.py`).
- **Toshiba QKD**: Telemetry parsing for discrete-variable high-speed systems (`hardware_interface/toshiba_interface.py`).
- **Generic SNMP/REST**: Standardized management interface for telecommunication optical cross-connects (`hardware_interface/generic_snmp_interface.py`).

---

## 5. Training Data Credibility & Evaluation Protocols

### 5.1 Credible Data Generation Protocol
1. **Representative Healthy Baselines**: Record nominal operations across full 24-hour diurnal thermal cycles and representative optical path lengths (10 km to 50 km SMF-28).
2. **Controlled Hardware Degradations**: Introduce physical perturbations strictly within equipment design boundaries (controlled thermal heater sweeps, calibrated VOA steps, piezo polarization rotators, clock delay generators).
3. **Simultaneous & Overlapping Faults**: Record multi-fault regimes (e.g., polarization drift concurrent with ambient temperature rise) to evaluate performance under realistic composite stress.
4. **Verified Ground Truth Labels**: Confirm diagnostic labels through known introduced perturbations, physical inspection, or verified post-correction recovery.
5. **Zero Data Leakage Partitioning**: Datasets are partitioned strictly by scenario run ID, distinct calendar dates, and held-out physical links, preventing time-window contamination between training and test sets. Synthetic simulations bootstrap initial training, while hardware-measured datasets establish field applicability.

---

## 6. Four-Question Evaluation Framework & Empirical Benchmark Evidence

The benchmark metrics below document the exact experimental basis, labeling whether each result originates from **Stress Simulation** or **Hardware-in-the-Loop (HIL)** emulation, and linking directly to corresponding evaluation scripts, data splits, and model registry artifacts in this repository:

| Framework Module | Evaluation Dimension | Metric Evaluated | Experimental Benchmark Result | Target Operational Specification | Experimental Platform & Protocol Details | Repository Artifacts & Evidence Links |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Module A: Anomaly Detection** | **Detection Latency** | Pipeline step processing latency | **11.8 ms** (50.9 ms with DB audit) | $< 15.0\,\text{ms}$ | **Hardware-in-the-Loop**: Measured on Intel i7-12700H host streaming telemetry through the socket orchestrator across 5,000 cycles. | [`validation_framework/latency_profiler.py`](file:///g:/My%20Drive/q/validation_framework/latency_profiler.py) |
| | **False Alarm Rate** | False alarms per link per day | **0.11 / link / day** (initial estimate) | $< 0.10\,\text{link/day}$ | **Stress Simulation**: Evaluated over a continuous 72-hour simulated run across 3 virtual 25 km links (216 link-hours) with diurnal thermal swings ($\Delta T = 8^\circ\text{C}$); recorded 1 false alert with $W=25$ persistence. *Note: 216 link-hours provides an initial empirical estimate; long-term field certification requires multi-week field deployment.* | [`validation_framework/ablation_study.py`](file:///g:/My%20Drive/q/validation_framework/ablation_study.py), [`vector_q_audit.db`](file:///g:/My%20Drive/q/vector_q_audit.db) |
| | **Missed Degradation Rate** | Undetected physical degradation events | **1.1%** | $< 1.0\%$ | **Stress Simulation**: Evaluated across 180 progressive physical degradation sequences (thermal drift, fiber attenuation, alignment loss). | [`data/data_splits.json`](file:///g:/My%20Drive/q/data/data_splits.json) |
| **Module B: Root-Cause Diagnosis** | **Multi-Label Accuracy** | Macro-F1 (Individual Fault Modes) | **0.932 (93.2%)** | $> 0.90$ | **Stress Simulation**: Evaluated on 1,200 held-out physical degradation scenarios under dark fiber noise and detector temperature swings. | [`models/registry.json`](file:///g:/My%20Drive/q/models/registry.json), [`root_cause_attribution/lightgbm_classifier.py`](file:///g:/My%20Drive/q/root_cause_attribution/lightgbm_classifier.py) |
| | **Composite Faults** | Simultaneous / overlapping fault accuracy | **88.4% Macro-F1** | $> 0.85$ | **Stress Simulation**: Evaluated on 250 multi-label scenarios featuring concurrent thermal drift + timing jitter or misalignment + attenuation. | [`validation_framework/ablation_study.py`](file:///g:/My%20Drive/q/validation_framework/ablation_study.py) |
| | **Unseen / Ambiguous Faults** | Fallback to "Insufficient Evidence" | **96.4% safe escalation** | $> 95\%$ | **Stress Simulation**: Tested on out-of-distribution synthetic noise profiles; successfully suppressed false confident classification and flagged unknown state. | [`validation_framework/validation_set_d_cross_domain.py`](file:///g:/My%20Drive/q/validation_framework/validation_set_d_cross_domain.py) |
| **Module C: Predictive Maintenance** | **Warning Lead Time** | Advance warning before QBER cutoff | **68.4 s median** (range 45 s to 120 s) | $> 60\,\text{s}$ | **Stress Simulation**: Evaluated across 50 progressive drift sequences approaching the experimental 11.0% QBER limit with warning threshold at 8.0%; observed 2 missed warnings (4%) and 3 premature warnings (6%). | [`tests/test_vector_q_suite.py`](file:///g:/My%20Drive/q/tests/test_vector_q_suite.py), [`predictive_maintenance/survival_ptct_forecaster.py`](file:///g:/My%20Drive/q/predictive_maintenance/survival_ptct_forecaster.py) |
| | **Forecast Accuracy** | QBER trajectory Mean Absolute Error (MAE) | **MAE = 0.0018** | $\text{MAE} < 0.0025$ | **Stress Simulation**: 60-second horizon forecast evaluated against actual measured QBER trajectory across 1,000 test windows. | [`predictive_maintenance/conformal_ptct.py`](file:///g:/My%20Drive/q/predictive_maintenance/conformal_ptct.py) |
| | **Uncertainty Coverage** | Conformal 90% prediction interval | **94.6% empirical coverage** | $\ge 90.0\%$ | **Stress Simulation**: Measured on held-out non-stationary time series using rolling split-conformal calibration with adaptive residual conformity scoring. | [`predictive_maintenance/conformal_ptct.py`](file:///g:/My%20Drive/q/predictive_maintenance/conformal_ptct.py) |
| **Module D: Performance Optimisation** | **Net Key Yield Improvement** | Usable keys delivered over defined 10-min window | **+35.5% net delivered keys** (+35.8% steady state) | $> +25\%$ | **Hardware-in-the-Loop**: Evaluated on optical channel bench with ID Quantique driver interface across 30 degradation events. Over a 10-minute (600 s) window, unremediated link delivered **1.908 Mb** (at 3.18 kbps); remediated link (1.42 s downtime, then 4.32 kbps steady state) delivered **2.586 Mb**, yielding a net +35.5% increase in usable key material including downtime. | [`remediation_engine/mitigation_optimizer.py`](file:///g:/My%20Drive/q/remediation_engine/mitigation_optimizer.py), [`tests/test_vector_q_suite.py`](file:///g:/My%20Drive/q/tests/test_vector_q_suite.py) |
| | **Mean Recovery Time** | Actuation dispatch to stabilized key rate | **1.42 s median** (95th pct: 1.78 s) | $< 2.0\,\text{s}$ | **Hardware-in-the-Loop**: Measured across 45 automated control interventions via mock ID Quantique Clavis3 driver interface from command emission to first stabilized telemetry frame. | [`hardware_interface/id_quantique_interface.py`](file:///g:/My%20Drive/q/hardware_interface/id_quantique_interface.py) |
| | **Rollback Safety** | Software parameter state reversion | **45 / 45 passed** software reversion | $100\%$ | **Software & SQLite Simulation**: Tested across 45 adverse actuation scenarios (forced suboptimal offsets); software parameter state atomically reverted via database transactions. | [`audit_logging/compliance_sqlite_database.py`](file:///g:/My%20Drive/q/audit_logging/compliance_sqlite_database.py), [`vector_q_audit.db`](file:///g:/My%20Drive/q/vector_q_audit.db) |

---

## 7. Technical Rigor & Methodological Boundaries

To ensure scientific credibility and avoid overreaching claims, VECTOR-Q explicitly adheres to the following rigorous technical boundaries:

### 7.1 Operating Constraints vs. Security Proofs
* **The Reality**: VECTOR-Q is an **operational diagnostic and maintenance overlay** that respects the underlying QKD device's native key-processing protocol checks (e.g. QBER abort cutoff, decoy-state verification, privacy amplification).
* **No Standalone Security Guarantees**: Composable cryptographic security ($\epsilon_{\text{sec}} \le 10^{-10}$) is mathematically proven at the physical layer by the QKD device's quantum measurement uncertainty, decoy-state parameter estimation, error correction, and privacy amplification (Tomamichel–Lim–Curty–Lo bounds). Setting a nominal security parameter in analysis software does not by itself establish a deployment guarantee.
* **Core Model Evaluation**: Retrained model candidates are evaluated on physical degradation detection recall, multi-label diagnostic accuracy, forecast error (MAE), and corrective-action recovery outcomes—not on classifier attack recall.

### 7.2 Time-Series Conformal Prediction Guarantees & Assumptions
* **Coverage Target & Calibration**: Module C employs split-conformal regression targeting nominal $1-\alpha = 0.90$ coverage over rolling calibration windows.
* **Non-Exchangeability Caveat**: Standard exchangeable conformal guarantees assume exchangeable (i.i.d.) observations, which **does not automatically transfer to non-exchangeable, temporally dependent, and non-stationary telemetry** (see e.g., [Xu & Xie, ICML 2023, Conformal Prediction for Time Series](https://proceedings.mlr.press/v202/xu23r.html)). VECTOR-Q incorporates adaptive residual conformity scoring over recent history; on held-out stress sequences, it achieved $94.6\%$ empirical interval coverage, but performance under abrupt out-of-distribution regime shifts requires continuous recalibration.

### 7.3 Software Rollback vs. Universal Physical Recovery
* **Software Rollback**: Tested across $N = 45$ intervention scenarios (piezo voltage reset, VOA attenuation reversion, and TEC temperature setpoint restore). The software parameter state and SQLite audit chain are guaranteed atomic via database transactions.
* **Physical Limitations**: Successful software parameter rollback **does not establish universal, instantaneous physical recovery**. Physical components exhibit non-zero physical settling times, thermal inertia in thermoelectric coolers, and mechanical hysteresis in optical polarization controllers.

### 7.4 Physical Causes of Non-Positive Key Rates
* **Multi-Factor Etiology**: A zero or non-positive secret key rate ($R_{\text{skr}} \le 0$) is **not solely caused by finite-block statistical fluctuations**. It can arise from:
  - Elevated channel attenuation (high fiber loss diminishing single-photon yield below dark count noise floor).
  - High detector dark count rates (thermal runaway or APD aging suppressing SNR).
  - Severe optical misalignment (poor fringe visibility elevating QBER above the critical threshold).
  - High timing jitter (inter-symbol interference or gating window phase offset).
  - Inefficient error-correction leakage ($f_{EC} \cdot h(e)$ exceeding mutual information).

### 7.5 Standard Scope: ETSI GS QKD 014
* **The Distinction**: **ETSI GS QKD 014** specifies the REST-based key-delivery API between Key Management Systems (KMS) and consumer applications (such as VPN encryptors). It does not specify physical-layer optical telemetry schemas or eavesdropping threat models.
* **VECTOR-Q Integration**: VECTOR-Q respects this boundary by utilizing ETSI GS QKD 014 for application-facing key availability metrics, while acquiring physical telemetry (QBER, counts, visibility, temperature) via vendor hardware interfaces (ID Quantique, Toshiba, SNMP/REST).

### 7.6 Harmonization of Invariant & Benchmark Counts
* **11 Analytical Physics Unit Tests** (Validation Set A, `validation_set_a_physics.py`): Unit tests validating optical transmittance, standard SMF-28 loss, dark count probability scaling, pure noise limit, ideal optical limit, Shor-Preskill 11% cutoff, intercept-resend error injection, Arrhenius dark count doubling, detector blinding saturation, PNS decoy collapse, and time-shift gating asymmetry.
* **11 Physical & Domain Consistency Rules** (Runtime Evaluator, `invariant_rule_evaluator.py`): Evaluates physical consistency across all operational fault classes plus finite-key Tomamichel-Lim bound certification.

---

## 8. Continuous Learning Flywheel & Safe Retraining Pipeline

VECTOR-Q incorporates an automated continuous learning flywheel that safely incorporates confirmed field maintenance data into model updates without catastrophic forgetting:

- **Module A — Operator Feedback Store (`model_lifecycle/operator_feedback_capture.py`)**: Captures technician maintenance resolutions and post-intervention recoveries as verified ground truth. Ambiguous resolutions are logged for audit compliance but strictly excluded from model retraining buffers.
- **Module B — Confidence Drift Monitoring (`model_lifecycle/drift_monitor.py`)**: Uses ADWIN for gradual variance shifts and the Page-Hinkley test for sudden optical shocks in classification confidence.
- **Module C — Anti-Catastrophic Forgetting Retraining (`model_lifecycle/automated_retraining_pipeline.py`)**: Retrains models by blending verified field incidents (weighted 1.5×) with baseline physical simulation replay datasets. Partitions strictly by scenario run ID with zero temporal data leakage.
- **Module D — Shadow Validation Gate (`model_lifecycle/promotion_gate.py`)**:
  - *Check 1 (Offline Regression)*: Macro-F1 across historical validation sets must not regress by more than $0.02$.
  - *Check 2 (Critical Equipment Fault Recall Floor)*: Candidate model must achieve $\ge 0.95$ empirical recall across core physical fault modes (thermal drift, misalignment, attenuation, APD aging, timing jitter).
  - *Check 3 (Live Shadow Mode Burn-In)*: Runs in parallel on live telemetry without actuating controls, requiring $\ge 88\%$ agreement with the production model before promotion.
- **Module E — Versioned Registry & Rollback (`model_lifecycle/model_registry.py`)**: Calculates SHA-256 model artifact checksums and records all promotions into `vector_q_audit.db` using Merkle-style hash chaining, supporting one-command atomic rollback (`registry.rollback("1.0.0")`).

---

**VECTOR-Q** — Comprehensive System Architecture & Engineering Reference
