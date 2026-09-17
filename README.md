# VECTOR-Q: QKD Performance Diagnosis and Maintenance Framework

> **Connecting Operational & Environmental Dynamics to Quantum Network Degradation, Performance Forecasting, and Validated Corrective Action**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-16%2F16%20passing-brightgreen.svg)]()

---

## 📌 About the Project

**VECTOR-Q** is an operational AI and physics-integrated framework designed for real-time degradation detection, root-cause diagnosis, predictive maintenance forecasting, and closed-loop performance optimisation in Quantum Key Distribution (QKD) systems.

Rather than treating diagnostic monitoring as an isolated scalar alert, VECTOR-Q directly connects environmental fluctuations and equipment behaviour to performance degradation, recommending bounded corrective adjustments and validating the diagnosis through measured post-action recovery.

> 📖 **Comprehensive Technical Reference**: For the complete, in-depth A-to-Z technical architecture, mathematical models, experimental protocols, and benchmark evidence, please see [ARCHITECTURE.md](file:///g:/My%20Drive/q/ARCHITECTURE.md).

---

## 🎯 Problem Statement & Our Solution

### The Core Problem
In deployed commercial and telecommunication QKD networks, **operational fragility** is a major barrier:
1. **Symptom vs. Cause**: Standard monitoring observes only symptoms—a spike in Quantum Bit Error Rate (QBER) or a drop in Secret Key Rate (SKR). However, a rising QBER can stem from fiber bends, diurnal thermal drift, detector APD trap aging, optical misalignment, or timing jitter.
2. **Lack of Predictive Lead Time**: Conventional systems react only after critical thresholds are breached, dropping the key-generation session and disrupting operational security.
3. **Open-Loop Inaction**: Standard monitoring alerts operators but provides no verified, bounded corrective actions to restore link throughput without manual intervention.

### The VECTOR-Q Solution
VECTOR-Q implements an end-to-end **closed-loop diagnosis-and-verification workflow** that answers four foundational operational questions:

1. **Is the QKD system degrading?**  
   *(Module A: Anomaly Detection)* — Unsupervised Isolation Forest on verified healthy baselines, dynamic $[\mu \pm 3\sigma]$ envelopes, and persistence filtering ($W=25$) detect sudden faults and gradual drift while suppressing alert fatigue.
2. **What is the most likely cause?**  
   *(Module B: Multi-Label Root-Cause Diagnosis)* — Multi-label classification identifies co-occurring physical fault modes (thermal, optical alignment, channel loss, detector aging, timing, source fluctuations) with TreeSHAP feature attribution and domain consistency validation.
3. **When will maintenance be needed?**  
   *(Module C: Performance Forecasting & Predictive Maintenance)* — Quantile regression and conformal prediction continuously project future QBER and SKR trajectories, estimating the Projected Time to Critical Threshold (PTCT) to provide actionable early warning for maintenance preparation.
4. **Which corrective action will improve performance?**  
   *(Module D: Performance Optimisation & Closed-Loop Recovery)* — Evaluates candidate bounded control adjustments (piezo polarization, VOA attenuation, detector temperature), executes tuning, and measures post-action recovery with automated rollback if performance degrades.

```
 [Operational & Environmental Telemetry]
                   │
                   ▼
 [Feature Dynamics & Rolling Moments]
         │                   │
         ▼                   ▼
 [Module A: Anomaly]   [Module C: Forecasting]
         │                   │
         ▼                   ▼
 [Module B: Diagnosis] ──────┴─► [Module D: Optimisation]
                                          │
                                          ▼
                               [Closed-Loop Verification]
                                          │
                                          ▼
                               [Operator Feedback Store]
```

---

## 🚀 How to Run VECTOR-Q

### 1. First-Time Setup (Once per PC)

> **Requirement**: Python 3.10 is recommended.

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Running Diagnostics & Tests

```powershell
# Basic diagnostics test (10 streaming cycles)
python tests/run_basic_test.py

# Advanced features test (causal DBN, survival forecasting, graph network embeddings)
python tests/run_advanced_test.py

# Hardware interface verification (mock ID Quantique / SNMP drivers)
python tests/run_hardware_test.py

# Execute full automated test suite (16 core unit & integration tests)
python -m pytest tests/test_vector_q_suite.py -v
```

---

### 3. Launching the Live Operations Dashboard

Launch the interactive Streamlit telemetry and operations intelligence center:

```powershell
streamlit run operations_dashboard/streamlit_app.py
```

The dashboard opens in your browser at `http://localhost:8501`, featuring:
- **Live System Telemetry**: Real-time QBER, SKR, photon counts, and visibility streaming.
- **Fault Injection & Diagnostics**: Simulate physical degradations and inspect multi-label attributions.
- **Explainability**: Interactive TreeSHAP feature importance plots.
- **PTCT Forecasting**: Time-to-critical-threshold survival curves and conformal bounds.
- **Digital Twin**: What-if actuation simulation and parameter sensitivity analysis.
- **Audit Logs**: Cryptographically hash-chained compliance records (`vector_q_audit.db`).

---

## 📁 Project Structure

```
q/
├── README.md                           # Quick start, problem statement, solution overview
├── ARCHITECTURE.md                     # Comprehensive A-to-Z technical specification
├── TROUBLESHOOTING.md                  # Common operational issues and solutions
├── requirements.txt                    # Python package dependencies
├── config/
│   ├── qkd_system_parameters.py        # Optical, physical, and threshold configurations
│   ├── adaptive_baseline_engine.py     # Dynamic [μ ± 3σ] baseline tracking
│   └── config.yaml                     # System-wide configuration
├── physics_engine/
│   ├── optical_channel_models.py       # Analytical physical models (BB84, decoy-state)
│   ├── quantum_telemetry_emulator.py   # Multi-mode physical telemetry generator
│   └── finite_key_analysis.py          # Tomamichel-Lim finite-key security bounds
├── anomaly_detection/
│   ├── isolation_forest_detector.py    # Unsupervised baseline anomaly detector
│   └── sliding_window_features.py      # 33 rolling moments & temporal dynamics
├── root_cause_attribution/
│   ├── lightgbm_classifier.py          # Multi-label fault classification engine
│   └── causal_attribution_engine.py    # Dynamic Bayesian network & causal DAGs
├── physics_validation/
│   └── invariant_rule_evaluator.py     # Physical domain consistency validator
├── predictive_maintenance/
│   ├── survival_ptct_forecaster.py     # Cox PH time-to-threshold forecaster
│   └── conformal_ptct.py               # Conformal prediction with certified bounds
├── remediation_engine/
│   ├── mitigation_optimizer.py         # Multi-objective bounded actuation optimizer
│   └── adaptive_decoy_optimizer.py     # Active decoy parameter exploration
├── streaming_pipeline/
│   ├── qkd_network_orchestrator.py     # End-to-end streaming pipeline orchestrator
│   ├── advanced_orchestrator.py        # Orchestrator with causal & GNN analytics
│   ├── causal_network_intelligence.py  # Causal graph intelligence
│   └── graph_network_embeddings.py     # Multi-node network topology embeddings
├── hardware_interface/
│   ├── base_hardware_interface.py      # Abstract telemetry hardware interface
│   ├── id_quantique_interface.py       # ID Quantique Clavis3/Cerberis driver
│   ├── toshiba_interface.py            # Toshiba QKD interface
│   └── generic_snmp_interface.py       # SNMP monitoring driver
├── model_lifecycle/
│   ├── operator_feedback_capture.py    # Ground-truth feedback store
│   ├── drift_monitor.py                # ADWIN & Page-Hinkley drift monitors
│   ├── automated_retraining_pipeline.py# Retraining pipeline with replay buffer
│   ├── promotion_gate.py               # Shadow validation & regression gate
│   └── model_registry.py               # Versioned model artifact registry
├── operations_dashboard/
│   └── streamlit_app.py                # Multi-tab QNOC operations dashboard
└── tests/
    ├── test_vector_q_suite.py          # Core unit & integration test suite
    ├── test_advanced_features.py       # Advanced analytics test suite
    ├── run_basic_test.py               # 10-cycle quick diagnostic test
    ├── run_advanced_test.py            # 5-cycle advanced diagnostic test
    └── run_hardware_test.py            # Hardware interface verification test
```

---

## ⚖️ System Operating Boundaries

VECTOR-Q is designed as an **operational diagnostic and maintenance layer** that functions strictly within the operating constraints defined by the QKD device's native key-processing subsystem. It does not replace physical protocol checks (such as device-level QBER critical abort cutoffs, basis reconciliation, or privacy amplification), but rather ensures maximum link uptime and optical efficiency during live operations.

For full technical specifications, mathematical derivations, experimental protocols, and benchmark evidence, refer to [ARCHITECTURE.md](file:///g:/My%20Drive/q/ARCHITECTURE.md).

---

**VECTOR-Q** — QKD Performance Diagnosis and Maintenance Framework
