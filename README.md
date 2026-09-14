# Q-SENTINEL: Quantum Key Distribution Network Intelligence Platform

> **AI-Powered Diagnostics, Predictive Maintenance, and Autonomous Resilience for QKD Networks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is Q-SENTINEL?

Q-SENTINEL is an **AI platform for real-time diagnostics and predictive maintenance of Quantum Key Distribution (QKD) networks**. It combines machine learning with quantum physics to detect faults, predict failures, and optimize performance.

**Key Capabilities:**
- Real-time anomaly detection across quantum links
- Root-cause diagnosis for 10 fault types (hardware, environmental, attacks)
- Predictive maintenance with time-to-failure forecasting
- Physics-validated predictions (ensures optical laws are respected)
- Causal analysis ("why did this fail?")
- Counterfactual reasoning ("what if we had done X?")
- Multi-link network intelligence
- Hardware integration for ID Quantique, Toshiba, and generic QKD systems

---

## How to Run Q-SENTINEL (Step by Step)

### Step 1: Install Dependencies

```bash
# Go to project folder (wherever you downloaded/cloned it)
cd path/to/q-sentinel

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate       # Linux/Mac

# Install all packages
pip install -r requirements.txt
```

### Step 2: Train the Models (First Time Only)

```bash
# Train models (takes 5-10 minutes)
python scripts/train_attribution_models.py
```

This creates 3 model files in `models/` folder:
- `isolation_forest.joblib`
- `isolation_scaler.joblib`
- `lightgbm_classifier.joblib`

### Step 3: Run Basic Diagnostics

```bash
python tests/run_basic_test.py
```

This runs 10 diagnostic cycles and shows:
- QBER values
- Anomaly detection
- Root cause identification
- Confidence scores

### Step 4: Run with Advanced Features

```bash
python tests/run_advanced_test.py
```

This runs 5 cycles with advanced AI:
- Causal root-cause analysis
- Time-to-failure prediction
- Risk assessment

### Step 5: Launch Interactive Dashboard

```bash
streamlit run operations_dashboard/streamlit_app.py
```

Then open browser to: `http://localhost:8501`

**Dashboard Features:**
- Real-time telemetry charts
- Fault injection testing
- AI explainability (why predictions were made)
- Time-to-failure forecasting
- What-if scenario simulator

### Step 6: Test Hardware Interface

```bash
python tests/run_hardware_test.py
```

Tests connection to QKD hardware (uses mock hardware for testing)

### Step 7: Run Automated Tests

```bash
# Run all automated tests
pytest tests/ -v
```

---

## Project Structure

```
q-sentinel/
├── config/                          # System configuration
│   ├── qkd_system_parameters.py    # Physics constants, fault classes
│   └── adaptive_baseline_engine.py # Baseline tracking
│
├── physics_engine/                  # Telemetry generation
│   ├── quantum_telemetry_emulator.py      # Simulated QKD telemetry
│   └── hardware_telemetry_source.py       # Real hardware interface
│
├── anomaly_detection/               # Layer 2: Anomaly Detection
│   ├── sliding_window_features.py  # Feature extraction (35 features)
│   └── isolation_forest_detector.py # Unsupervised anomaly detection
│
├── physics_validation/              # Layer 3: Physics Rules
│   └── invariant_rule_evaluator.py # 7 physics invariants
│
├── root_cause_attribution/          # Layer 4: Root-Cause Analysis
│   ├── lightgbm_classifier.py      # Multi-class fault classification
│   ├── shap_feature_explainer.py   # Explainable AI
│   └── causal_attribution_engine.py # NEW: Causal + counterfactual
│
├── predictive_maintenance/          # Layer 6: Forecasting
│   ├── threshold_crossing_forecaster.py  # Basic PTCT
│   ├── survival_ptct_forecaster.py      # NEW: Survival analysis
│   └── conformal_ptct.py                # NEW: Conformal prediction
│
├── remediation_engine/              # Layer 7: Recommendations
│   └── mitigation_optimizer.py     # Action optimization
│
├── digital_twin/                    # Layer 8: Simulation
│   └── digital_twin_lite.py        # Forward trajectory simulation
│
├── incident_intelligence/           # Layer 8: Reporting
│   └── incident_report_generator.py # Automated reports
│
├── streaming_pipeline/              # Layer 9: Orchestration
│   ├── qkd_network_orchestrator.py        # Main pipeline
│   ├── advanced_orchestrator.py           # NEW: With advanced AI
│   ├── causal_network_intelligence.py     # NEW: DBN causal inference
│   └── graph_network_embeddings.py        # NEW: GNN for networks
│
├── hardware_interface/              # NEW: Real Hardware
│   ├── base_hardware_interface.py  # Base interface
│   ├── id_quantique_interface.py   # ID Quantique (Clavis, Cerberis)
│   ├── toshiba_interface.py        # Toshiba QKD
│   └── generic_*_interface.py      # SNMP/REST generic
│
├── model_lifecycle/                 # MLOps
│   ├── model_registry.py           # Version management
│   └── automated_retraining_pipeline.py # Auto-retraining
│
├── audit_logging/                   # Layer 9: Compliance
│   └── compliance_sqlite_database.py # Audit trail
│
├── validation_framework/            # Testing
│   ├── validation_set_*_*.py       # 7 validation sets (A-G)
│   └── data_split_manifest.py      # Train/test splits
│
├── tests/                           # Unit tests
│   ├── test_q_sentinel_suite.py    # Basic tests
│   ├── test_advanced_features.py   # NEW: Advanced AI tests
│   └── fixtures.py                 # Test data
│
├── operations_dashboard/            # Interactive UI
│   └── streamlit_app.py            # Real-time dashboard
│
├── scripts/                         # Training
│   ├── train_attribution_models.py      # Basic training
│   ├── train_advanced_models.py         # NEW: Advanced training
│   └── execute_full_validation.py       # Run all tests
│
└── models/                          # Trained models
    ├── isolation_forest.joblib
    ├── lightgbm_classifier.joblib
    └── advanced/                    # Advanced models
```

---

## Core Concepts

### 1. Fault Classes (10 Types)

Q-SENTINEL diagnoses 10 different fault types:

**Hardware Faults:**
1. **Optical Misalignment** - Laser/detector alignment issues
2. **Detector Degradation** - APD aging, efficiency loss
3. **Timing Jitter** - Clock synchronization problems

**Environmental:**
4. **Channel Attenuation** - Fiber loss, bending
5. **Thermal Drift** - Temperature-induced dark counts

**Quantum Attacks:**
6. **Intercept-Resend** - Eve measures and resends qubits
7. **Detector Blinding** - Bright light attack on APDs
8. **Photon Number Splitting (PNS)** - Multi-photon attack
9. **Time-Shift Attack** - Timing manipulation

**Baseline:**
10. **Normal** - Healthy operation

### 2. Physics Invariants (7 Rules)

AI predictions are validated against physics laws:

1. **Heisenberg Bound**: QBER ≥ 11% → Insecure (Eve present)
2. **SKR Non-Negativity**: Secure key rate cannot be negative
3. **Signal Conservation**: Detected counts ≤ Transmitted counts
4. **Visibility Bound**: 0 ≤ Visibility ≤ 1
5. **Dark Count Thermal**: Dark counts increase with temperature
6. **Loss-Distance**: Attenuation increases with fiber length
7. **Anti-Correlation**: High QBER → Low SKR

### 3. Feature Engineering (35 Features)

From raw telemetry → 35 engineered features:

**Raw Features (9):**
- QBER, SKR, Raw counts, Dark counts
- Singles (Alice/Bob), Visibility
- Temperature, Timing jitter

**Time-Series Features (26):**
- Slopes (25-point window)
- Accelerations (2nd derivative)
- Standard errors
- Running statistics (min, max, std)
- Cross-correlations

---

## Advanced Features (NEW in v2.0)

### 1. Dynamic Bayesian Network (Causal Inference)

**What it does**: Identifies true causal relationships (not just correlations)

```python
from streaming_pipeline.causal_network_intelligence import DynamicBayesianNetworkRCA

dbn = DynamicBayesianNetworkRCA()

result = dbn.infer_root_cause(
    observed_anomalies={"link_0": True, "link_1": True},
    telemetry_features={
        "link_0": {"qber": 0.10, "temperature_celsius": 35},
        "link_1": {"qber": 0.09, "temperature_celsius": 32}
    }
)

print(f"Root Cause: {result.root_cause_node}")
print(f"Causal Chain: {result.causal_chain}")  # Shows cause → effect path
print(f"Causal Strength: {result.causal_strength:.3f}")
```

**When to use**: Multi-link failures, cascading faults, determining if A caused B

---

### 2. Survival Analysis (Time-to-Failure)

**What it does**: Predicts time until system crosses threshold, handles censored data

```python
from predictive_maintenance.survival_ptct_forecaster import SurvivalPTCTForecaster
import pandas as pd

forecaster = SurvivalPTCTForecaster(threshold_qber=0.11)

# Train on historical data
forecaster.fit(training_data, ['qber_slope', 'qber_accel', 'temperature', 'dark_counts'])

# Predict
current = pd.DataFrame({
    'qber_slope': [0.0003],
    'qber_accel': [0.00002],
    'temperature': [30.0],
    'dark_counts': [3500]
})

result = forecaster.predict_survival(current)
print(f"Median Time to Failure: {result.median_ttf:.1f}s")
print(f"95% Confidence: [{result.confidence_lower:.1f}, {result.confidence_upper:.1f}]")
print(f"Hazard Ratio: {result.hazard_ratio:.2f}x baseline risk")
```

**When to use**: Predictive maintenance, SLA management, proactive interventions

---

### 3. Conformal Prediction (Guaranteed Confidence)

**What it does**: Provides prediction intervals with **provable** coverage guarantees

```python
from predictive_maintenance.conformal_ptct import ConformalPTCTForecaster
from sklearn.ensemble import RandomForestRegressor

base_model = RandomForestRegressor(n_estimators=50)
predictor = ConformalPTCTForecaster(base_model, significance=0.10)  # 90% coverage

# Two-stage training (train + calibration)
predictor.fit(X_train, y_train, X_calibrate, y_calibrate)

# Predict with guaranteed coverage
result = predictor.predict(X_test[0])
print(f"Prediction: {result.median_ttf:.1f}s")
print(f"90% Interval: [{result.lower_bound:.1f}, {result.upper_bound:.1f}]")
print(f"Coverage Guarantee: {result.coverage_guarantee:.0%}")  # Will be ≥90%
```

**When to use**: Risk management, when you need calibrated uncertainty, regulatory compliance

---

### 4. Graph Neural Network (Network Intelligence)

**What it does**: Learns patterns across entire network topology

```python
from streaming_pipeline.graph_network_embeddings import GNNNetworkIntelligence
import numpy as np

gnn = GNNNetworkIntelligence(n_features=35)

# Network with 5 links in mesh topology
link_features = {f"link_{i}": np.random.randn(35) for i in range(5)}
adjacency = np.array([
    [0,1,1,0,0],
    [1,0,1,1,0],
    [1,1,0,1,1],
    [0,1,1,0,1],
    [0,0,1,1,0]
])

result = gnn.embed_network(link_features, adjacency)
print(f"Network Health: {result.network_health_score:.3f}")
print(f"Cascade Risks: {result.cascade_risk_per_link}")
```

**When to use**: Multi-link networks, cascading failure prediction, topology-aware diagnostics

---

### 5. Counterfactual Reasoning (What-If Analysis)

**What it does**: Answers "what would have happened if we did X?"

```python
from root_cause_attribution.causal_attribution_engine import CausalAttributionEngine

engine = CausalAttributionEngine()
engine.fit(training_data)  # Learn causal structure

# Counterfactual: "What if we had cooled the detector?"
result = engine.counterfactual_query(
    observed_state={'qber': 'High', 'temperature': 'Hot'},
    intervention={'temperature': 'Cold'}
)

print(f"Observed QBER: {result.observed_qber:.4f}")
print(f"Counterfactual QBER: {result.counterfactual_qber:.4f}")
print(f"Improvement: {result.delta:+.4f}")
print(f"Interpretation: {result.interpretation}")
# Example output: "If we had set temperature=Cold, QBER would improve by 0.0150"
```

**When to use**: Post-incident analysis, maintenance planning, operator training

---

### 6. Hardware Integration

**What it does**: Connect to real QKD hardware

```python
from hardware_interface import IDQuantiqueInterface, ToshibaQKDInterface, MockQKDHardware

# ID Quantique (Clavis/Cerberis)
idq = IDQuantiqueInterface(host="192.168.1.100", link_id="lab_link", community="public")
if idq.connect():
    telemetry = idq.read_telemetry()
    print(f"Real Hardware QBER: {telemetry.qber:.4f}")
    idq.disconnect()

# Toshiba QKD
toshiba = ToshibaQKDInterface(host="192.168.1.101", link_id="field_link")
toshiba.connect()
telemetry = toshiba.read_telemetry()

# Mock hardware for testing
with MockQKDHardware(link_id="test") as hardware:
    telemetry = hardware.read_telemetry()
```

**Supported Systems:**
- ID Quantique (Clavis2, Clavis3, Cerberis3) - SNMP
- Toshiba QKD - REST API
- Generic SNMP devices
- Generic REST API devices

---

## Training Models

```bash
# Train all models at once
python scripts/train_attribution_models.py

# This creates 3 files in models/ folder:
# - isolation_forest.joblib
# - isolation_scaler.joblib  
# - lightgbm_classifier.joblib
```

That's it! Models are trained and ready to use.

---

## Testing

```bash
# Run basic tests
pytest tests/test_q_sentinel_suite.py -v

# Run advanced tests
pytest tests/test_advanced_features.py -v

# Run full validation
python scripts/execute_full_validation.py
```

**What gets tested:**
- Anomaly detection accuracy
- Classification accuracy
- Physics rules validation
- Advanced AI features
- Hardware interfaces
- System latency

---

## Configuration

### System Parameters

Edit `config/qkd_system_parameters.py`:

```python
# QKD System Parameters
fiber_distance_km = 25.0
wavelength_nm = 1550
repetition_rate_hz = 1e9

# Thresholds
qber_warning_threshold = 0.08
qber_critical_threshold = 0.11
min_skr_bps = 500

# Detector Parameters
dark_count_rate_hz = 3000
detector_efficiency = 0.15
timing_jitter_ps = 100
```

### Adaptive Baselines

Baseline tracking automatically adjusts to:
- Temperature variations
- Seasonal fiber loss changes
- Detector aging

Disable with:
```python
orchestrator = QKDNetworkOrchestrator(...)
orchestrator.baseline_engine.enabled = False
```

---

## Dashboard Features

Launch: `streamlit run operations_dashboard/streamlit_app.py`

**Tabs:**
1. **System Status**: Real-time telemetry charts
2. **Fault Injection**: Test different scenarios
3. **Explainability**: SHAP feature importance
4. **PTCT Forecasting**: Time-to-failure predictions
5. **Digital Twin**: What-if simulations
6. **Model Performance**: Accuracy tracking
7. **Audit Log**: Compliance trail

---

## API Reference

### Main Classes

#### QKDNetworkOrchestrator
```python
orchestrator = QKDNetworkOrchestrator(
    config=QKDPhysicsConfig(),
    emulator=QuantumTelemetryEmulator(),
    anomaly_detector=IsolationForestAnomalyDetector(),
    classifier=LightGBMRootCauseClassifier()
)

result = orchestrator.process_step(dt_seconds=1.0)
# Returns: PipelineStepResult with all diagnostics
```

#### AdvancedQKDOrchestrator
```python
orchestrator = AdvancedQKDOrchestrator(
    enable_causal_inference=True,
    enable_survival_analysis=True,
    enable_conformal_prediction=True,
    enable_gnn=False,  # Multi-link only
    enable_counterfactual=True
)

result = orchestrator.process_single_timestep()
# Returns: AdvancedPipelineResult with base + advanced analytics
```

### Key Methods

```python
# Anomaly Detection
detector.fit_healthy_baseline(X_healthy)
result = detector.predict_sample(features)

# Root-Cause Attribution
classifier.fit(X_train, y_train)
result = classifier.predict_sample(features, physics_validation)

# PTCT Forecasting
forecaster.compute_ptct(current_qber, dqber_dt, slope_se, qber_accel)

# Hardware Interface
hardware.connect()
telemetry = hardware.read_telemetry()
result = hardware.send_command(HardwareCommand(...))
```



## Common Issues

**Problem: "No module named 'numpy'"**
```bash
# Activate virtual environment first!
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Problem: "Models not found"**
```bash
# Train models first
python scripts/train_attribution_models.py
```

**Problem: Dashboard won't start**
```bash
# Check if port 8501 is in use
# Kill process or use different port:
streamlit run operations_dashboard/streamlit_app.py --server.port 8502
```

**Problem: Slow performance**
```python
# Disable heavy features you don't need
orchestrator = AdvancedQKDOrchestrator(
    enable_causal_inference=False,  # Disable if not needed
    enable_gnn=False                # Disable if single link
)
```

---

## Dependencies

Install with: `pip install -r requirements.txt`

**Main packages:**
- numpy, pandas, scipy - Data processing
- scikit-learn, lightgbm - Machine learning
- torch, torch-geometric - Graph neural networks
- lifelines - Survival analysis
- pgmpy, causalnex - Causal inference
- streamlit, plotly - Dashboard
- pysnmp, requests - Hardware interfaces
- pytest - Testing

See `requirements.txt` for complete list.

---

**Q-SENTINEL** - AI-Powered Quantum Network Intelligence Platform
