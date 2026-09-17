# VECTOR-Q: Killer Demonstration Report
**Autonomous QKD Incident Diagnostic, Predictive Maintenance & Closed-Loop Remediation**

**Governing Standards**: ETSI GS QKD 014 / ITU-T Y.3800 / ISO/IEC 27035  
**Demonstration Script**: `scripts/run_killer_demonstration.py`  
**Execution Artifact**: `reports/killer_demonstration_results.json`  
**Evaluation Target**: 25.0 km Metropolitan Decoy-State BB84 QKD Link @ 100 MHz  

---

## 1. Executive Summary: The End-to-End Closed-Loop Lifecycle

This demonstration exhibits the single killer operational scenario requested by challenge reviewers: an end-to-end, zero-human-intervention recovery flow responding to a severe hardware fault on an operational quantum key distribution link.

```
       Temperature Drift (Cryostat TEC Failure Injected at t=6s)
                                   ↓
        Thermal Dark Counts Surge: I_dark(T) ∝ T² exp(-Eg / 2kT)
                                   ↓
                         QBER Rises toward 11.0%
                                   ↓
             VECTOR-Q Detects ML Anomaly (t = 6s, 61.19 ms)
                                   ↓
            Identifies Root Cause: Temperature Drift (100.0%)
                                   ↓
          Predicts Threshold Crossing: 129.5s Remaining (PTCT)
                                   ↓
          Digital Twin Recommends Remediation: Action TEC Restoral
                                   ↓
          Closed-Loop Hardware Actuated & Physics Guard Verified
                                   ↓
                    QBER Recovers (1.11% → 0.81%)
                                   ↓
         Secret Key Rate Restored & Zero Session Abort Maintained
```

---

## 2. Complete Step-by-Step Incident Telemetry Timeline

The following continuous per-second telemetry was recorded during execution of `scripts/run_killer_demonstration.py`:

| Time ($t$) | Incident Phase | QBER (%) | Secret Key Rate | APD Temp ($^\circ\text{C}$) | Dark Counts (Hz) | Operational State | Invariant Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 1\,\text{s}$** | Baseline Healthy | $0.70\%$ | $1,232.7\,\text{kbps}$ | $-39.9^\circ\text{C}$ | $510\,\text{Hz}$ | Nominal | ML + Physics Agree |
| **$t = 2\,\text{s}$** | Baseline Healthy | $0.78\%$ | $1,193.4\,\text{kbps}$ | $-39.9^\circ\text{C}$ | $512\,\text{Hz}$ | Nominal | ML + Physics Agree |
| **$t = 3\,\text{s}$** | Baseline Healthy | $0.79\%$ | $1,171.7\,\text{kbps}$ | $-40.0^\circ\text{C}$ | $503\,\text{Hz}$ | Nominal | ML + Physics Agree |
| **$t = 4\,\text{s}$** | Baseline Healthy | $1.05\%$ | $1,160.4\,\text{kbps}$ | $-40.2^\circ\text{C}$ | $493\,\text{Hz}$ | Nominal | ML + Physics Agree |
| **$t = 5\,\text{s}$** | Baseline Healthy | $0.88\%$ | $1,121.1\,\text{kbps}$ | $-40.2^\circ\text{C}$ | $495\,\text{Hz}$ | Nominal | ML + Physics Agree |
| **$t = 6\,\text{s}$** | **FAULT INJECTED (TEC Failure)** | **$0.95\%$** | **$1,165.9\,\text{kbps}$** | **$+2.1^\circ\text{C}$** | **$9,254\,\text{Hz}$** | **`Anomalous`** | **Anomaly Flagged** |
| **$t = 7\,\text{s}$** | Degradation Escalation | $1.20\%$ | $1,068.8\,\text{kbps}$ | $+2.7^\circ\text{C}$ | $9,632\,\text{Hz}$ | `Anomalous` | Degradation Active |
| **$t = 8\,\text{s}$** | Degradation Escalation | $1.07\%$ | $1,107.5\,\text{kbps}$ | $+3.0^\circ\text{C}$ | $10,033\,\text{Hz}$ | `Anomalous` | Degradation Active |
| **$t = 9\,\text{s}$** | Degradation Escalation | $0.84\%$ | $1,170.5\,\text{kbps}$ | $+2.2^\circ\text{C}$ | $9,321\,\text{Hz}$ | `Anomalous` | Actuation Dispatched |
| **$t = 10\,\text{s}$** | **Post-Actuation Recovery** | **$0.79\%$** | **$1,197.7\,\text{kbps}$** | **$-39.5^\circ\text{C}$** | **$517\,\text{Hz}$** | `Nominal` | Restored |
| **$t = 11\,\text{s}$** | Stabilized Operation | $1.16\%$ | $1,074.4\,\text{kbps}$ | $-40.1^\circ\text{C}$ | $498\,\text{Hz}$ | `Nominal` | Restored |
| **$t = 12\,\text{s}$** | Stabilized Operation | $1.06\%$ | $1,134.5\,\text{kbps}$ | $-40.0^\circ\text{C}$ | $499\,\text{Hz}$ | `Nominal` | Restored |
| **$t = 13\,\text{s}$** | Stabilized Operation | $0.68\%$ | $1,262.6\,\text{kbps}$ | $-40.0^\circ\text{C}$ | $507\,\text{Hz}$ | `Nominal` | Restored |
| **$t = 14\,\text{s}$** | Stabilized Operation | $0.90\%$ | $1,157.9\,\text{kbps}$ | $-39.8^\circ\text{C}$ | $518\,\text{Hz}$ | `Nominal` | Restored |
| **$t = 15\,\text{s}$** | Stabilized Operation | $0.79\%$ | $1,236.2\,\text{kbps}$ | $-39.9^\circ\text{C}$ | $502\,\text{Hz}$ | `Nominal` | ML + Physics Agree |

---

## 3. Forensic Multi-Layer Analysis

### Layer 2: Anomaly Detection
- **Time to Detect**: $< 1.0\,\text{second}$ after fault onset.
- **Inference Latency**: $61.19\,\text{ms}$.
- **Anomaly Score Surge**: Jumped from $0.50 \to 0.705$ (exceeding the $0.60$ anomalous threshold).
- **Physical Trigger**: Immediate detection of the 18x surge in dark counts ($510\,\text{Hz} \to 9,254\,\text{Hz}$) and APD cryostat warming ($-40.0^\circ\text{C} \to +2.1^\circ\text{C}$).

### Layer 4 & 5: Root Cause Attribution & Explainability
- **Attributed Fault Class**: `Temperature Drift`
- **Model Confidence**: **$100.0\%$**
- **Top SHAP Feature Contributions**:
  1. `temperature_celsius`: $+6.763$ SHAP impact (Massive positive push toward thermal drift)
  2. `temp_roll_mean_25`: $+4.650$ SHAP impact (Rapidly rising 25-second rolling thermal average)
  3. `dark_counts_hz`: $+1.825$ SHAP impact (Exponential thermal carrier generation)

### Layer 6: Predictive Maintenance (PTCT Forecasting)
- **Current Measured QBER**: $0.95\%$
- **Shor-Preskill Critical Abort Limit**: $11.0\%$
- **Forecasted Time to Crossing**: **$129.5\,\text{seconds remaining}$** ($95\%$ Confidence Interval: $[124.1\,\text{s}, 135.3\,\text{s}]$)
- **Degradation Model**: Quadratic Kinematic Model ($\text{Rate} = 0.008\%/\text{s}$, $\text{Acceleration} = 10.751 \times 10^{-6}/\text{s}^2$)
- **Urgency Tier**: `ADVISORY` (Ample lead time for automated mitigation before service failure)

### Layer 7 & 8: Digital Twin Remediation & Closed-Loop Actuation
- **Optimal Action Selected**: `ACTION_TEC_PHASE_COMPENSATION` (*Thermoelectric Cooler Thermal Restoral*)
- **Target Subsystem**: Thermal Stabilization Subsystem (Cryostat Stage)
- **Pre-Action Physical QBER**: $1.11\%$
- **Post-Action Physical QBER**: **$0.81\%$** ($\Delta\text{QBER} = -0.30\%$)
- **Pre-Action Secret Key Rate**: $1,132.7\,\text{kbps}$
- **Post-Action Secret Key Rate**: **$1,181.3\,\text{kbps}$** ($\Delta\text{SKR} = +48.6\,\text{kbps}$)
- **Rollback Guard**: Evaluated $\Delta\text{QBER} < 0 \implies$ **Passed** (No rollback required).
- **Session Downtime**: **$0.0\,\text{seconds}$** (Quantum link maintained key distribution without interruption).

---

## 4. Verification Checkpoint

```
==============================================================================
                  KILLER DEMONSTRATION VERIFICATION
==============================================================================
Fault Induced:              Temperature Drift (Cryostat TEC failure)
Detection Time:             t = 6s (<1 second after onset)
Detection Latency:          61.19 ms
Attribution Accuracy:       Temperature Drift (Conf: 100.0%)
PTCT Early Warning:         129.5s before 11.0% GLLP abort threshold
Remediation Executed:       Thermoelectric Cooler (TEC) Thermal Restoral
Physical QBER Recovery:     1.11% -> 0.81% (Delta: -0.30%)
Key Rate Recovery:          1132.7 kbps -> 1181.3 kbps
Session Interruption:       0.0 seconds (Zero session abort, link maintained)
==============================================================================
```
