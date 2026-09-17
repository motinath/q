# VECTOR-Q: External Validation Dataset Assessment Report
**Forensic Inventory, Schema Mapping, and Evaluation Capability Analysis for Real-World QKD Testbed Data**

- **Assessment Target**: `data/external_validation/`
- **Data Origin**: TCD / CONNECT Centre Quantum-Classical Coexistence Testbed (IrelandQCI Project, EU DIGITAL Europe Grant No. 101091520 & Research Ireland Grants 21/US-C2C/3750, 13/RC/2077_P2)
- **Primary Source DOI**: [10.5281/zenodo.21132087](https://doi.org/10.5281/zenodo.21132087) / [Zenodo Record 21132088](https://zenodo.org/records/21132088)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Scope & Constraints**: Forensic audit and assessment only. Zero modifications to existing VECTOR-Q challenge datasets; zero code or model training executed.

---

## 1. Executive Summary

The `data/external_validation/` directory contains an authentic, non-synthetic, multi-platform experimental dataset collected across two major research campaigns on a shared metropolitan-scale optical testbed:
1. **Coexistence Campaign (10 CSV files, 11 JSON stats files, 3,893 records)**: Evaluates a commercial **Toshiba MU** QKD system under varying fiber spool lengths ($0\,\text{km}$ vs $50\,\text{km}$) and active classical channel injection via ROADM ($\text{Noise} \in \{\text{none}, 3, 7, 9, 12\}\,\text{dBm}$) across 0–30 dB attenuation sweeps.
2. **Characterisation Campaign (13 CSV files, 1 JSON stats file, 50,638 records)**: Evaluates four standalone QKD platforms (**Toshiba LE Production**, **Toshiba LE Research**, **ID Quantique Clavis3**, **ID Quantique ClavisXGR**) across fine-grained VOA sweeps (up to 30.6 dB) and 8 dynamic perturbation/transient settling regimes ($1\text{s}, 2\text{s}, 5\text{s}, 20\text{s}, \text{continuous}, \text{continuous\_noisy}, \text{random}$).

**Grand Total**: **23 CSV files**, **12 JSON files**, and **54,531 raw physical experimental records**.

All 23 CSV files are 100% structurally intact with zero missing values, providing a rich benchmark for evaluating VECTOR-Q's core optical error detection, noise tolerance, and degradation forecasting on real QKD hardware.

---

## 2. Complete File & Column Inventory

### 2.1 Coexistence Campaign (Group A: 10 CSV Files, 3,893 Records)
*All Coexistence files share the identical 5-column schema:* `['Noise_dBm', 'Sweep_km', 'VOA_dB', 'QBER', 'SecureKeyRate_bps']`

| File Name | Rows | Columns | QBER Range | SKR Range (bps) | VOA Range (dB) | Experimental Intervention |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `baseline_no_signal_no_added_fibre_full_sweep.csv` | 471 | 5 | 0.0312 – 0.0787 | 0 – 352,387 | 10 – 30 dB | Baseline back-to-back ($0\,\text{km}$), 0 noise |
| `baseline_no_signal_50km_added_fibre_full_sweep.csv` | 372 | 5 | 0.0339 – 0.0840 | 0 – 56,621 | 0 – 12 dB | Baseline $50\,\text{km}$ SMF fiber, 0 noise |
| `roadm_3dbm_injected_signal_no_added_fibre_full_sweep.csv` | 400 | 5 | 0.0324 – 0.0772 | 0 – 325,550 | 10 – 30 dB | $0\,\text{km}$, +3 dBm classical injected power |
| `roadm_3dbm_injected_signal_50km_added_fibre_full_sweep.csv` | 354 | 5 | 0.0368 – 0.0740 | 0 – 55,201 | 0 – 12 dB | $50\,\text{km}$, +3 dBm classical injected power |
| `roadm_7dbm_injected_signal_no_added_fibre_full_sweep.csv` | 354 | 5 | 0.0292 – 0.0796 | 0 – 333,317 | 10 – 30 dB | $0\,\text{km}$, +7 dBm classical injected power |
| `roadm_7dbm_injected_signal_50km_added_fibre_full_sweep.csv` | 348 | 5 | 0.0362 – 0.0723 | 0 – 53,852 | 0 – 12 dB | $50\,\text{km}$, +7 dBm classical injected power |
| `roadm_9dbm_injected_signal_no_added_fibre_full_sweep.csv` | 319 | 5 | 0.0261 – 0.0717 | 0 – 354,598 | 10 – 30 dB | $0\,\text{km}$, +9 dBm classical injected power |
| `roadm_9dbm_injected_signal_50km_added_fibre_full_sweep.csv` | 357 | 5 | 0.0336 – 0.0777 | 0 – 63,311 | 0 – 12 dB | $50\,\text{km}$, +9 dBm classical injected power |
| `roadm_12dbm_injected_signal_no_added_fibre_full_sweep.csv` | 566 | 5 | 0.0262 – 0.0865 | 0 – 314,106 | 10 – 30 dB | $0\,\text{km}$, +12 dBm classical injected power |
| `roadm_12dbm_injected_signal_50km_added_fibre_full_sweep.csv` | 352 | 5 | 0.0405 – 0.8410 | 0 – 50,477 | 0 – 12 dB | $50\,\text{km}$, +12 dBm classical injected power |
| **Coexistence Subtotal** | **3,893** | — | — | — | — | **10 Distinct Optical Regimes** |

### 2.2 Standalone Platform Characterisation (Group B: 5 CSV Files, 48,139 Records)

| File Name | Rows | Columns | QBER Range | SKR Range (bps) | VOA Range (dB) | System Platform Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `characterisation_toshiba_production_full_sweep.csv` | 17,274 | 5 | 0.0166 – 0.0706 | 541 – 2,758,846 | 6.0 – 30.0 dB | `Toshiba_LE_Production` (1 GHz Decoy BB84) |
| `characterisation_toshiba_production_25samples.csv` | 6,024 | 5 | 0.0170 – 0.0706 | 541 – 2,648,667 | 6.0 – 30.0 dB | `Toshiba_LE_Production` (25 samples/point) |
| `characterisation_toshiba_research_3runs.csv` | 663 | 6 | 0.0000 – 0.0785 | 0 – 2,169,195 | 6.0 – 30.0 dB | `Toshiba_LE_Research` (3 independent runs) |
| `characterisation_clavis3_full_sweep.csv` | 769 | 5 | 0.0058 – 0.0521 | 0 – 2,441 | 10.0 – 17.6 dB | `IDQ_Clavis3` (625 MHz COW protocol) |
| `characterisation_clavisxgr_full_sweep.csv` | 23,409 | 5 | 0.0009 – 0.0570 | 143 – 6,755 | 6.0 – 30.6 dB | `IDQ_ClavisXGR` (Next-gen COW platform) |
| **Characterisation Subtotal** | **48,139** | — | — | — | — | **4 Hardware Architectures** |

*Note: Group B schemas use `['System', 'Fibre_km', 'VOA_dB', 'QBER', 'SecureKeyRate_bps']` (plus `Run` in `toshiba_research_3runs`).*

### 2.3 Dynamic Perturbation & Transient Settling (Group C: 8 CSV Files, 2,499 Records)
*All Perturbation files evaluate `Toshiba_LE_Research` and share the schema:* `['System', 'Condition', 'VOA_dB', 'QBER', 'SecureKeyRate_bps']`

| File Name | Rows | Condition Label | QBER Range | SKR Range (bps) | Attenuation | Perturbation / Settling Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `characterisation_perturbation_baseline.csv` | 315 | `baseline` | 0.0155 – 0.0309 | 268,569 – 1,346,019 | 9 – 15 dB | Fixed static attenuation baseline |
| `characterisation_perturbation_1s_settling.csv` | 312 | `1s_settling` | 0.0139 – 0.0356 | 204,510 – 1,212,025 | 9 – 15 dB | Sudden step loss with 1s settling time |
| `characterisation_perturbation_2s_settling.csv` | 312 | `2s_settling` | 0.0147 – 0.0324 | 176,326 – 1,248,864 | 9 – 15 dB | Sudden step loss with 2s settling time |
| `characterisation_perturbation_5s_settling.csv` | 312 | `5s_settling` | 0.0156 – 0.0334 | 233,695 – 1,255,570 | 9 – 15 dB | Sudden step loss with 5s settling time |
| `characterisation_perturbation_20s_settling.csv` | 312 | `20s_settling` | 0.0151 – 0.0371 | 152,617 – 1,297,417 | 9 – 15 dB | Sudden step loss with 20s settling time |
| `characterisation_perturbation_continuous.csv` | 312 | `continuous` | 0.0154 – 0.0319 | 187,779 – 1,094,124 | 9 – 15 dB | Continuous dynamic VOA ramp (0s settling) |
| `characterisation_perturbation_continuous_noisy.csv` | 312 | `continuous_noisy` | 0.0148 – 0.0379 | 178,285 – 1,210,502 | 9 – 15 dB | Continuous dynamic ramp with amplitude jitter |
| `characterisation_perturbation_random.csv` | 312 | `random` | 0.0161 – 0.0316 | 193,781 – 1,094,582 | 9 – 15 dB | Randomized step timing and amplitude |
| **Perturbation Subtotal** | **2,499** | — | — | — | — | **8 Controlled Transient Dynamics** |

### 2.4 JSON Statistical Metadata Files (12 Files)
- **10 Individual Stats Files (`<basename>_stats.json`)**: Pre-computed statistical profiles for each Coexistence CSV file containing:
  - `row_count`: Total valid measurement epochs.
  - `voa_blocks`: Breakdown per VOA attenuation level including sample counts, mean, std, median, min, and max for both QBER and SKR.
  - `overall`: Complete run distribution and count of collapsed key generation sessions ($\text{SKR} = 0$).
- **`csv_summary_stats.json`**: Aggregated dictionary cross-referencing all 10 coexistence experiments.
- **`characterisation_summary_stats.json`**: Aggregated statistics across all 13 standalone characterisation and perturbation runs.

---

## 3. Telemetry Characteristics & Measurement Dynamics

### 3.1 Sampling Rate & Temporal Cadence
- **Measurement Unit**: Discrete **QKD Key Distillation Epochs** (SNMP blocks).
- **Temporal Resolution**: In commercial QKD systems (Toshiba and IDQ), each row corresponds to one accumulated privacy amplification block. Under nominal attenuation (6–15 dB), key blocks are produced every **1 to 5 seconds**. At high attenuation (>25 dB), block accumulation times stretch to **10 to 30 seconds**.
- **Settling Time Experiments**: The perturbation series explicitly parameterizes temporal tracking response by constraining post-actuation stabilization time:
  $$\Delta t_{\text{settle}} \in \{0\,\text{s (continuous)}, 1\,\text{s}, 2\,\text{s}, 5\,\text{s}, 20\,\text{s}\}$$
  This directly mirrors VECTOR-Q's internal physical closed-loop actuation settling time parameter ($\tau_{\text{settle}} \approx 1.8 - 3.0\,\text{s}$).

### 3.2 Available Optical Metrics
1. **QBER (Quantum Bit Error Rate)**:
   - Measured as a floating-point fraction ($0.0 \le \text{QBER} \le 1.0$).
   - Ranges from ultra-clean operational states ($\approx 0.0009$ on ClavisXGR; $\approx 0.016$ on Toshiba LE) up to total optical breakdown ($\text{QBER} = 0.8410$ under +12 dBm noise on 50 km fiber).
2. **SecureKeyRate_bps (Secret Key Rate)**:
   - Integer bits per second delivered to the Key Management System (KMS).
   - Ranges from 0 bps (session abort / threshold breach) up to **2.76 Mbps** on the GHz Toshiba LE production platform.
3. **VOA_dB (Optical Channel Attenuation)**:
   - Controllable calibrated attenuation injected via dual ADVA OPM40 variable optical attenuators.
   - Sweep ranges from 0.0 dB up to 30.6 dB (representing link lengths up to 150 km of SMF-28 equivalent loss).
4. **Sweep_km / Fibre_km (Physical Fiber Length)**:
   - $0\,\text{km}$ (back-to-back laboratory baseline) vs $50\,\text{km}$ (installed dark fiber reel, providing $\approx 10\,\text{dB}$ of intrinsic Rayleigh attenuation and chromatic dispersion).
5. **Noise_dBm (Injected Classical Power)**:
   - Auxiliary optical power injected through a Reconfigurable Optical Add-Drop Multiplexer (ROADM) across values: $\text{None}, +3\,\text{dBm}, +7\,\text{dBm}, +9\,\text{dBm}, +12\,\text{dBm}$, producing Spontaneous Raman Scattering (SpRS) photons directly within the single-photon quantum window.

### 3.3 Available Fault & Perturbation Labels
The external dataset contains explicit, controlled physical condition annotations:
- **Baseline Healthy Condition**: Clean back-to-back operation without injected optical noise.
- **WDM Raman In-Band Crosstalk / Classical Injection**: Labeled by injection power level ($+3, +7, +9, +12\,\text{dBm}$).
- **Excess Transmission Path Loss**: Labeled by $50\,\text{km}$ spool insertion.
- **Dynamic Optical Transients / Macrobends**: Labeled by perturbation settling modes (`1s_settling`, `2s_settling`, `5s_settling`, `20s_settling`, `continuous`, `continuous_noisy`, `random`).
- **Complete Channel Collapse**: Labeled by zero-SKR states ($\text{SKR} = 0, \text{QBER} \ge 11.0\%$).

---

## 4. Feature Mapping: External Data vs. VECTOR-Q 34-Feature Vector

VECTOR-Q operates on an official **34-feature vector** (`config/dataset_governance.py`). Below is the forensic alignment mapping:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│       EXTERNAL TESTBED MEASUREMENT           │          VECTOR-Q 34-FEATURE EQUIVALENT      │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ QBER                                         │ qber                                         │
│ SecureKeyRate_bps                            │ skr_bps                                      │
│ VOA_dB + (Sweep_km * 0.20 dB/km)             │ channel_attenuation_db                       │
│ SecureKeyRate_bps / max(1e-5, QBER)          │ skr_to_qber_ratio                            │
│ Rolling Mean of QBER (W=25)                  │ qber_roll_mean_25                            │
│ Rolling Std of QBER (W=25)                   │ qber_roll_std_25                             │
│ Rolling Variance of QBER (W=25)              │ qber_roll_var_25                             │
│ Interquartile Range of QBER (W=25)           │ qber_iqr_25                                  │
│ First Difference of QBER (ΔQ/Δt)             │ qber_slope_25                                │
│ Second Difference of QBER (Δ²Q/Δt²)          │ qber_acceleration_25                         │
│ Rolling Correlation (Loss vs QBER)           │ corr_counts_loss (Proxy)                     │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

### Detailed Feature Coverage Breakdown:
| VECTOR-Q Governance Group | Total Features | Directly Mapped from External Data | Deterministically Computable via Moments | Missing / Not Logged in External CSVs |
| :--- | :---: | :---: | :---: | :---: |
| **Group 1: Operational Observables** | 7 | 3 (`qber`, `skr_bps`, `channel_loss`) | 0 | 4 (`raw_counts`, `dark_counts`, `visibility`, `jitter`) |
| **Group 2: Environmental Observables** | 5 | 0 | 0 | 5 (`temp`, `humidity`, `vibration`, `voltage`, `strain`) |
| **Group 3: Maintenance Metadata** | 4 | 0 | 0 | 4 (`operating_hours`, `calibration_time`, `aging_idx`, `events`) |
| **Group 4: Physical Ratios & Moments** | 18 | 1 (`skr_to_qber_ratio`) | 6 (`roll_mean`, `roll_std`, `var`, `iqr`, `slope`, `accel`) | 11 (Cross-environmental correlations, internal click ratios) |
| **TOTALS** | **34** | **4 Direct** | **6 Synthesized Moments** | **24 Omitted Optical / Environmental Internals** |

> 📌 **Key Technical Takeaway**: The external dataset provides ground-truth operational outputs (`qber`, `skr_bps`, `loss`), allowing testing of the operational behavior of VECTOR-Q's anomaly detection and degradation forecasting. However, because subsystem-internal hardware registers (cryostat temperature, APD dark counts, piezo voltage) were omitted from the public SNMP dumps, full 34-feature multi-label root-cause classifiers cannot receive complete physical inputs without feature imputation or reduced-dimension projection.

---

## 5. Potential Independent Evaluation Experiments

### Experiment Suite 1: Unsupervised Anomaly Detection (Pillar 1 Validation)
* **Experiment 1A: Out-of-Distribution Coexistence Noise Detection**
  * *Method*: Fit Isolation Forest on baseline clean operation (`baseline_no_signal_no_added_fibre_full_sweep.csv`) under nominal attenuation ($10-15\,\text{dB}$).
  * *Evaluation*: Stream coexistence files (`roadm_3dbm` to `roadm_12dbm`). Determine if Raman noise surges trigger anomaly alerts, and verify false-positive suppression on clean 50 km baseline runs.
* **Experiment 1B: Dynamic Perturbation Transient Alerting**
  * *Method*: Ingest the 8 perturbation datasets (`characterisation_perturbation_*`).
  * *Evaluation*: Verify whether sudden step loss perturbations trigger anomaly flags during settling intervals ($1\text{s}, 2\text{s}$), and verify that alarms automatically clear once the system settles ($20\text{s}$).

### Experiment Suite 2: Cross-Platform Protocol Invariant Testing (Pillar 2 Generalization)
* **Experiment 2A: BB84 vs. Coherent One-Way (COW) Transferability**
  * *Method*: Evaluate VECTOR-Q detection manifolds calibrated on Toshiba Decoy-State BB84 when applied to ID Quantique Clavis3 and ClavisXGR COW platforms.
  * *Evaluation*: Determine whether the error-rate and loss frontiers produce consistent invariant warnings across distinct quantum protocols without platform-specific recalibration.
* **Experiment 2B: Optical Attenuation vs. In-Band Optical Noise Discrimination**
  * *Method*: Classify whether degradation is caused by pure optical path attenuation (macrobend / distance) versus in-band classical noise injection (Raman crosstalk).
  * *Physical Signature*: Pure loss reduces SKR with gradual QBER growth; Raman noise injection causes sharp QBER spikes ($>7\%$) even at low-to-moderate attenuation ($10-15\,\text{dB}$).

### Experiment Suite 3: Predictive Maintenance & Threshold Forecaster (Pillar 3 Validation)
* **Experiment 3A: Channel Attenuation Ramp & Key Collapse Early Warning**
  * *Method*: Evaluate the Dual-Horizon Quantile Forecaster on progressive attenuation sweeps ($10\,\text{dB} \to 30\,\text{dB}$).
  * *Evaluation*: Measure the Projected Time to Critical Threshold (PTCT) lead time before QBER crosses $11.0\%$ or SKR drops to 0 bps, comparing against Linear and Persistence baselines.
* **Experiment 3B: Dynamic Recovery Time Estimation**
  * *Method*: Measure how accurately the forecaster tracks recovery dynamics across the $1\text{s}$, $2\text{s}$, $5\text{s}$, and $20\text{s}$ settling runs.

---

## 6. Dataset Usability & Limitations

### 6.1 Usability Verdict
- **Overall Usability**: **HIGH** for Operational Anomaly Detection (Pillar 1) and Performance Forecasting (Pillar 3).
- **Usability for Subsystem Root Cause Attribution (Pillar 2)**: **PARTIAL / CONSTRAINED**. Can classify macroscopic channel perturbations (high loss vs. in-band noise injection), but cannot diagnose internal hardware root causes (e.g. APD cooling failure, clock jitter wander) due to unrecorded sub-assembly telemetry.
- **Usability for Closed-Loop Optimization (Pillar 4)**: **HIGH (Benchmark Surrogate)**. The perturbation settling datasets provide empirical response curves for testing post-action recovery verification.

### 6.2 Identified Limitations
1. **Absence of Environmental Observables**: No ambient temperature, humidity, vibration, or fiber strain measurements were recorded during these testbed runs.
2. **Absence of Internal Detector Channels**: No separate dark count rate ($Hz$), raw click counts ($Hz$), or optical visibility ($V$) channels; only end-to-end QBER and usable key rate are reported.
3. **Block-Averaged Sampling**: Records reflect key distillation block completion rather than continuous 1.0 Hz wall-clock intervals. Block duration varies from $\approx 1\,\text{s}$ at low attenuation to $>20\,\text{s}$ near the cutoff limit.
4. **No Direct Hardware Control Registers**: The files represent passive monitoring sweeps; external hardware registers (`tec_drive_current`, `epc_bias_voltage`) are not included.

---

## 7. Recommended Next Steps for External Evaluation Pipeline

To build an independent validation pipeline without altering any existing repository models or datasets:

1. **Implement Dedicated External Data Adapter** (`validation_framework/external_qkd_adapter.py`):
   - Standardize column names (`QBER` $\to$ `qber`, `SecureKeyRate_bps` $\to$ `skr_bps`, `VOA_dB` $\to$ `channel_attenuation_db`).
   - Compute rolling window moments ($W=25$) on sequential measurement blocks.
2. **Execute Pillar 1 Baseline Detection Audit**:
   - Benchmark Isolation Forest, EWMA, and CUSUM against the 10 Coexistence noise injection regimes.
3. **Benchmark Pillar 3 Quantile Forecaster**:
   - Evaluate early warning lead time on the fine-grained attenuation sweeps (`characterisation_clavisxgr` and `characterisation_toshiba_production`).
4. **Generate External Evaluation Dossier** (`reports/external_validation_results.json`):
   - Export independent performance metrics to demonstrate VECTOR-Q's generalization on external hardware data.
