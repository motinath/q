# VECTOR-Q Empirical Optimization & Closed-Loop Benchmark Report

## Executive Summary
- **Net Usable Key Yield Gain**: **+56.39%** over Default (No-Action) Policy
- **Cumulative Downtime Reduction**: **-85.53%** (123.0s vs 850.0s)
- **Successful Autonomous Actuations**: **25**
- **Inappropriate / Failed Actuations**: **0** (Rate: 0.0%)

## Per-Condition Matched Comparison Table

| Condition | Default Keys (bits) | VECTOR-Q Keys (bits) | Key Gain (%) | Default Downtime | VECTOR-Q Downtime | Mean Recovery Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Operation** | 145,851,291 | 145,851,291 | **+0.0%** | 0.0s | 0.0s | N/As |
| **Thermal Drift** | 143,700,248 | 145,826,906 | **+1.48%** | 0.0s | 0.0s | 0.0s |
| **Optical Misalignment** | 42,732,541 | 144,471,723 | **+238.08%** | 85.0s | 1.0s | 1.0s |
| **Channel Loss Event** | 53,502,697 | 144,292,935 | **+169.69%** | 0.0s | 0.0s | 0.0s |
| **Combined Fault (Thermal + Misalignment)** | 42,275,308 | 116,907,882 | **+176.54%** | 85.0s | 23.6s | 23.6s |
| **Unfamiliar Anomaly (OOD Perturbation)** | 49,443,019 | 49,443,019 | **+0.0%** | 0.0s | 0.0s | N/As |

## Methodology & Verification Guardrails
- **Matched Protocol**: Identical random seeds, device physics parameters, and disturbance waveforms are fed into both policies in lockstep.
- **Zero Cherry-Picking**: Every condition run for identical time horizon; all secret keys generated are strictly integrated without clipping.
- **Safety Gate**: VECTOR-Q does not execute any corrective action when the operational state is `Normal` or `Insufficient Evidence`, guaranteeing zero inappropriate actions during uncharacterized anomalies or healthy links.