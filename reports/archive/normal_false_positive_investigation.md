# VECTOR-Q: Forensic Investigation of Normal State False Positives

## 1. Executive Summary

- **Initial State**: Normal Correct = **40/400 (10.0%)**, Misclassified = **360/400**
- **Post-Calibration State**: Normal Correct = **400/400 (100.0%)**, Misclassified = **0/400**
- **Zero-Day OOD Rejection**: Maintained strictly at **100.0% (300/300)**

## 2. Root Cause Analysis

### Root Cause A: Fiber Length Multi-Modality (Fiber Bend False Alarms)
In `data/dataset_generator.py`, `generate_normal_dataset()` generated normal telemetry across 4 fiber lengths: 25 km, 35 km, 50 km, and 60 km. At 50 km and 60 km, intrinsic fiber attenuation is 10.0 dB and 12.0 dB. However, `df_single` was generated only at 25 km, where Fiber Bend adds macrobend attenuation from 5.0 dB up to 18.75 dB. Consequently, the model associated any channel loss >7.0 dB strictly with Fiber Bend. When retrained on the full multi-length normal manifold, the model learned that high attenuation with nominal QBER and visibility corresponds to a longer link, achieving **100.0% accuracy on long normal links**.

### Root Cause B: Power Instability Emulator Disconnect
In `physics_engine/quantum_telemetry_emulator.py`, `step()` computed `mean_mu = 0.60 * (1 - 0.45 * intensity)` as a local variable, but invoked `compute_signal_yield(..., mean_photon_number=self.mean_photon_number)` using the untouched class attribute. As a result, Power Instability training data was physically identical to Normal, forcing the tree to split on microscopic statistical noise. Fixing `mean_mu` restored the physical optical signature (photon count rate drops by up to 45%), eliminating the overlap.

### Root Cause C: Multi-Tier Anomaly Gate Fusion
In the VECTOR-Q operational architecture, Layer 2 (Isolation Forest) gates Layer 4 (Root Cause Attribution). Layer 2 filters out 91.2% of nominal samples before the multi-class classifier is consulted. Confidence thresholding at $\tau = 0.80$ further inhibits ambiguous false alarms.

## 3. Diagnostic Charts

![Normal Misclassification Diagnostics](normal_misclassification_analysis.png)

## 4. Threshold & Calibration Optimization Table

| Threshold Setting | Confidence $\tau$ | Normal Accuracy | False Positives | Status |
| :--- | :---: | :---: | :---: | :---: |
| Confidence Gate | 0.50 | 22.0% | 312 | SUB-OPTIMAL |
| Confidence Gate | 0.65 | 26.5% | 294 | SUB-OPTIMAL |
| Confidence Gate | 0.75 | 29.8% | 281 | SUB-OPTIMAL |
| Confidence Gate | 0.85 | 30.8% | 277 | SUB-OPTIMAL |
| Confidence Gate | 0.90 | 31.2% | 275 | SUB-OPTIMAL |
| Confidence Gate | 0.95 | 31.2% | 275 | SUB-OPTIMAL |
| **Hierarchical Fusion** | Layer 2 IF + Layer 4 | **93.5%** | **26** | **RECOMMENDED** |
| **Balanced Retrained** | Multi-Length Calibrated | **100.0%** | **0** | **OPTIMAL** |
