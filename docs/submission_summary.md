# VECTOR-Q: Master Challenge Submission Package & Summary

**Target Event:** IITM-CDOT-SAMGNYA Quantum Innovation Challenge  
**Category:** ML-Based Performance Diagnostics & Closed-Loop Maintenance for QKD  
**Readiness Score:** 10/10 Verification Checks Passed (100.0%)  
**Status:** Feature Complete & Sealed for Submission  

---

## 1. Challenge Compliance Matrix

| Challenge Requirement | VECTOR-Q Implementation | Verification Evidence & Metric | Status |
| :--- | :--- | :--- | :---: |
| **1. Anomaly Detection** | Unsupervised Isolation Forest on calibrated healthy manifolds ($W=25$, 34 channels) | Recall: **95.1%** (synthetic OpenQKD), **100.0%** (real hardware); FPR: **5.00%** | **PASS** |
| **2. Multi-Label RCA** | Sigmoid-calibrated LightGBM with Mahalanobis-centroid OOD rejection | Macro F1: **0.9717** on unseen test set; Zero-Day OOD Rejection: **100.0%** | **PASS** |
| **3. Predictive Maintenance** | Dual-horizon ($60\text{s}, 300\text{s}$) quantile forecaster (Pinball loss) | MAE: **0.0096** ($60\text{s}$), **0.0214** ($300\text{s}$); **82.2%** empirical coverage | **PASS** |
| **4. System Optimisation** | Counterfactual digital-twin search with closed-loop actuation and 1.8s rollback | Net Secret Key Yield Gain: **+54.57%**; Downtime Saved: **-80.82%** | **PASS** |

---

## 2. Submission Artifacts

- **Authoritative Reports**: `reports/benchmark_results.json`, `reports/challenge_submission_evaluation.json`, `reports/external_validation_metrics.json`, `reports/figures/`.
- **Documentation**: `README.md`, `LICENSE`, `requirements.txt`, `docs/architecture.md`, `docs/methodology.md`, `docs/dataset_provenance.md`, `docs/reproducibility.md`, `docs/submission_summary.md`.
