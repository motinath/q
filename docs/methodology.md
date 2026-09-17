# VECTOR-Q: Research & Algorithmic Methodology

**Governing Standards:** ETSI GS QKD 014 / IEEE ML Evaluation Guidelines / ITU-T Y.3800  
**Scope:** Machine learning formulation, evaluation protocols, and physical modeling methodologies  

---

## 1. Machine Learning Problem Formulations

### 1.1 Unsupervised Anomaly Detection Formulation
Given a continuous stream of standardized telemetry vectors $\mathbf{x}_t \in \mathbb{R}^{34}$, the anomaly detection objective is to assign an anomaly score $S(\mathbf{x}_t)$ without requiring labeled anomaly training data:
$$S(\mathbf{x}_t) = -\mathbb{E}_{t \in \mathcal{T}} \left[ h(\mathbf{x}_t) ight]$$
where $h(\mathbf{x})$ represents path length in an ensemble of isolation trees. The binary decision rule is formulated at a fixed false positive rate target $lpha_{	ext{FPR}} = 0.05$ on clean nominal operating data $\mathcal{D}_{	ext{nom}}$:
$$\hat{y}_t = \mathbb{I}\left( S(\mathbf{x}_t) \ge 	au ight), \quad 	ext{where } P_{x \sim \mathcal{D}_{	ext{nom}}}(S(\mathbf{x}) \ge 	au) \le 0.05$$

### 1.2 Root Cause Attribution Formulation
Diagnosing root causes across the 9-class ontology is formulated as multi-class gradient boosting with non-parametric isotonic calibration:
$$P(Y = k \mid \mathbf{x}) = \sigma_{	ext{iso}}\left( f_k(\mathbf{x}) ight)$$
where $f_k(\mathbf{x})$ is the raw margin score from LightGBM for class $k \in \{0, \dots, 8\}$, and $\sigma_{	ext{iso}}$ is fitted on a held-out calibration partition using pool-adjacent-violators (PAV).

For multi-label compound faults (e.g. concurrent thermal drift and optical misalignment), binary relevance estimators are trained independently per fault key $k \in \{	ext{thermal}, 	ext{misalign}, 	ext{loss}, 	ext{apd}\}$:
$$\hat{y}_k = \mathbb{I}\left( P_k(Y_k = 1 \mid \mathbf{x}) \ge 0.50 ight)$$

### 1.3 Quantile Trajectory Forecasting Formulation
For a forecast horizon $H$ steps ahead, the conditional quantile $\hat{q}_lpha(\mathbf{x}_t)$ is optimized by minimizing the asymmetric pinball loss:
$$\mathcal{L}_lpha(y_{t+H}, \hat{q}_lpha) = \max\left( lpha (y_{t+H} - \hat{q}_lpha), (lpha - 1)(y_{t+H} - \hat{q}_lpha) ight)$$
To avoid quantile crossing artifacts ($\hat{q}_{0.10} > \hat{q}_{0.50}$ or $\hat{q}_{0.50} > \hat{q}_{0.90}$), monotonic sorting projections are enforced:
$$\hat{q}_{0.10}^* = \min(\hat{q}_{0.10}, \hat{q}_{0.50}), \quad \hat{q}_{0.90}^* = \max(\hat{q}_{0.90}, \hat{q}_{0.50})$$

---

## 2. Optical Physical Invariant Modeling

### 2.1 Optical Raman Scattering in Coexistence Regimes
When quantum single photons co-propagate with classical DWDM channels in the $1550\,	ext{nm}$ window, spontaneous Raman scattering introduces noise photons into the quantum passband:
$$I_{	ext{Raman}} = P_{	ext{classical}} \cdot L_{	ext{fiber}} \cdot \gamma_{	ext{Raman}} \cdot \Delta \lambda_{	ext{filter}}$$
The resulting QBER degradation is modeled as:
$$	ext{QBER}_{	ext{coex}} = rac{e_{	ext{det}} \cdot \mu \eta + rac{1}{2} (Y_0 + I_{	ext{Raman}})}{\mu \eta + Y_0 + I_{	ext{Raman}}}$$
This formulation allows VECTOR-Q to differentiate benign fiber loss from classical cross-talk.

### 2.2 Diurnal Thermal Expansion & Polarization Drift
Fiber core temperature fluctuations induce optical path length variations and birefringence shifts:
$$\Delta L = L \cdot lpha_{	ext{thermal}} \cdot \Delta T, \quad \Delta 	heta_{	ext{pol}} = eta_{	ext{birefringence}} \cdot \Delta T$$
These cyclic diurnal variations cause slow, bounded QBER drifts without triggering abrupt key rate collapse.

---

## 3. Evaluation & Benchmark Protocols

### 3.1 Strict Partitioning & Seed Governance
To prevent distribution leakage and optimistic bias:
- **Episode-Level Splitting**: Challenge data is partitioned strictly by episode IDs across 60 independent runs (Train: Runs 1–36, Cal: Runs 37–48, Test: Runs 49–60).
- **Temporal Sliding Window Separation**: Chronological sweeps enforce a minimum 5-step forward lookahead gap.
- **Deterministic Random Seeds**: All stochastic processes are initialized with `seed=42`.

### 3.2 Closed-Loop Counterfactual Benchmark Methodology
Closed-loop actuation policies are evaluated within a digital twin simulating Poisson incident arrivals:
1. **Default Policy**: No automated action; sessions remain degraded until catastrophic abort ($E \ge 11.0\%$).
2. **Static Threshold Policy**: Actuates immediately whenever QBER exceeds $5.0\%$, incurring high false actuation penalties on zero-day links.
3. **VECTOR-Q Adaptive Policy**: Gated by physical invariant validation and calibrated RCA confidence, triggering targeted optical compensation only when risk is low.
