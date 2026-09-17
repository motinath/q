"""
Investigate Why Baselines Beat VECTOR-Q and Determine Optimal Fixes
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_governance import ALL_FEATURE_COLUMNS
from anomaly_detection.isolation_forest_detector import IsolationForestAnomalyDetector
from validation_framework.baseline_models import CUSUMDetector, RandomForestRCABaseline, GradientBoostingRCABaseline, LinearTrendForecaster
from root_cause_attribution.lightgbm_classifier import LightGBMRootCauseClassifier
from predictive_maintenance.threshold_crossing_forecaster import ThresholdCrossingForecaster
from config.qkd_system_parameters import QKDPhysicsConfig

df_norm = pd.read_parquet("data/normal_dataset.parquet")
df_single = pd.read_parquet("data/single_fault_dataset.parquet")
df_mixed = pd.read_parquet("data/mixed_fault_dataset.parquet")
df_unknown = pd.read_parquet("data/unknown_fault_dataset.parquet")

# =========================================================================
# 1. ANOMALY DETECTION: Isolation Forest vs CUSUM
# =========================================================================
print("=== 1. ANOMALY DETECTION: Isolation Forest vs CUSUM ===")
X_train_norm = df_norm[ALL_FEATURE_COLUMNS].values[:1500]
qber_train_norm = df_norm["qber"].values[:1500]

test_norm = df_norm.iloc[1500:2000].copy()
test_single = df_single[df_single["is_anomaly"] == 1].iloc[:500].copy()
test_mixed = df_mixed.iloc[:200].copy()
test_unknown = df_unknown.iloc[:200].copy()

eval_df = pd.concat([test_norm, test_single, test_mixed, test_unknown], ignore_index=True)
X_eval = eval_df[ALL_FEATURE_COLUMNS].values
qber_eval = eval_df["qber"].values
y_true_anom = eval_df["is_anomaly"].values

# CUSUM Baseline
cusum = CUSUMDetector(k_slack_factor=0.5, h_threshold_factor=4.0)
cusum.fit(qber_train_norm)
cusum_preds = cusum.predict(qber_eval)
fp_c = np.sum((y_true_anom == 0) & (cusum_preds == 1))
fn_c = np.sum((y_true_anom == 1) & (cusum_preds == 0))
tp_c = np.sum((y_true_anom == 1) & (cusum_preds == 1))
tn_c = np.sum((y_true_anom == 0) & (cusum_preds == 0))
print(f"CUSUM: Prec={precision_score(y_true_anom, cusum_preds):.4f}, Rec={recall_score(y_true_anom, cusum_preds):.4f}, F1={f1_score(y_true_anom, cusum_preds):.4f}, FPR={fp_c/(fp_c+tn_c):.4f}, FP_count={fp_c}")

# Isolation Forest with various thresholds
iso = IsolationForestAnomalyDetector(n_estimators=100, random_state=42)
iso.fit(X_train_norm)
scores = iso.score_samples(X_eval)

print("\nIsolation Forest across decision thresholds:")
for th in [0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
    preds = (scores >= th).astype(int)
    fp = np.sum((y_true_anom == 0) & (preds == 1))
    fn = np.sum((y_true_anom == 1) & (preds == 0))
    tp = np.sum((y_true_anom == 1) & (preds == 1))
    tn = np.sum((y_true_anom == 0) & (preds == 0))
    fpr = fp / (fp + tn)
    prec = precision_score(y_true_anom, preds)
    rec = recall_score(y_true_anom, preds)
    f1 = f1_score(y_true_anom, preds)
    print(f"  Th: {th:.2f} -> Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}, FP_count: {fp}")

# =========================================================================
# 2. ROOT CAUSE ATTRIBUTION: LightGBM vs RF vs GBDT
# =========================================================================
print("\n=== 2. ROOT CAUSE ATTRIBUTION: LightGBM vs RF vs GBDT ===")
known_df = df_single[df_single["fault_class"] != "Unknown Fault"].copy()
rng_rca = np.random.RandomState(42)
train_idx = rng_rca.rand(len(known_df)) < 0.80
train_df = known_df[train_idx]
test_df = known_df[~train_idx]

X_train_rca = train_df[ALL_FEATURE_COLUMNS].values
y_train_rca = train_df["fault_id"].values
X_test_rca = test_df[ALL_FEATURE_COLUMNS].values
y_test_rca = test_df["fault_id"].values
X_unknown = df_unknown[ALL_FEATURE_COLUMNS].values

# LightGBM default
lgb_default = LightGBMRootCauseClassifier(random_state=42)
lgb_default.fit(X_train_rca, y_train_rca)
p_def = np.argmax(lgb_default.predict_proba(X_test_rca), axis=1)
print(f"LightGBM default: Acc={accuracy_score(y_test_rca, p_def):.4f}, F1={f1_score(y_test_rca, p_def, average='macro'):.4f}")

# LightGBM tuned
lgb_tuned = LightGBMRootCauseClassifier(
    n_estimators=300,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=7,
    random_state=42,
)
lgb_tuned.fit(X_train_rca, y_train_rca)
p_tuned = np.argmax(lgb_tuned.predict_proba(X_test_rca), axis=1)
probs_unk = lgb_tuned.predict_proba(X_unknown)
rej_unk = np.mean(np.max(probs_unk, axis=1) < 0.65)
print(f"LightGBM tuned:   Acc={accuracy_score(y_test_rca, p_tuned):.4f}, F1={f1_score(y_test_rca, p_tuned, average='macro'):.4f}, UnknownRej={rej_unk:.4f}")

# RF & GBDT
rf = RandomForestRCABaseline(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X_train_rca, y_train_rca)
p_rf = np.argmax(rf.predict_proba(X_test_rca), axis=1)
print(f"Random Forest:    Acc={accuracy_score(y_test_rca, p_rf):.4f}, F1={f1_score(y_test_rca, p_rf, average='macro'):.4f}")

gb = GradientBoostingRCABaseline(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train_rca, y_train_rca)
p_gb = np.argmax(gb.predict_proba(X_test_rca), axis=1)
probs_unk_gb = gb.predict_proba(X_unknown)
rej_gb = np.mean(np.max(probs_unk_gb, axis=1) < 0.65)
print(f"Gradient Boost:   Acc={accuracy_score(y_test_rca, p_gb):.4f}, F1={f1_score(y_test_rca, p_gb, average='macro'):.4f}, UnknownRej={rej_gb:.4f}")

# =========================================================================
# 3. PREDICTIVE MAINTENANCE: Dual-Engine PTCT vs Linear Extrapolation
# =========================================================================
print("\n=== 3. PREDICTIVE MAINTENANCE: PTCT vs Linear Extrapolation ===")
q_limit = 0.11
rng_ptct = np.random.RandomState(42)

# Test 1: Accelerating Non-Linear degradation (real physical thermal runaway / trap aging)
trajectories_accel = []
for i in range(50):
    t_cross_true = rng_ptct.uniform(30.0, 90.0)
    q0 = 0.02
    accel = rng_ptct.uniform(0.00004, 0.00010)  # genuine acceleration
    # solve q0 + v0*t_cross + 0.5*accel*t_cross^2 = q_limit
    v0 = (q_limit - q0 - 0.5 * accel * (t_cross_true**2)) / t_cross_true
    hist_ts = np.arange(25)
    hist_q = [q0 + v0 * t + 0.5 * accel * (t**2) + rng_ptct.normal(0, 0.0003) for t in hist_ts]
    # At t=24: current slope is v0 + accel * 24
    curr_slope = v0 + accel * 24.0
    remaining_true = t_cross_true - 24.0
    trajectories_accel.append((np.array(hist_q), remaining_true, curr_slope, accel))

forecaster_vq = ThresholdCrossingForecaster(config=QKDPhysicsConfig(qber_abort_threshold=q_limit))
forecaster_lin = LinearTrendForecaster(qber_limit=q_limit)

vq_errs = []
lin_errs = []
for hist_q, rem_true, curr_slope, accel in trajectories_accel:
    vq_res = forecaster_vq.compute_ptct(current_qber=hist_q[-1], dqber_dt=curr_slope, qber_acceleration=accel)
    lin_pred = forecaster_lin.forecast_ptct(hist_q[-1], curr_slope)
    vq_val = vq_res.t_cross_seconds if vq_res.t_cross_seconds is not None else rem_true * 2.0
    lin_val = lin_pred if lin_pred is not None else rem_true * 2.0
    vq_errs.append(abs(vq_val - rem_true))
    lin_errs.append(abs(lin_val - rem_true))

print(f"Accelerating Degradation (Kinematic):")
print(f"  VECTOR-Q PTCT MAE: {np.mean(vq_errs):.2f} s")
print(f"  Linear Trend MAE:  {np.mean(lin_errs):.2f} s")
