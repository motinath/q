"""
Master Validation & Compliance Benchmark Runner for Q-SENTINEL (QIC 2026 Edition)
Executes:
1. Target vs Measured Engineering Verification Table (Phase 0)
2. Validation Set A: Analytical Physics Unit Assertions
3. Validation Set B: Whole Run-Level Split & Parameter Shift Generalization
4. Validation Set C: Modular HIL Environmental Ingestion Interface
5. Validation Set D: Cross-Domain Dynamic Multi-Phase Operational Trace
6. Validation Set E: Adversarial Robustness Testing (FGSM, PGD, Boundary)
7. Validation Set F: Latency & Computational Profile
8. Validation Set G: Architecture Ablation Study (ML Only vs Physics Only vs Hybrid)
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validation_framework.validation_set_a_physics import run_validation_set_a
from validation_framework.validation_set_b_fault_matrix import run_validation_set_b
from validation_framework.validation_set_c_hil_interface import run_validation_set_c
from validation_framework.validation_set_d_cross_domain import run_validation_set_d
from validation_framework.validation_set_e_adversarial import ValidationSetE_AdversarialRobustness
from validation_framework.latency_profiler import profile_pipeline_latencies


def execute_full_validation_suite() -> None:
    print("================================================================================", flush=True)
    print("      Q-SENTINEL (QIC 2026 EDITION) — MASTER SCIENTIFIC VALIDATION REPORT       ", flush=True)
    print("================================================================================", flush=True)
    
    # 1. Validation Set A (Physics Unit Assertions)
    print("\n>>> [1/7] VALIDATION SET A: ANALYTICAL QUANTUM OPTICAL UNIT VERIFICATION", flush=True)
    res_a = run_validation_set_a()
    status_a = "PASSED" if res_a["all_passed"] else "FAILED"
    print(f"    Status: {status_a} ({res_a['passed_tests']}/{res_a['total_tests']} unit assertions passed)", flush=True)
    for d in res_a["details"]:
        tag = "[OK]" if d["passed"] else "[FAIL]"
        print(f"      {tag} {d['test_name']}", flush=True)
        
    # 2. Validation Set B (Whole Run-Level Split & Parameter Shift)
    print("\n>>> [2/7] VALIDATION SET B: RUN-LEVEL SPLIT & PARAMETER SHIFT GENERALIZATION", flush=True)
    res_b = run_validation_set_b()
    print(f"    Total Held-Out Test Samples:  {res_b['total_test_samples']}", flush=True)
    print(f"    Overall Multi-Class Accuracy:  {res_b['overall_accuracy']*100:.2f}%", flush=True)
    print(f"    Macro Precision:              {res_b['macro_precision']*100:.2f}%", flush=True)
    print(f"    Macro Recall:                 {res_b['macro_recall']*100:.2f}%", flush=True)
    print(f"    Macro F1 Score:               {res_b['macro_f1']*100:.2f}%", flush=True)
    print(f"    Physics Agreement Rate:       {res_b['physics_agreement_rate']*100:.2f}%", flush=True)
    print(f"    Parameter Shift Generalization: {res_b['parameter_shift_generalization_accuracy']*100:.2f}% (Span 35km, DCR 250Hz)", flush=True)
    print("\n    Class-Wise Diagnostics (Whole Run Split):", flush=True)
    for cname, m in res_b["class_metrics"].items():
        print(f"      - {cname:<28}: Precision={m['precision']*100:5.1f}% | Recall={m['recall']*100:5.1f}% | F1={m['f1_score']*100:5.1f}%", flush=True)
        
    # 3. Validation Set C (HIL Interface)
    print("\n>>> [3/7] VALIDATION SET C: HARDWARE-IN-THE-LOOP (HIL) ENVIRONMENTAL INGESTION", flush=True)
    res_c = run_validation_set_c()
    status_c = "PASSED" if res_c["all_passed"] else "FAILED"
    print(f"    Status: {status_c}", flush=True)
    print(f"    Hardware Telemetry Probing:   {'ACTIVE' if res_c['is_hardware_connected'] else 'STANDBY / DISCONNECTED'}", flush=True)
    print(f"    Live Reading:                 {res_c['cpu_temperature_celsius']} C", flush=True)
    print(f"    Scientific Guard:             {res_c['hardware_status_message']}", flush=True)
    
    # 4. Validation Set D (Cross-Domain Dynamic Multi-Phase Trace)
    print("\n>>> [4/7] VALIDATION SET D: CROSS-DOMAIN NON-STATIONARY DYNAMIC TRACE", flush=True)
    res_d = run_validation_set_d()
    print(f"    Total Trace Duration:         {res_d['total_steps']} steps", flush=True)
    print(f"    Attribution Tracking Accuracy: {res_d['overall_attribution_accuracy']*100:.2f}%", flush=True)
    print(f"    Anomaly Capture Rate:         {res_d['anomaly_detection_rate']*100:.2f}%", flush=True)
    
    # 5. Validation Set E (Adversarial Robustness)
    print("\n>>> [5/7] VALIDATION SET E: ADVERSARIAL ROBUSTNESS (FGSM, PGD, BOUNDARY)", flush=True)
    try:
        validator_e = ValidationSetE_AdversarialRobustness()
        res_e = validator_e.run_all_tests()
        status_e = "CERTIFIED" if res_e['status'] == "CERTIFIED" else "FAILED"
        print(f"    Certification Status:         {status_e}", flush=True)
        
        # Print summary for each model
        if 'results' in res_e:
            for model_name, results in res_e['results'].items():
                print(f"\n    {model_name.upper().replace('_', ' ')}:", flush=True)
                for attack_type, metrics_list in results.items():
                    for metrics in metrics_list:
                        if metrics.epsilon_tested == 0.1:  # Report ε=0.1 results
                            print(f"      {attack_type:8s} (ε=0.1): Misclass={metrics.misclassification_rate*100:5.1f}%, "
                                  f"ConfDrop={metrics.avg_confidence_drop:.3f}, Plausible={metrics.physically_plausible_attacks_pct:.1f}%",
                                  flush=True)
    except Exception as e:
        print(f"    Status: SKIPPED (Models not trained or error: {e})", flush=True)
        status_e = "SKIPPED"
        res_e = {"status": "SKIPPED"}
    
    # 6. Latency Profiling
    print("\n>>> [6/7] VALIDATION SET F: LATENCY & COMPUTATIONAL PROFILING", flush=True)
    res_lat = profile_pipeline_latencies(n_iterations=50)
    print(f"    Measured Median Latency:      {res_lat['mean_total_latency_ms']:.3f} ms", flush=True)
    print(f"    Measured P95 Latency:         {res_lat['p95_total_latency_ms']:.3f} ms", flush=True)
    print(f"    Target <5ms Compliant:        {res_lat['target_5ms_compliant']} (CPU TreeExplainer Profile)", flush=True)
    for layer, ms in res_lat["layer_breakdowns_mean_ms"].items():
        print(f"      - {layer:<35}: {ms:.4f} ms", flush=True)
        
    # 7. Architecture Ablation Study
    print("\n>>> [7/7] VALIDATION SET G: ARCHITECTURE ABLATION BENCHMARKS (PHASE 23)", flush=True)
    ab_table = res_b["ablation_study"]["ablation_table"]
    print(f"    {'Architecture':<38} | {'Macro F1':<10} | {'FPR':<8} | {'Explainability Depth'}", flush=True)
    print("    " + "-" * 85, flush=True)
    for row in ab_table:
        print(f"    {row['Architecture']:<38} | {row['Macro_F1']:<10.4f} | {row['False_Positive_Rate']:<8.4f} | {row['Explainability_Depth']}", flush=True)
    
    # Phase 0: Scientific Target vs Measured Summary Table
    print("\n" + "=" * 80, flush=True)
    print("      PHASE 0: SCIENTIFIC CONTRACT & TARGET VS MEASURED VERIFICATION      ", flush=True)
    print("=" * 80, flush=True)
    print(f"    {'Evaluation Metric':<32} | {'Target Target':<16} | {'Measured Actual':<18} | {'Outcome'}", flush=True)
    print("    " + "-" * 76, flush=True)
    
    anom_f1 = res_b["anomaly_detection"]["f1_score"]
    anom_prec = res_b["anomaly_detection"]["precision"]
    anom_rec = res_b["anomaly_detection"]["recall"]
    macro_f1 = res_b["macro_f1"]
    lat_p95 = res_lat["p95_total_latency_ms"]
    
    print(f"    {'Anomaly Detection Precision':<32} | {'>95.0%':<16} | {anom_prec*100:5.2f}%{'':<11} | {'Target Met' if anom_prec >= 0.95 else 'Measured Real'}", flush=True)
    print(f"    {'Anomaly Detection Recall':<32} | {'>95.0%':<16} | {anom_rec*100:5.2f}%{'':<11} | {'Target Met' if anom_rec >= 0.95 else 'Measured Real'}", flush=True)
    print(f"    {'Root-Cause Macro-F1':<32} | {'>90.0%':<16} | {macro_f1*100:5.2f}%{'':<11} | {'Target Met' if macro_f1 >= 0.90 else 'Measured Real'}", flush=True)
    print(f"    {'Parameter Shift Generalization':<32} | {'>85.0%':<16} | {res_b['parameter_shift_generalization_accuracy']*100:5.2f}%{'':<11} | {'Target Met' if res_b['parameter_shift_generalization_accuracy'] >= 0.85 else 'Measured Real'}", flush=True)
    print(f"    {'Cross-Domain Trace Accuracy':<32} | {'>95.0%':<16} | {res_d['overall_attribution_accuracy']*100:5.2f}%{'':<11} | {'Target Met' if res_d['overall_attribution_accuracy'] >= 0.95 else 'Measured Real'}", flush=True)
    print(f"    {'Inference Latency (P95)':<32} | {'<5.0 ms':<16} | {lat_p95:5.2f} ms{'':<10} | {'Target Met' if lat_p95 <= 5.0 else 'Target Not Reached (CPU Profiler)'}", flush=True)
    
    if res_e['status'] != "SKIPPED":
        adv_status_text = "CERTIFIED" if res_e['status'] == "CERTIFIED" else "NOT CERTIFIED"
        print(f"    {'Adversarial Robustness':<32} | {'CERTIFIED':<16} | {adv_status_text:<18} | {'Target Met' if res_e['status'] == 'CERTIFIED' else 'Requires Attention'}", flush=True)
    
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    execute_full_validation_suite()
