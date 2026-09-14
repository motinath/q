"""
Advanced Q-SENTINEL Test - Run with all AI features
"""

from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator

# Initialize with advanced AI
orchestrator = AdvancedQKDOrchestrator(
    enable_causal_inference=True,
    enable_survival_analysis=True,
    enable_counterfactual=True
)

print("Running Q-SENTINEL Advanced Diagnostics...\n")

# Run 5 diagnostic cycles with advanced analytics
for i in range(5):
    result = orchestrator.process_single_timestep()
    
    print(f"=== Cycle {i+1} ===")
    print(f"QBER: {result.base_result.sample.qber:.4f}")
    print(f"Anomaly: {result.base_result.anomaly.is_anomaly}")
    print(f"Root Cause: {result.base_result.attribution.predicted_class}")
    
    if result.causal_inference:
        print(f"Causal Root: {result.causal_inference.root_cause_node}")
        print(f"Causal Confidence: {result.causal_inference.confidence:.3f}")
    
    if result.survival_forecast:
        print(f"Time to Failure: {result.survival_forecast.median_ttf:.1f}s")
        print(f"Risk Level: {result.survival_forecast.hazard_ratio:.2f}x baseline")
    
    print()

print("✓ Advanced diagnostics complete!")
