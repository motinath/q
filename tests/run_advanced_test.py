"""
Advanced VECTOR Q Test - Run with all AI features
"""

import sys
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator

def test_advanced_features():
    """Test advanced VECTOR Q features over 5 cycles."""
    # Initialize with advanced AI
    orchestrator = AdvancedQKDOrchestrator(
        enable_causal_inference=True,
        enable_survival_analysis=True,
        enable_counterfactual=True
    )
    
    print("Running VECTOR Q Advanced Diagnostics...\n")
    
    # Run 5 diagnostic cycles with advanced analytics
    for i in range(5):
        result = orchestrator.process_single_timestep()
        
        # Validate result
        assert hasattr(result, 'base_result'), "Missing base_result"
        assert result.base_result.sample.qber >= 0.0, "QBER must be non-negative"
        
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
    
    print("[PASS] Advanced diagnostics complete!")

if __name__ == "__main__":
    try:
        test_advanced_features()
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

