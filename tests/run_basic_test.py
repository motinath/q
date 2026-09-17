"""
Basic VECTOR Q Test - Run diagnostics for 10 cycles
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

from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator

def test_basic_diagnostics():
    """Test basic VECTOR Q pipeline with 10 cycles."""
    # Initialize system
    emulator = QuantumTelemetryEmulator(random_seed=42)
    orchestrator = QKDNetworkOrchestrator(emulator=emulator)
    
    print("Running VECTOR Q Basic Diagnostics...\n")
    
    # Run 10 diagnostic cycles
    for i in range(10):
        result = orchestrator.process_step(dt_seconds=1.0)
        
        # Validate result
        assert result.sample.qber >= 0.0, "QBER must be non-negative"
        assert result.attribution.predicted_class in [
            "Normal", "Optical Misalignment", "Detector APD Degradation",
            "Timing Jitter", "Channel Attenuation Event", "Thermal Drift",
            "Intercept-Resend", "Detector Blinding", "Photon Number Splitting",
            "Time-Shift Attack"
        ], f"Invalid predicted class: {result.attribution.predicted_class}"
        assert 0.0 <= result.attribution.confidence <= 1.0, "Confidence must be in [0,1]"
        
        print(f"Cycle {i+1}:")
        print(f"  QBER: {result.sample.qber:.4f}")
        print(f"  Anomaly: {result.anomaly.is_anomaly}")
        print(f"  Root Cause: {result.attribution.predicted_class}")
        print(f"  Confidence: {result.attribution.confidence:.3f}")
        print()
    
    print("[PASS] Basic diagnostics complete!")

if __name__ == "__main__":
    try:
        test_basic_diagnostics()
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


