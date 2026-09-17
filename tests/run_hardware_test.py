"""
Hardware Interface Test - Test with mock hardware
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
from hardware_interface import MockQKDHardware

def test_hardware_interface():
    """Test hardware interface with mock device."""
    # Initialize orchestrator
    orchestrator = AdvancedQKDOrchestrator()
    
    # Connect mock hardware (for testing)
    hardware = MockQKDHardware(link_id="test_link")
    orchestrator.connect_hardware(hardware)
    
    print("Running VECTOR Q Hardware Test...\n")
    
    # Run monitoring
    for i in range(5):
        result = orchestrator.process_single_timestep()
        
        # Validate result
        assert hasattr(result, 'hardware_telemetry'), "Missing hardware telemetry"
        
        print(f"Reading {i+1}:")
        if result.hardware_telemetry:
            assert result.hardware_telemetry.qber >= 0.0, "Hardware QBER must be non-negative"
            assert result.hardware_telemetry.skr >= 0.0, "Hardware SKR must be non-negative"
            assert result.hardware_telemetry.status in ["operational", "degraded", "failed"], \
                f"Invalid hardware status: {result.hardware_telemetry.status}"
            
            print(f"  Hardware QBER: {result.hardware_telemetry.qber:.4f}")
            print(f"  Hardware SKR: {result.hardware_telemetry.skr:.0f} bps")
            print(f"  Hardware Status: {result.hardware_telemetry.status}")
        
        print(f"  AI Root Cause: {result.base_result.attribution.predicted_class}")
        print()
    
    orchestrator.disconnect_hardware()
    print("[PASS] Hardware test complete!")

if __name__ == "__main__":
    try:
        test_hardware_interface()
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

