"""
Hardware Interface Test - Test with mock hardware
"""

from streaming_pipeline.advanced_orchestrator import AdvancedQKDOrchestrator
from hardware_interface import MockQKDHardware

# Initialize orchestrator
orchestrator = AdvancedQKDOrchestrator()

# Connect mock hardware (for testing)
hardware = MockQKDHardware(link_id="test_link")
orchestrator.connect_hardware(hardware)

print("Running Q-SENTINEL Hardware Test...\n")

# Run monitoring
for i in range(5):
    result = orchestrator.process_single_timestep()
    
    print(f"Reading {i+1}:")
    if result.hardware_telemetry:
        print(f"  Hardware QBER: {result.hardware_telemetry.qber:.4f}")
        print(f"  Hardware SKR: {result.hardware_telemetry.skr:.0f} bps")
        print(f"  Hardware Status: {result.hardware_telemetry.status}")
    
    print(f"  AI Root Cause: {result.base_result.attribution.predicted_class}")
    print()

orchestrator.disconnect_hardware()
print("✓ Hardware test complete!")
