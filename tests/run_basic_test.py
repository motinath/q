"""
Basic Q-SENTINEL Test - Run diagnostics for 10 cycles
"""

from streaming_pipeline.qkd_network_orchestrator import QKDNetworkOrchestrator
from physics_engine.quantum_telemetry_emulator import QuantumTelemetryEmulator

# Initialize system
emulator = QuantumTelemetryEmulator(fiber_distance_km=25.0)
orchestrator = QKDNetworkOrchestrator(telemetry_source=emulator)

print("Running Q-SENTINEL Basic Diagnostics...\n")

# Run 10 diagnostic cycles
for i in range(10):
    result = orchestrator.process_step(dt_seconds=1.0)
    
    print(f"Cycle {i+1}:")
    print(f"  QBER: {result.sample.qber:.4f}")
    print(f"  Anomaly: {result.anomaly.is_anomaly}")
    print(f"  Root Cause: {result.attribution.predicted_class}")
    print(f"  Confidence: {result.attribution.confidence:.3f}")
    print()

print("✓ Basic diagnostics complete!")
