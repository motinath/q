"""
Base Hardware Interface

Abstract base class for all QKD hardware interfaces.

Author: VECTOR Q Development Team
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time


@dataclass
class HardwareTelemetry:
    """Standardized telemetry from QKD hardware."""
    timestamp: float
    link_id: str
    
    # Quantum metrics
    qber: float
    skr: float  # Secure Key Rate (bps)
    raw_key_rate: float  # Raw detection rate
    
    # Detection statistics
    signal_counts: int
    dark_counts: int
    singles_alice: int
    singles_bob: int
    
    # System parameters
    temperature: float  # °C
    optical_power: float  # dBm
    visibility: float  # [0, 1]
    
    # Error rates
    bit_error_rate: float
    phase_error_rate: Optional[float] = None
    
    # Timing
    clock_offset: Optional[float] = None  # ns
    timing_jitter: Optional[float] = None  # ps
    
    # System state
    status: str = "operational"  # operational, degraded, failed
    alarm_state: str = "normal"  # normal, warning, critical
    
    # Raw data (vendor-specific)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class HardwareCommand:
    """Command to send to QKD hardware."""
    command_type: str  # restart, calibrate, adjust_power, etc.
    parameters: Dict[str, Any]
    target_component: Optional[str] = None  # alice, bob, channel


@dataclass
class HardwareCommandResult:
    """Result of hardware command execution."""
    success: bool
    message: str
    execution_time: float
    new_state: Optional[Dict[str, Any]] = None


class BaseQKDHardwareInterface(ABC):
    """
    Abstract base class for QKD hardware interfaces.
    
    Defines standard API for:
    - Reading telemetry
    - Sending commands
    - Configuration management
    - Health monitoring
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        link_id: str,
        auth_credentials: Optional[Dict[str, str]] = None
    ):
        """
        Initialize hardware interface.
        
        Args:
            host: Hostname or IP address
            port: Communication port
            link_id: Unique identifier for this link
            auth_credentials: Authentication (e.g., {"username": "admin", "password": "..."})
        """
        self.host = host
        self.port = port
        self.link_id = link_id
        self.auth_credentials = auth_credentials or {}
        self.logger = logging.getLogger(f"{__name__}.{link_id}")
        
        self.is_connected = False
        self.last_telemetry_time = 0.0
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to hardware.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to hardware."""
        pass
    
    @abstractmethod
    def read_telemetry(self) -> HardwareTelemetry:
        """
        Read current telemetry from hardware.
        
        Returns:
            HardwareTelemetry object with standardized metrics
        
        Raises:
            ConnectionError: If hardware unreachable
            ValueError: If telemetry data invalid
        """
        pass
    
    @abstractmethod
    def send_command(self, command: HardwareCommand) -> HardwareCommandResult:
        """
        Send command to hardware.
        
        Args:
            command: HardwareCommand specifying action
        
        Returns:
            HardwareCommandResult with execution status
        """
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get detailed system status.
        
        Returns:
            Dictionary with system health, alarms, configuration
        """
        pass
    
    def read_telemetry_buffered(
        self,
        buffer_size: int = 10,
        interval: float = 1.0
    ) -> List[HardwareTelemetry]:
        """
        Read multiple telemetry samples over time.
        
        Args:
            buffer_size: Number of samples to collect
            interval: Time between samples (seconds)
        
        Returns:
            List of HardwareTelemetry samples
        """
        buffer = []
        
        for _ in range(buffer_size):
            try:
                sample = self.read_telemetry()
                buffer.append(sample)
                time.sleep(interval)
            except Exception as e:
                self.logger.warning(f"Telemetry read failed: {e}")
        
        return buffer
    
    def validate_connection(self) -> bool:
        """
        Validate connection is active and responsive.
        
        Returns:
            True if connection healthy
        """
        if not self.is_connected:
            return False
        
        try:
            # Attempt to read telemetry
            _ = self.read_telemetry()
            return True
        except:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# ============================================================================
# Mock Hardware Interface for Testing
# ============================================================================

class MockQKDHardware(BaseQKDHardwareInterface):
    """
    Mock hardware interface for testing and simulation.
    
    Returns synthetic telemetry without requiring real hardware.
    """
    
    def __init__(self, link_id: str = "mock_link"):
        super().__init__(
            host="localhost",
            port=0,
            link_id=link_id
        )
        self.simulated_qber = 0.05
        self.simulated_skr = 1000
    
    def connect(self) -> bool:
        """Simulate connection."""
        self.logger.info(f"Mock connection to {self.link_id}")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Simulate disconnection."""
        self.is_connected = False
        self.logger.info(f"Mock disconnection from {self.link_id}")
    
    def read_telemetry(self) -> HardwareTelemetry:
        """Generate synthetic telemetry."""
        import numpy as np
        
        if not self.is_connected:
            raise ConnectionError("Not connected to hardware")
        
        # Add random noise
        self.simulated_qber += np.random.normal(0, 0.002)
        self.simulated_qber = np.clip(self.simulated_qber, 0.02, 0.15)
        
        self.simulated_skr = max(0, 1200 - (self.simulated_qber - 0.05) * 10000)
        
        telemetry = HardwareTelemetry(
            timestamp=time.time(),
            link_id=self.link_id,
            qber=self.simulated_qber,
            skr=self.simulated_skr,
            raw_key_rate=self.simulated_skr * 1.5,
            signal_counts=int(15000 + np.random.normal(0, 500)),
            dark_counts=int(3000 + np.random.normal(0, 200)),
            singles_alice=int(20000 + np.random.normal(0, 1000)),
            singles_bob=int(20000 + np.random.normal(0, 1000)),
            temperature=25.0 + np.random.normal(0, 2),
            optical_power=-45.0 + np.random.normal(0, 1),
            visibility=0.98 + np.random.normal(0, 0.01),
            bit_error_rate=self.simulated_qber,
            phase_error_rate=self.simulated_qber * 1.2,
            clock_offset=np.random.normal(0, 50),
            timing_jitter=np.random.uniform(50, 150),
            status="operational",
            alarm_state="normal"
        )
        
        self.last_telemetry_time = telemetry.timestamp
        return telemetry
    
    def send_command(self, command: HardwareCommand) -> HardwareCommandResult:
        """Simulate command execution."""
        self.logger.info(f"Mock command: {command.command_type} with {command.parameters}")
        
        # Simulate command effects
        if command.command_type == "restart":
            self.simulated_qber = 0.05
            message = "System restarted successfully"
        elif command.command_type == "adjust_power":
            power_delta = command.parameters.get("delta_dbm", 0)
            message = f"Power adjusted by {power_delta} dBm"
        else:
            message = f"Command {command.command_type} executed"
        
        return HardwareCommandResult(
            success=True,
            message=message,
            execution_time=0.1
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Return mock system status."""
        return {
            "link_id": self.link_id,
            "status": "operational",
            "uptime_seconds": 86400,
            "alarms": [],
            "configuration": {
                "wavelength": 1550,
                "protocol": "BB84"
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Hardware Interface - Example\n")
    
    # Use mock hardware for testing
    with MockQKDHardware(link_id="test_link") as hardware:
        print("Connected to mock hardware\n")
        
        # Read single telemetry sample
        telemetry = hardware.read_telemetry()
        print(f"QBER: {telemetry.qber:.4f}")
        print(f"SKR: {telemetry.skr:.0f} bps")
        print(f"Temperature: {telemetry.temperature:.1f}°C")
        print(f"Status: {telemetry.status}\n")
        
        # Read buffered samples
        print("Reading 5 buffered samples...")
        samples = hardware.read_telemetry_buffered(buffer_size=5, interval=0.5)
        print(f"Collected {len(samples)} samples")
        print(f"QBER range: [{min(s.qber for s in samples):.4f}, {max(s.qber for s in samples):.4f}]\n")
        
        # Send command
        command = HardwareCommand(
            command_type="adjust_power",
            parameters={"delta_dbm": -2}
        )
        result = hardware.send_command(command)
        print(f"Command result: {result.message}")
        
        # Get system status
        status = hardware.get_system_status()
        print(f"\nSystem Status: {status['status']}")
    
    print("\n✓ Hardware interface example complete")
