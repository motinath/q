"""
Hardware-in-the-Loop (HIL) Telemetry Acquisition Interface
Standard: ETSI GS QKD 014 / Layer 1 Source A Interface
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class HardwareTelemetrySnapshot:
    """Represents a measured telemetry sample from physical hardware."""
    timestamp: float
    is_hardware_connected: bool
    source_name: str
    cpu_temperature_celsius: Optional[float]
    ambient_temperature_celsius: Optional[float]
    hardware_status_message: str
    metadata: Dict[str, Any]


class BaseTelemetrySource(ABC):
    """Abstract Base Class for swappable quantum network telemetry sources."""
    
    @abstractmethod
    def acquire_sample(self) -> Dict[str, Any]:
        """Acquire a single synchronized telemetry snapshot."""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Query physical connection status."""
        pass


class HardwareTelemetrySource(BaseTelemetrySource):
    """
    Layer 1 Source A: Real Hardware-in-the-Loop interface.
    Probes physical CPU thermal sensors or serial interfaces (e.g. Arduino/ESP32 detector monitors).
    Strict Rule Compliance:
    - Never fabricates readings. If no hardware sensor is connected, reports is_connected = False
      and returns None for unmeasured physical values.
    """
    
    def __init__(self, serial_port: Optional[str] = None, baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self._check_hardware_availability()
        
    def _check_hardware_availability(self) -> None:
        """Attempts to discover hardware sensors without throwing unhandled exceptions."""
        if self.serial_port:
            try:
                import serial
                self.serial_connection = serial.Serial(self.serial_port, self.baud_rate, timeout=1.0)
                logger.info(f"Connected to QKD Hardware Sensor on port {self.serial_port}")
            except Exception as e:
                logger.warning(f"Could not connect to serial port {self.serial_port}: {e}")
                self.serial_connection = None

    def is_connected(self) -> bool:
        """Returns True only if an active physical telemetry device or sensor is readable."""
        if self.serial_connection and self.serial_connection.is_open:
            return True
        if PSUTIL_AVAILABLE:
            try:
                temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
                if temps:
                    return True
            except Exception:
                pass
        return False

    def acquire_sample(self) -> HardwareTelemetrySnapshot:
        """
        Reads genuine hardware temperature / status.
        Does not mock or simulate values.
        """
        current_time = time.time()
        cpu_temp = None
        
        if PSUTIL_AVAILABLE and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Look for coretemp or cpu_thermal
                    for name, entries in temps.items():
                        if entries:
                            cpu_temp = float(entries[0].current)
                            break
            except Exception as e:
                logger.debug(f"psutil temperature read failed: {e}")
                
        connected = self.is_connected()
        status_msg = (
            f"Hardware Active (CPU Temp: {cpu_temp:.1f} C)" 
            if (connected and cpu_temp is not None)
            else "HIL Telemetry: Physical QKD hardware not connected / Standby"
        )
        
        return HardwareTelemetrySnapshot(
            timestamp=current_time,
            is_hardware_connected=connected,
            source_name="HIL_Source_A",
            cpu_temperature_celsius=cpu_temp,
            ambient_temperature_celsius=None,
            hardware_status_message=status_msg,
            metadata={
                "serial_port": self.serial_port,
                "psutil_available": PSUTIL_AVAILABLE
            }
        )
