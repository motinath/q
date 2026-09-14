"""
ID Quantique Hardware Interface

Interface for ID Quantique QKD systems (Clavis, Cerberis).
Uses SNMP and proprietary API.

Author: Q-SENTINEL Development Team
Version: 2.0.0
"""

import logging
from typing import Dict, Optional, Any
import time

from .base_hardware_interface import (
    BaseQKDHardwareInterface,
    HardwareTelemetry,
    HardwareCommand,
    HardwareCommandResult
)

try:
    from pysnmp.hlapi import *
    PYSNMP_AVAILABLE = True
except ImportError:
    PYSNMP_AVAILABLE = False
    logging.warning("pysnmp not installed. ID Quantique interface will use mock mode.")


class IDQuantiqueInterface(BaseQKDHardwareInterface):
    """
    Interface for ID Quantique QKD systems.
    
    Supported models:
    - Clavis2
    - Clavis3
    - Cerberis3
    
    Communication protocol:
    - SNMP v2c/v3 for telemetry
    - REST API for commands (if available)
    - SSH for advanced configuration
    """
    
    # ID Quantique SNMP OIDs (example - would need actual OIDs from vendor)
    OID_QBER = "1.3.6.1.4.1.12345.1.1.1"
    OID_SKR = "1.3.6.1.4.1.12345.1.1.2"
    OID_RAW_RATE = "1.3.6.1.4.1.12345.1.1.3"
    OID_SIGNAL_COUNTS = "1.3.6.1.4.1.12345.1.2.1"
    OID_DARK_COUNTS = "1.3.6.1.4.1.12345.1.2.2"
    OID_TEMPERATURE = "1.3.6.1.4.1.12345.1.3.1"
    OID_STATUS = "1.3.6.1.4.1.12345.1.4.1"
    
    def __init__(
        self,
        host: str,
        port: int = 161,
        link_id: str = "idq_link",
        community: str = "public",
        snmp_version: str = "2c"
    ):
        """
        Initialize ID Quantique interface.
        
        Args:
            host: Device IP address
            port: SNMP port (default 161)
            link_id: Link identifier
            community: SNMP community string
            snmp_version: SNMP version ("2c" or "3")
        """
        super().__init__(host, port, link_id)
        self.community = community
        self.snmp_version = snmp_version
        
        if not PYSNMP_AVAILABLE:
            self.logger.warning("pysnmp unavailable. Using mock mode.")
            self.use_mock = True
        else:
            self.use_mock = False
    
    def connect(self) -> bool:
        """Establish SNMP connection."""
        if self.use_mock:
            self.is_connected = True
            return True
        
        try:
            # Test SNMP connectivity with a simple GET
            iterator = getCmd(
                SnmpEngine(),
                CommunityData(self.community),
                UdpTransportTarget((self.host, self.port), timeout=2, retries=1),
                ContextData(),
                ObjectType(ObjectIdentity(self.OID_STATUS))
            )
            
            errorIndication, errorStatus, errorIndex, varBinds = next(iterator)
            
            if errorIndication or errorStatus:
                self.logger.error(f"SNMP connection failed: {errorIndication or errorStatus}")
                return False
            
            self.is_connected = True
            self.logger.info(f"Connected to ID Quantique device at {self.host}")
            return True
        
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close SNMP connection."""
        self.is_connected = False
        self.logger.info("Disconnected from ID Quantique device")
    
    def read_telemetry(self) -> HardwareTelemetry:
        """Read telemetry via SNMP."""
        if not self.is_connected:
            raise ConnectionError("Not connected to device")
        
        if self.use_mock:
            return self._read_mock_telemetry()
        
        try:
            # Fetch all telemetry OIDs
            telemetry_data = self._snmp_get_bulk([
                self.OID_QBER,
                self.OID_SKR,
                self.OID_RAW_RATE,
                self.OID_SIGNAL_COUNTS,
                self.OID_DARK_COUNTS,
                self.OID_TEMPERATURE,
                self.OID_STATUS
            ])
            
            # Parse telemetry
            telemetry = HardwareTelemetry(
                timestamp=time.time(),
                link_id=self.link_id,
                qber=float(telemetry_data.get(self.OID_QBER, 0.05)),
                skr=float(telemetry_data.get(self.OID_SKR, 1000)),
                raw_key_rate=float(telemetry_data.get(self.OID_RAW_RATE, 1500)),
                signal_counts=int(telemetry_data.get(self.OID_SIGNAL_COUNTS, 15000)),
                dark_counts=int(telemetry_data.get(self.OID_DARK_COUNTS, 3000)),
                singles_alice=int(telemetry_data.get("singles_alice", 20000)),
                singles_bob=int(telemetry_data.get("singles_bob", 20000)),
                temperature=float(telemetry_data.get(self.OID_TEMPERATURE, 25.0)),
                optical_power=-45.0,  # Would read from device
                visibility=0.98,  # Would calculate from counts
                bit_error_rate=float(telemetry_data.get(self.OID_QBER, 0.05)),
                status=self._parse_status(telemetry_data.get(self.OID_STATUS, "1")),
                alarm_state="normal",
                raw_data=telemetry_data
            )
            
            return telemetry
        
        except Exception as e:
            self.logger.error(f"Telemetry read failed: {e}")
            raise
    
    def _snmp_get_bulk(self, oids: list) -> Dict[str, Any]:
        """Perform SNMP GET for multiple OIDs."""
        if not PYSNMP_AVAILABLE:
            return {}
        
        results = {}
        
        try:
            for oid in oids:
                iterator = getCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((self.host, self.port)),
                    ContextData(),
                    ObjectType(ObjectIdentity(oid))
                )
                
                errorIndication, errorStatus, errorIndex, varBinds = next(iterator)
                
                if not errorIndication and not errorStatus:
                    for varBind in varBinds:
                        results[str(varBind[0])] = str(varBind[1])
        
        except Exception as e:
            self.logger.warning(f"SNMP GET failed: {e}")
        
        return results
    
    def _parse_status(self, status_code: str) -> str:
        """Parse ID Quantique status code to standard status."""
        status_map = {
            "1": "operational",
            "2": "degraded",
            "3": "failed",
            "4": "maintenance"
        }
        return status_map.get(status_code, "unknown")
    
    def send_command(self, command: HardwareCommand) -> HardwareCommandResult:
        """
        Send command to ID Quantique device.
        
        Commands typically sent via SSH or proprietary API.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to device")
        
        self.logger.info(f"Sending command: {command.command_type}")
        
        # In production, would use SSH/API
        # For now, simulate
        if self.use_mock or True:  # Always simulate for safety
            return HardwareCommandResult(
                success=True,
                message=f"Command {command.command_type} simulated (ID Quantique)",
                execution_time=0.5
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status from ID Quantique device."""
        if not self.is_connected:
            raise ConnectionError("Not connected to device")
        
        # Would read comprehensive status via SNMP/API
        return {
            "link_id": self.link_id,
            "vendor": "ID Quantique",
            "model": "Clavis3",
            "status": "operational",
            "firmware_version": "3.2.1",
            "uptime_seconds": 86400,
            "alarms": [],
            "configuration": {
                "protocol": "BB84",
                "wavelength": 1550,
                "alice_location": "Node_A",
                "bob_location": "Node_B"
            }
        }
    
    def _read_mock_telemetry(self) -> HardwareTelemetry:
        """Generate mock telemetry for testing."""
        import numpy as np
        
        return HardwareTelemetry(
            timestamp=time.time(),
            link_id=self.link_id,
            qber=0.05 + np.random.normal(0, 0.005),
            skr=1000 + np.random.normal(0, 50),
            raw_key_rate=1500 + np.random.normal(0, 75),
            signal_counts=int(15000 + np.random.normal(0, 500)),
            dark_counts=int(3000 + np.random.normal(0, 200)),
            singles_alice=20000,
            singles_bob=20000,
            temperature=25.0 + np.random.normal(0, 2),
            optical_power=-45.0,
            visibility=0.98,
            bit_error_rate=0.05,
            status="operational",
            alarm_state="normal"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("ID Quantique Hardware Interface - Example\n")
    
    # Initialize interface
    idq = IDQuantiqueInterface(
        host="192.168.1.100",
        link_id="idq_link_1",
        community="public"
    )
    
    # Connect
    if idq.connect():
        print("Connected successfully\n")
        
        # Read telemetry
        telemetry = idq.read_telemetry()
        print(f"QBER: {telemetry.qber:.4f}")
        print(f"SKR: {telemetry.skr:.0f} bps")
        print(f"Temperature: {telemetry.temperature:.1f}°C")
        print(f"Status: {telemetry.status}\n")
        
        # Get system status
        status = idq.get_system_status()
        print(f"Vendor: {status['vendor']}")
        print(f"Model: {status['model']}")
        print(f"Firmware: {status['firmware_version']}\n")
        
        # Disconnect
        idq.disconnect()
        print("Disconnected")
    else:
        print("Connection failed")
    
    print("\n✓ ID Quantique interface example complete")
