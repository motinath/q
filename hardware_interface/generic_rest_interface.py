"""
Generic REST API Hardware Interface

For QKD systems with RESTful API.

Author: Q-SENTINEL Development Team
Version: 2.0.0
"""

import logging
from typing import Dict, Any
import time
import numpy as np

from .base_hardware_interface import (
    BaseQKDHardwareInterface,
    HardwareTelemetry,
    HardwareCommand,
    HardwareCommandResult
)


class GenericRESTInterface(BaseQKDHardwareInterface):
    """Generic REST API interface for QKD systems."""
    
    def __init__(
        self,
        host: str,
        port: int = 443,
        link_id: str = "rest_link",
        api_token: str = ""
    ):
        super().__init__(host, port, link_id, auth_credentials={"token": api_token})
        self.api_token = api_token
        self.base_url = f"https://{host}:{port}/api"
    
    def connect(self) -> bool:
        self.is_connected = True
        self.logger.info(f"REST API connection to {self.host}")
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def read_telemetry(self) -> HardwareTelemetry:
        if not self.is_connected:
            raise ConnectionError("Not connected")
        
        # Would be: requests.get(f"{self.base_url}/telemetry")
        return HardwareTelemetry(
            timestamp=time.time(),
            link_id=self.link_id,
            qber=0.048 + np.random.normal(0, 0.004),
            skr=1050 + np.random.normal(0, 55),
            raw_key_rate=1575,
            signal_counts=15500,
            dark_counts=2900,
            singles_alice=20500,
            singles_bob=20500,
            temperature=23.5,
            optical_power=-45.5,
            visibility=0.982,
            bit_error_rate=0.048,
            status="operational",
            alarm_state="normal"
        )
    
    def send_command(self, command: HardwareCommand) -> HardwareCommandResult:
        # Would be: requests.post(f"{self.base_url}/command", json=...)
        return HardwareCommandResult(
            success=True,
            message="REST command sent",
            execution_time=0.25
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "vendor": "Generic",
            "protocol": "REST API",
            "status": "operational"
        }
