"""
Layer 9 — Audit & Compliance (Tamper-Evident SQLite Event Log & Replay Engine)
Standard: ISO/IEC 27001 & ETSI GS QKD 014 Telemetry Audit Compliance
Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class QKDAuditRecord:
    """Represents an immutable, cryptographically hashed QKD event record."""
    timestamp: float
    iso_time: str
    qber: float
    skr_bps: float
    raw_counts_hz: float
    dark_counts_hz: float
    visibility: float
    temperature_celsius: float
    timing_jitter_ps: float
    anomaly_status: int           # 1 = Anomaly, 0 = Normal
    anomaly_score: float
    root_cause_diagnosis: str
    confidence: float
    is_physics_verified: int      # 1 = Yes, 0 = No
    physics_signature: str
    top_shap_features_json: str
    ptct_seconds: Optional[float]
    remediation_action_id: str
    action_executed: str
    operator: str
    sha256_hash: str = ""


class ComplianceAuditDatabase:
    """
    SQLite Audit Trail Database.
    Guarantees persistence, tamper-evident record hashing, and replayability for quantum operations.
    """

    def __init__(self, db_path: str = "q_sentinel_audit.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite database connection with row factory enabled."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True) if os.path.dirname(self.db_path) else None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        """Creates table schema and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qkd_telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    iso_time TEXT NOT NULL,
                    qber REAL NOT NULL,
                    skr_bps REAL NOT NULL,
                    raw_counts_hz REAL NOT NULL,
                    dark_counts_hz REAL NOT NULL,
                    visibility REAL NOT NULL,
                    temperature_celsius REAL NOT NULL,
                    timing_jitter_ps REAL NOT NULL,
                    anomaly_status INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    root_cause_diagnosis TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_physics_verified INTEGER NOT NULL,
                    physics_signature TEXT NOT NULL,
                    top_shap_features_json TEXT NOT NULL,
                    ptct_seconds REAL,
                    remediation_action_id TEXT NOT NULL,
                    action_executed TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON qkd_telemetry_events (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_diagnosis ON qkd_telemetry_events (root_cause_diagnosis)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_anomaly ON qkd_telemetry_events (anomaly_status)")
            conn.commit()

    @staticmethod
    def _compute_hash(record_dict: Dict[str, Any]) -> str:
        """Generates SHA-256 integrity hash for the event data."""
        clean_dict = {k: v for k, v in record_dict.items() if k != "sha256_hash" and k != "id"}
        payload = json.dumps(clean_dict, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log_event(
        self,
        timestamp: float,
        qber: float,
        skr_bps: float,
        raw_counts_hz: float,
        dark_counts_hz: float,
        visibility: float,
        temperature_celsius: float,
        timing_jitter_ps: float,
        anomaly_status: bool,
        anomaly_score: float,
        root_cause_diagnosis: str,
        confidence: float,
        is_physics_verified: bool,
        physics_signature: str,
        top_shap_features: List[Dict[str, Any]],
        ptct_seconds: Optional[float],
        remediation_action_id: str,
        action_executed: str = "PENDING_OPERATOR_CONFIRMATION",
        operator: str = "AUTONOMOUS_SENTINEL_CORE",
    ) -> int:
        """
        Logs a synchronized telemetry and diagnostic event to SQLite.
        """
        iso_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        shap_json = json.dumps(top_shap_features)
        
        event_data = {
            "timestamp": float(timestamp),
            "iso_time": iso_str,
            "qber": float(qber),
            "skr_bps": float(skr_bps),
            "raw_counts_hz": float(raw_counts_hz),
            "dark_counts_hz": float(dark_counts_hz),
            "visibility": float(visibility),
            "temperature_celsius": float(temperature_celsius),
            "timing_jitter_ps": float(timing_jitter_ps),
            "anomaly_status": int(1 if anomaly_status else 0),
            "anomaly_score": float(anomaly_score),
            "root_cause_diagnosis": str(root_cause_diagnosis),
            "confidence": float(confidence),
            "is_physics_verified": int(1 if is_physics_verified else 0),
            "physics_signature": str(physics_signature),
            "top_shap_features_json": shap_json,
            "ptct_seconds": float(ptct_seconds) if ptct_seconds is not None else None,
            "remediation_action_id": str(remediation_action_id),
            "action_executed": str(action_executed),
            "operator": str(operator),
        }
        
        sha = self._compute_hash(event_data)
        event_data["sha256_hash"] = sha
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO qkd_telemetry_events (
                    timestamp, iso_time, qber, skr_bps, raw_counts_hz, dark_counts_hz,
                    visibility, temperature_celsius, timing_jitter_ps, anomaly_status,
                    anomaly_score, root_cause_diagnosis, confidence, is_physics_verified,
                    physics_signature, top_shap_features_json, ptct_seconds,
                    remediation_action_id, action_executed, operator, sha256_hash
                ) VALUES (
                    :timestamp, :iso_time, :qber, :skr_bps, :raw_counts_hz, :dark_counts_hz,
                    :visibility, :temperature_celsius, :timing_jitter_ps, :anomaly_status,
                    :anomaly_score, :root_cause_diagnosis, :confidence, :is_physics_verified,
                    :physics_signature, :top_shap_features_json, :ptct_seconds,
                    :remediation_action_id, :action_executed, :operator, :sha256_hash
                )
            """, event_data)
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves most recent events sorted by descending timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM qkd_telemetry_events 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def fetch_replay_sequence(self, start_timestamp: float, end_timestamp: float) -> List[Dict[str, Any]]:
        """Retrieves chronological events for historical replay."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM qkd_telemetry_events 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start_timestamp, end_timestamp))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def verify_database_integrity(self) -> Tuple[bool, int, List[int]]:
        """
        Verifies cryptographic SHA-256 hashes of all stored events to detect tampering.
        Returns: (is_valid, total_checked, corrupted_record_ids)
        """
        corrupted = []
        total = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM qkd_telemetry_events ORDER BY id ASC")
            rows = cursor.fetchall()
            total = len(rows)
            for row in rows:
                row_dict = dict(row)
                stored_hash = row_dict["sha256_hash"]
                computed_hash = self._compute_hash(row_dict)
                if stored_hash != computed_hash:
                    corrupted.append(row_dict["id"])
                    
        return (len(corrupted) == 0, total, corrupted)
