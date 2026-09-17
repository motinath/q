"""
Layer 9 — Audit & Compliance (Tamper-Evident SQLite Event Log & Replay Engine)
Standard: ISO/IEC 27001 & ETSI GS QKD 014 Telemetry Audit Compliance

P2.1: Implements proper SHA-256 hash chaining.
  - Each record stores a `previous_hash` field containing the SHA-256 hash of the
    immediately preceding record, forming a Merkle-style linked chain.
  - The genesis record uses `previous_hash = "GENESIS"`.
  - `verify_database_integrity()` verifies both per-record hashes AND the forward chain.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class QKDAuditRecord:
    """Represents an immutable, cryptographically chained QKD event record."""
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
    previous_hash: str = "GENESIS"   # P2.1: hash-chain link
    sha256_hash: str = ""


class ComplianceAuditDatabase:
    """
    SQLite Audit Trail Database.
    Guarantees persistence, tamper-evident chained record hashing, and replayability
    for quantum operations.  Each record's SHA-256 digest covers all payload fields
    plus the previous record's hash, forming a forward-linked chain that detects
    both record tampering and record deletion / reordering.
    """

    # Sentinel value for the very first record in the chain
    _GENESIS_HASH: str = "GENESIS"

    def __init__(self, db_path: str = "vector_q_audit.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite database connection with row factory enabled."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True) if os.path.dirname(self.db_path) else None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        """Creates table schema and indexes if they do not exist.
        P2.1: Adds `previous_hash` column for chain linking.
        """
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
                    previous_hash TEXT NOT NULL DEFAULT 'GENESIS',
                    sha256_hash TEXT NOT NULL
                )
            """)
            # Migrate existing DB that does not yet have previous_hash column
            try:
                cursor.execute("ALTER TABLE qkd_telemetry_events ADD COLUMN previous_hash TEXT NOT NULL DEFAULT 'GENESIS'")
            except sqlite3.OperationalError:
                pass  # Column already exists

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON qkd_telemetry_events (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_diagnosis ON qkd_telemetry_events (root_cause_diagnosis)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_anomaly ON qkd_telemetry_events (anomaly_status)")
            conn.commit()

    def _get_latest_hash(self, conn: sqlite3.Connection) -> str:
        """Returns the sha256_hash of the most recently inserted record, or GENESIS."""
        cursor = conn.cursor()
        cursor.execute("SELECT sha256_hash FROM qkd_telemetry_events ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row["sha256_hash"] if row else self._GENESIS_HASH

    @staticmethod
    def _compute_hash(record_dict: Dict[str, Any]) -> str:
        """
        Generates SHA-256 integrity hash for an event record.
        Excludes 'sha256_hash' itself and 'id' (auto-assigned by DB) from the payload.
        Includes 'previous_hash' so that the chain is tamper-evident.
        """
        clean_dict = {
            k: v for k, v in record_dict.items()
            if k not in ("sha256_hash", "id")
        }
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
        operator: str = "AUTONOMOUS_VECTOR_Q_CORE",
    ) -> int:
        """
        Logs a synchronized telemetry and diagnostic event to SQLite.
        P2.1: previous_hash is fetched inside the same connection transaction so the
        chain remains consistent even under concurrent access.
        """
        iso_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        shap_json = json.dumps(top_shap_features)

        with self._get_connection() as conn:
            # Fetch chain tip inside the same transaction to prevent races
            prev_hash = self._get_latest_hash(conn)

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
                "previous_hash": prev_hash,   # P2.1: chain link
            }

            sha = self._compute_hash(event_data)
            event_data["sha256_hash"] = sha

            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO qkd_telemetry_events (
                    timestamp, iso_time, qber, skr_bps, raw_counts_hz, dark_counts_hz,
                    visibility, temperature_celsius, timing_jitter_ps, anomaly_status,
                    anomaly_score, root_cause_diagnosis, confidence, is_physics_verified,
                    physics_signature, top_shap_features_json, ptct_seconds,
                    remediation_action_id, action_executed, operator,
                    previous_hash, sha256_hash
                ) VALUES (
                    :timestamp, :iso_time, :qber, :skr_bps, :raw_counts_hz, :dark_counts_hz,
                    :visibility, :temperature_celsius, :timing_jitter_ps, :anomaly_status,
                    :anomaly_score, :root_cause_diagnosis, :confidence, :is_physics_verified,
                    :physics_signature, :top_shap_features_json, :ptct_seconds,
                    :remediation_action_id, :action_executed, :operator,
                    :previous_hash, :sha256_hash
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
        Verifies the tamper-evident SHA-256 hash chain of all stored events.

        Two checks per record:
          1. Per-record hash: recompute SHA-256 from payload fields (including
             previous_hash) and compare against stored sha256_hash.
          2. Chain link: verify that this record's previous_hash equals the
             sha256_hash of the immediately preceding record.

        Returns:
            (is_valid, total_checked, corrupted_record_ids)
        """
        corrupted: List[int] = []
        total: int = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM qkd_telemetry_events ORDER BY id ASC")
            rows = cursor.fetchall()
            total = len(rows)

            expected_prev_hash = self._GENESIS_HASH
            for row in rows:
                row_dict = dict(row)
                record_id = row_dict["id"]

                # 1. Chain-link check
                if row_dict.get("previous_hash", self._GENESIS_HASH) != expected_prev_hash:
                    corrupted.append(record_id)
                    # Keep walking so we report all broken records
                    expected_prev_hash = row_dict.get("sha256_hash", "")
                    continue

                # 2. Per-record hash check
                stored_hash = row_dict["sha256_hash"]
                computed_hash = self._compute_hash(row_dict)
                if stored_hash != computed_hash:
                    corrupted.append(record_id)

                expected_prev_hash = stored_hash

        return (len(corrupted) == 0, total, corrupted)
