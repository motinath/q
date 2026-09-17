"""
Module A: Operator Feedback Capture & Labeled Event Store
Captures structured ground-truth resolutions from QNOC operators for AMBIGUOUS_ANOMALY
and PHYSICS_VETO cases, feeding the continuous learning flywheel.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class OperatorLabeledEvent:
    """Represents a human-verified ground-truth resolution event."""
    event_id: str                          # UUID
    timestamp: str                         # ISO-8601 string
    link_id: str

    # What the model saw
    feature_vector: List[float]            # 33-dimensional feature vector
    model_predicted_class: int             # 0 to 9
    model_confidence: float
    physics_guard_verdict: str             # "ML + Physics Agree" | "Physics Contradiction" | "Advisory"

    # What actually happened (ground truth assigned by operator)
    operator_assigned_class: int           # 0 to 9
    operator_confidence: str               # "certain" | "probable" | "uncertain"
    operator_id: str
    operator_notes: str                    # Free text for audit trail

    # Resolution metadata
    resolution_method: str                 # "visual_inspection" | "physical_test" | "hardware_log_crosscheck" | "attack_confirmed_external"
    corroborating_evidence: str            # Physical confirmation (e.g., "Technician verified fiber bend at splice box 4")

    # Lifecycle tracking
    used_in_training: bool = False
    training_run_id: Optional[str] = None


class OperatorFeedbackStore:
    """
    SQLite-backed store for operator feedback with audit integrity.
    Maintains the labeled_events table used to train candidate models.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "vector_q_audit.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operator_labeled_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    link_id TEXT NOT NULL,
                    feature_vector_json TEXT NOT NULL,
                    model_predicted_class INTEGER NOT NULL,
                    model_confidence REAL NOT NULL,
                    physics_guard_verdict TEXT NOT NULL,
                    operator_assigned_class INTEGER NOT NULL,
                    operator_confidence TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    operator_notes TEXT,
                    resolution_method TEXT NOT NULL,
                    corroborating_evidence TEXT,
                    used_in_training INTEGER DEFAULT 0,
                    training_run_id TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def record_resolution(
        self,
        link_id: str,
        feature_vector: List[float],
        model_predicted_class: int,
        model_confidence: float,
        physics_guard_verdict: str,
        operator_assigned_class: int,
        operator_confidence: str,
        operator_id: str,
        resolution_method: str,
        corroborating_evidence: str,
        operator_notes: str = "",
    ) -> OperatorLabeledEvent:
        """
        Records a new operator resolution into the feedback store.
        """
        event = OperatorLabeledEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            link_id=link_id,
            feature_vector=list(feature_vector),
            model_predicted_class=int(model_predicted_class),
            model_confidence=float(model_confidence),
            physics_guard_verdict=str(physics_guard_verdict),
            operator_assigned_class=int(operator_assigned_class),
            operator_confidence=operator_confidence.lower().strip(),
            operator_id=operator_id.strip(),
            operator_notes=operator_notes.strip(),
            resolution_method=resolution_method.strip(),
            corroborating_evidence=corroborating_evidence.strip(),
            used_in_training=False,
            training_run_id=None,
        )

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO operator_labeled_events (
                    event_id, timestamp, link_id, feature_vector_json,
                    model_predicted_class, model_confidence, physics_guard_verdict,
                    operator_assigned_class, operator_confidence, operator_id,
                    operator_notes, resolution_method, corroborating_evidence,
                    used_in_training, training_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.link_id,
                json.dumps(event.feature_vector),
                event.model_predicted_class,
                event.model_confidence,
                event.physics_guard_verdict,
                event.operator_assigned_class,
                event.operator_confidence,
                event.operator_id,
                event.operator_notes,
                event.resolution_method,
                event.corroborating_evidence,
                0,
                None,
            ))
            conn.commit()
        finally:
            conn.close()

        return event

    def get_unused_labels(self, min_confidence: str = "probable") -> List[OperatorLabeledEvent]:
        """
        Returns all labeled events ready for training.
        CRITICAL SAFEGUARD: 'uncertain' labels are excluded by default.
        """
        valid_confidences = ["certain"]
        if min_confidence == "probable":
            valid_confidences.append("probable")
        elif min_confidence == "all":
            valid_confidences.extend(["probable", "uncertain"])

        placeholders = ",".join("?" for _ in valid_confidences)
        query = f"""
            SELECT * FROM operator_labeled_events
            WHERE used_in_training = 0
            AND operator_confidence IN ({placeholders})
            ORDER BY timestamp ASC
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(query, valid_confidences).fetchall()
            events = []
            for r in rows:
                events.append(OperatorLabeledEvent(
                    event_id=r["event_id"],
                    timestamp=r["timestamp"],
                    link_id=r["link_id"],
                    feature_vector=json.loads(r["feature_vector_json"]),
                    model_predicted_class=r["model_predicted_class"],
                    model_confidence=r["model_confidence"],
                    physics_guard_verdict=r["physics_guard_verdict"],
                    operator_assigned_class=r["operator_assigned_class"],
                    operator_confidence=r["operator_confidence"],
                    operator_id=r["operator_id"],
                    operator_notes=r["operator_notes"] or "",
                    resolution_method=r["resolution_method"],
                    corroborating_evidence=r["corroborating_evidence"] or "",
                    used_in_training=bool(r["used_in_training"]),
                    training_run_id=r["training_run_id"],
                ))
            return events
        finally:
            conn.close()

    def mark_used_in_training(self, event_ids: List[str], training_run_id: str) -> None:
        """Marks a batch of events as ingested into a specific retraining run."""
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        query = f"""
            UPDATE operator_labeled_events
            SET used_in_training = 1, training_run_id = ?
            WHERE event_id IN ({placeholders})
        """
        params = [training_run_id] + list(event_ids)
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics on operator feedback accumulation."""
        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM operator_labeled_events").fetchone()[0]
            unused = conn.execute("SELECT COUNT(*) FROM operator_labeled_events WHERE used_in_training = 0").fetchone()[0]
            eligible = conn.execute(
                "SELECT COUNT(*) FROM operator_labeled_events WHERE used_in_training = 0 AND operator_confidence IN ('certain', 'probable')"
            ).fetchone()[0]
            by_class_rows = conn.execute(
                "SELECT operator_assigned_class, COUNT(*) as cnt FROM operator_labeled_events GROUP BY operator_assigned_class"
            ).fetchall()
            by_class = {row["operator_assigned_class"]: row["cnt"] for row in by_class_rows}
            return {
                "total_labeled_events": total,
                "unused_events": unused,
                "training_eligible_events": eligible,
                "class_distribution": by_class,
            }
        finally:
            conn.close()
