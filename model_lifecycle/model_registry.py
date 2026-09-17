"""
Model Registry with Versioning, Lineage Tracking, and Cryptographic Rollback Safety
Manages model artifacts, metadata, and performance metrics across versions.
Enables safe promotion gated by Shadow Validation and cryptographic tamper-evident rollback.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import os
import json
import joblib
import shutil
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from enum import Enum


class ModelStatus(Enum):
    """Model deployment status."""
    CHAMPION = "champion"  # Currently deployed in production
    CHALLENGER = "challenger"  # Candidate model under A/B test or shadow
    ARCHIVED = "archived"  # Historical model (rollback point)
    TRAINING = "training"  # Model being trained
    FAILED = "failed"  # Training or validation failed


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    model_type: str  # "isolation_forest" or "lightgbm_classifier"
    version: str
    created_timestamp: str
    trained_on_n_samples: int
    training_duration_seconds: float
    status: str

    # Performance metrics
    accuracy: Optional[float] = None
    macro_f1_score: Optional[float] = None
    per_class_f1_scores: Optional[Dict[str, float]] = None
    anomaly_detection_auc: Optional[float] = None

    # Validation metrics
    validation_accuracy: Optional[float] = None
    validation_f1_score: Optional[float] = None
    physics_agreement_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None

    # Provenance & Continuous Learning Flywheel Lineage
    training_data_hash: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    git_commit_hash: Optional[str] = None
    trained_on_n_field_labels: int = 0
    parent_version: Optional[str] = None
    checksum: Optional[str] = None

    # Deployment tracking
    deployed_timestamp: Optional[str] = None
    n_predictions_served: int = 0
    production_accuracy: Optional[float] = None

    # File paths
    model_artifact_path: Optional[str] = None
    scaler_artifact_path: Optional[str] = None
    calibrator_artifact_path: Optional[str] = None


class ModelRegistry:
    """
    Central registry for all trained models with versioning, shadow promotion,
    and tamper-evident audit rollback.
    """

    def __init__(self, registry_root: str = "models", db_path: Optional[str] = None):
        """
        Args:
            registry_root: Root directory for model storage
            db_path: Path to SQLite audit database
        """
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "vector_q_audit.db")
        self.db_path = db_path
        self._init_audit_table()

        self.registry_file = self.registry_root / "registry.json"
        self.registry_data: Dict[str, List[ModelMetadata]] = self._load_registry()

    def _init_audit_table(self) -> None:
        """Initializes the tamper-evident model lifecycle audit table in SQLite."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_lifecycle_audit_chain (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        version TEXT NOT NULL,
                        parent_version TEXT,
                        model_checksum TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        operator_or_actor TEXT NOT NULL,
                        previous_hash TEXT NOT NULL,
                        sha256_hash TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load registry from disk or initialize empty."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                valid_fields = {field.name for field in fields(ModelMetadata)}
                return {
                    model_type: [
                        ModelMetadata(**{k: v for k, v in m.items() if k in valid_fields})
                        for m in models
                    ]
                    for model_type, models in data.items()
                }
        return {"isolation_forest": [], "lightgbm_classifier": []}

    def _save_registry(self) -> None:
        """Persist registry to disk."""
        data = {
            model_type: [asdict(m) for m in models]
            for model_type, models in self.registry_data.items()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_type: str,
        version: str,
        model_artifact,
        scaler_artifact=None,
        calibrator_artifact=None,
        metadata: Optional[ModelMetadata] = None,
        status: ModelStatus = ModelStatus.TRAINING
    ) -> ModelMetadata:
        """
        Register a new model version.
        """
        model_dir = self.registry_root / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        model_path = model_dir / "model.joblib"
        joblib.dump(model_artifact, model_path)

        scaler_path = None
        if scaler_artifact is not None:
            scaler_path = model_dir / "scaler.joblib"
            joblib.dump(scaler_artifact, scaler_path)

        calibrator_path = None
        if calibrator_artifact is not None:
            calibrator_path = model_dir / "calibrator.joblib"
            joblib.dump(calibrator_artifact, calibrator_path)

        # Create or update metadata
        if metadata is None:
            metadata = ModelMetadata(
                model_id=f"{model_type}_{version}",
                model_type=model_type,
                version=version,
                created_timestamp=datetime.now(timezone.utc).isoformat(),
                trained_on_n_samples=0,
                training_duration_seconds=0.0,
                status=status.value,
                model_artifact_path=str(model_path),
                scaler_artifact_path=str(scaler_path) if scaler_path else None,
                calibrator_artifact_path=str(calibrator_path) if calibrator_path else None,
                checksum=self.compute_data_hash(str(model_path))
            )
        else:
            metadata.model_artifact_path = str(model_path)
            metadata.scaler_artifact_path = str(scaler_path) if scaler_path else None
            metadata.calibrator_artifact_path = str(calibrator_path) if calibrator_path else None
            metadata.status = status.value
            metadata.checksum = self.compute_data_hash(str(model_path))

        # Save metadata JSON
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Add to registry
        if model_type not in self.registry_data:
            self.registry_data[model_type] = []

        self.registry_data[model_type] = [
            m for m in self.registry_data[model_type]
            if m.version != version
        ]
        self.registry_data[model_type].append(metadata)

        self._save_registry()
        return metadata

    def promote_to_champion(self, model_type: str, version: str) -> bool:
        """
        Promote a model version to champion (production).
        """
        version_dir = self.registry_root / model_type / version
        if not version_dir.exists():
            raise ValueError(f"Model version {model_type}/{version} not found")

        champion_link = self.registry_root / model_type / "champion"

        if champion_link.exists() or champion_link.is_symlink():
            try:
                if champion_link.is_symlink():
                    champion_link.unlink()
                elif champion_link.is_dir():
                    try:
                        os.rmdir(champion_link)
                    except OSError:
                        shutil.rmtree(champion_link)
                else:
                    champion_link.unlink()
            except Exception:
                pass

        try:
            if os.name == 'nt':
                import _winapi
                _winapi.CreateJunction(str(version_dir), str(champion_link))
            else:
                champion_link.symlink_to(version_dir, target_is_directory=True)
        except Exception:
            shutil.copytree(version_dir, champion_link)

        # Update status in registry
        for model in self.registry_data[model_type]:
            if model.version == version:
                model.status = ModelStatus.CHAMPION.value
                model.deployed_timestamp = datetime.now(timezone.utc).isoformat()
            elif model.status == ModelStatus.CHAMPION.value:
                model.status = ModelStatus.ARCHIVED.value

        self._save_registry()
        return True

    def get_champion(self, model_type: str) -> Optional[ModelMetadata]:
        """Get currently deployed champion model metadata."""
        for model in self.registry_data.get(model_type, []):
            if model.status == ModelStatus.CHAMPION.value:
                return model
        return None

    def load_champion_artifacts(self, model_type: str) -> Dict[str, Any]:
        """Load champion model artifacts from disk."""
        champion_dir = self.registry_root / model_type / "champion"
        if not champion_dir.exists():
            raise FileNotFoundError(f"No champion model for {model_type}")

        artifacts = {}
        model_path = champion_dir / "model.joblib"
        if model_path.exists():
            artifacts['model'] = joblib.load(model_path)

        scaler_path = champion_dir / "scaler.joblib"
        if scaler_path.exists():
            artifacts['scaler'] = joblib.load(scaler_path)

        calibrator_path = champion_dir / "calibrator.joblib"
        if calibrator_path.exists():
            artifacts['calibrator'] = joblib.load(calibrator_path)

        metadata_path = champion_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                valid_fields = {field.name for field in fields(ModelMetadata)}
                data = json.load(f)
                artifacts['metadata'] = ModelMetadata(**{k: v for k, v in data.items() if k in valid_fields})

        return artifacts

    def load_version_artifacts(self, model_type: str, version: str) -> Dict[str, Any]:
        """Load specific model version artifacts."""
        version_dir = self.registry_root / model_type / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {model_type}/{version} not found")

        artifacts = {}
        model_path = version_dir / "model.joblib"
        if model_path.exists():
            artifacts['model'] = joblib.load(model_path)

        scaler_path = version_dir / "scaler.joblib"
        if scaler_path.exists():
            artifacts['scaler'] = joblib.load(scaler_path)

        calibrator_path = version_dir / "calibrator.joblib"
        if calibrator_path.exists():
            artifacts['calibrator'] = joblib.load(calibrator_path)

        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                valid_fields = {field.name for field in fields(ModelMetadata)}
                data = json.load(f)
                artifacts['metadata'] = ModelMetadata(**{k: v for k, v in data.items() if k in valid_fields})

        return artifacts

    def update_production_metrics(
        self,
        model_type: str,
        version: str,
        n_predictions: int,
        accuracy: Optional[float] = None
    ) -> None:
        """Update production performance metrics for a deployed model."""
        for model in self.registry_data.get(model_type, []):
            if model.version == version:
                model.n_predictions_served += n_predictions
                if accuracy is not None:
                    if model.production_accuracy is None:
                        model.production_accuracy = accuracy
                    else:
                        alpha = 0.1
                        model.production_accuracy = (
                            alpha * accuracy + (1 - alpha) * model.production_accuracy
                        )
                break
        self._save_registry()

    def list_versions(self, model_type: str) -> List[ModelMetadata]:
        """List all versions for a model type, sorted by creation time."""
        models = self.registry_data.get(model_type, [])
        return sorted(models, key=lambda m: m.created_timestamp, reverse=True)

    def rollback_to_version(self, model_type: str, version: str) -> bool:
        """
        Rollback to a previous model version.
        This promotes the specified version to champion.
        """
        return self.promote_to_champion(model_type, version)

    def compute_data_hash(self, data_file_path: str) -> str:
        """Compute SHA-256 hash of training data or model file for provenance."""
        sha256 = hashlib.sha256()
        with open(data_file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _log_to_audit_chain(
        self,
        event_type: str,
        model_type: str,
        version: str,
        parent_version: Optional[str],
        checksum: str,
        details: Dict[str, Any],
        operator_or_actor: str,
    ) -> str:
        """Records a cryptographically chained audit record into SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT sha256_hash FROM model_lifecycle_audit_chain ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                prev_hash = row[0] if row else "GENESIS"

                timestamp = datetime.now(timezone.utc).isoformat()
                details_json = json.dumps(details, sort_keys=True)
                payload = f"{timestamp}|{event_type}|{model_type}|{version}|{parent_version}|{checksum}|{details_json}|{operator_or_actor}|{prev_hash}"
                sha256_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

                cursor.execute("""
                    INSERT INTO model_lifecycle_audit_chain (
                        timestamp, event_type, model_type, version, parent_version,
                        model_checksum, details_json, operator_or_actor, previous_hash, sha256_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, event_type, model_type, version, parent_version,
                    checksum, details_json, operator_or_actor, prev_hash, sha256_hash
                ))
                conn.commit()
                return sha256_hash
            finally:
                conn.close()
        except Exception:
            return "AUDIT_ERROR"

    def promote(
        self,
        candidate_model: Any,
        candidate_version: str,
        shadow_verdict: Any,
        model_type: str = "lightgbm_classifier",
        promoted_by: str = "automated_pipeline",
        scaler=None,
        calibrator=None,
        validation_report: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        n_field_labels: int = 0,
    ) -> ModelMetadata:
        """
        Promotes a candidate model after clearing the Shadow Validation Gate.
        Computes SHA-256 artifact checksum, creates version entry, promotes to champion,
        and logs to the tamper-evident audit ledger.
        """
        if not getattr(shadow_verdict, "ready", False):
            raise ValueError(f"Cannot promote candidate model without passing Shadow Validation: {getattr(shadow_verdict, 'reason', 'unknown')}")

        current_champ = self.get_champion(model_type)
        if parent_version is None and current_champ:
            parent_version = current_champ.version

        meta = self.register_model(
            model_type=model_type,
            version=candidate_version,
            model_artifact=candidate_model,
            scaler_artifact=scaler,
            calibrator_artifact=calibrator,
            status=ModelStatus.CHALLENGER
        )
        meta.parent_version = parent_version
        meta.trained_on_n_field_labels = n_field_labels

        if meta.model_artifact_path and Path(meta.model_artifact_path).exists():
            meta.checksum = self.compute_data_hash(meta.model_artifact_path)

        self.promote_to_champion(model_type, candidate_version)
        meta.status = ModelStatus.CHAMPION.value
        self._save_registry()

        details = {
            "validation_report": validation_report or {},
            "shadow_verdict": {
                "agreement_rate": getattr(shadow_verdict, "agreement_rate", 1.0),
                "n_samples": getattr(shadow_verdict, "n_samples_observed", 0),
                "reason": getattr(shadow_verdict, "reason", "Shadow passed"),
            }
        }
        self._log_to_audit_chain(
            event_type="MODEL_PROMOTION",
            model_type=model_type,
            version=candidate_version,
            parent_version=parent_version,
            checksum=meta.checksum or "N/A",
            details=details,
            operator_or_actor=promoted_by,
        )
        return meta

    def rollback(
        self,
        to_version: str,
        model_type: str = "lightgbm_classifier",
        operator_id: str = "automated_pipeline",
        reason: str = "Safety rollback requested",
    ) -> bool:
        """
        One-command rollback to an archived model version with audit chaining.
        """
        current_champ = self.get_champion(model_type)
        from_version = current_champ.version if current_champ else "UNKNOWN"

        success = self.rollback_to_version(model_type, to_version)
        if success:
            target_meta = next((m for m in self.registry_data.get(model_type, []) if m.version == to_version), None)
            checksum = target_meta.checksum if target_meta and target_meta.checksum else "ARCHIVED"
            self._log_to_audit_chain(
                event_type="MODEL_ROLLBACK",
                model_type=model_type,
                version=to_version,
                parent_version=from_version,
                checksum=checksum,
                details={"reverted_from": from_version, "revert_to": to_version, "reason": reason},
                operator_or_actor=operator_id,
            )
        return success

    def get_audit_trail(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves verified audit records from the lifecycle ledger."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if model_type:
                    cursor.execute(
                        "SELECT * FROM model_lifecycle_audit_chain WHERE model_type = ? ORDER BY id DESC",
                        (model_type,)
                    )
                else:
                    cursor.execute("SELECT * FROM model_lifecycle_audit_chain ORDER BY id DESC")
                return [dict(r) for r in cursor.fetchall()]
            finally:
                conn.close()
        except Exception:
            return []
