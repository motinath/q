"""
Model Registry with Versioning and Performance Tracking

Manages model artifacts, metadata, and performance metrics across versions.
Enables safe rollback and champion/challenger comparison.

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import json
import joblib
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class ModelStatus(Enum):
    """Model deployment status."""
    CHAMPION = "champion"  # Currently deployed in production
    CHALLENGER = "challenger"  # Candidate model under A/B test
    ARCHIVED = "archived"  # Historical model
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
    
    # Provenance
    training_data_hash: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    git_commit_hash: Optional[str] = None
    
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
    Central registry for all trained models with versioning and rollback.
    
    Structure:
        models/
        ├── registry.json              # Master registry index
        ├── isolation_forest/
        │   ├── v1.0.0/
        │   │   ├── model.joblib
        │   │   ├── scaler.joblib
        │   │   └── metadata.json
        │   ├── v1.1.0/
        │   └── champion -> v1.0.0     # Symlink to production model
        └── lightgbm_classifier/
            ├── v1.0.0/
            ├── v1.1.0/
            ├── v2.0.0/
            └── champion -> v2.0.0
    """
    
    def __init__(self, registry_root: str = "models"):
        """
        Args:
            registry_root: Root directory for model storage
        """
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_root / "registry.json"
        self.registry_data: Dict[str, List[ModelMetadata]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, List[ModelMetadata]]:
        """Load registry from disk or initialize empty."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                # Deserialize metadata objects
                return {
                    model_type: [ModelMetadata(**m) for m in models]
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
        
        Args:
            model_type: "isolation_forest" or "lightgbm_classifier"
            version: Semantic version string (e.g., "1.2.0")
            model_artifact: Trained model object
            scaler_artifact: Optional scaler object
            calibrator_artifact: Optional calibration model
            metadata: Optional pre-constructed metadata
            status: Initial model status
        
        Returns:
            ModelMetadata object
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
                created_timestamp=datetime.now().isoformat(),
                trained_on_n_samples=0,
                training_duration_seconds=0.0,
                status=status.value,
                model_artifact_path=str(model_path),
                scaler_artifact_path=str(scaler_path) if scaler_path else None,
                calibrator_artifact_path=str(calibrator_path) if calibrator_path else None
            )
        else:
            metadata.model_artifact_path = str(model_path)
            metadata.scaler_artifact_path = str(scaler_path) if scaler_path else None
            metadata.calibrator_artifact_path = str(calibrator_path) if calibrator_path else None
            metadata.status = status.value
        
        # Save metadata JSON
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Add to registry
        if model_type not in self.registry_data:
            self.registry_data[model_type] = []
        
        # Remove existing entry for this version if present
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
        
        Creates a 'champion' symlink pointing to the version directory.
        """
        version_dir = self.registry_root / model_type / version
        if not version_dir.exists():
            raise ValueError(f"Model version {model_type}/{version} not found")
        
        champion_link = self.registry_root / model_type / "champion"
        
        # Remove existing champion link
        if champion_link.exists() or champion_link.is_symlink():
            champion_link.unlink()
        
        # Create new champion symlink (Windows: use directory junction)
        try:
            import os
            if os.name == 'nt':  # Windows
                import _winapi
                _winapi.CreateJunction(str(version_dir), str(champion_link))
            else:  # Unix
                champion_link.symlink_to(version_dir, target_is_directory=True)
        except Exception as e:
            # Fallback: copy directory
            if champion_link.exists():
                shutil.rmtree(champion_link)
            shutil.copytree(version_dir, champion_link)
        
        # Update status in registry
        for model in self.registry_data[model_type]:
            if model.version == version:
                model.status = ModelStatus.CHAMPION.value
                model.deployed_timestamp = datetime.now().isoformat()
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
                artifacts['metadata'] = ModelMetadata(**json.load(f))
        
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
                artifacts['metadata'] = ModelMetadata(**json.load(f))
        
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
                    # Running average
                    if model.production_accuracy is None:
                        model.production_accuracy = accuracy
                    else:
                        alpha = 0.1  # Smoothing factor
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
        """Compute SHA-256 hash of training data for provenance."""
        sha256 = hashlib.sha256()
        with open(data_file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
