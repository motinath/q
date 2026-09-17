"""
VECTOR Q Installation Validation Script

Run this after installation to verify everything is working correctly.
"""

import sys
import os
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def validate_installation():
    """Comprehensive installation validation."""
    print("=" * 60)
    print("VECTOR Q INSTALLATION VALIDATOR")
    print("=" * 60)
    print()
    
    errors = []
    warnings = []
    
    # 1. Check Python version
    print("1. Checking Python version...")
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   [FAIL] Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)")
    elif sys.version_info > (3, 14):
        warnings.append(f"Python 3.10-3.14 recommended, found {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   [WARN] Python {sys.version_info.major}.{sys.version_info.minor} (3.10-3.14 tested)")
    else:
        print(f"   [OK] Python {sys.version_info.major}.{sys.version_info.minor} (Supported: 3.10 - 3.14)")
    
    # 2. Check required packages
    print("\n2. Checking required packages...")
    required = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 
                'joblib', 'streamlit', 'matplotlib', 'plotly', 'pytest', 'requests', 'shap', 'reportlab']
    
    package_map = {
        'scikit-learn': 'sklearn',
    }
    
    for package in required:
        import_name = package_map.get(package, package.replace('-', '_'))
        try:
            __import__(import_name)
            print(f"   [OK] {package}")
        except ImportError:
            errors.append(f"Missing required package: {package}")
            print(f"   [FAIL] {package} (NOT INSTALLED)")
    
    # 3. Check optional packages
    print("\n3. Checking optional packages...")
    optional = {
        'pgmpy': 'Dynamic Bayesian Networks',
        'lifelines': 'Survival Analysis',
        'nonconformist': 'Conformal Prediction',
        'torch': 'Graph Neural Networks',
        'pysnmp': 'SNMP Hardware Interface'
    }
    
    for package, feature in optional.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"   [OK] {package} ({feature})")
        except ImportError:
            warnings.append(f"Optional package {package} not installed - {feature} will use fallback")
            print(f"   [WARN] {package} (optional - {feature} will use fallback)")
    
    # 4. Check project structure
    print("\n4. Checking project structure...")
    required_dirs = [
        'config', 'physics_engine', 'anomaly_detection', 'physics_validation',
        'root_cause_attribution', 'predictive_maintenance', 'remediation_engine',
        'audit_logging', 'streaming_pipeline', 'validation_framework', 'tests',
        'models', 'hardware_interface', 'digital_twin', 'incident_intelligence',
        'operations_dashboard'
    ]
    
    project_root = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   [OK] {dir_name}/")
        else:
            errors.append(f"Missing directory: {dir_name}")
            print(f"   [FAIL] {dir_name}/ (MISSING)")
    
    # 5. Check trained models
    print("\n5. Checking trained models...")
    model_files = [
        'models/isolation_forest.joblib',
        'models/lightgbm_classifier.joblib'
    ]
    
    for model_file in model_files:
        model_path = project_root / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   [OK] {model_file} ({size_mb:.1f} MB)")
        else:
            warnings.append(f"Model not found: {model_file} - run training script")
            print(f"   [WARN] {model_file} (needs training)")
    
    # 6. Test imports
    print("\n6. Testing module imports...")
    test_imports = [
        ('streaming_pipeline.qkd_network_orchestrator', 'QKDNetworkOrchestrator'),
        ('streaming_pipeline.advanced_orchestrator', 'AdvancedQKDOrchestrator'),
        ('physics_engine.quantum_telemetry_emulator', 'QuantumTelemetryEmulator'),
        ('anomaly_detection.isolation_forest_detector', 'IsolationForestAnomalyDetector'),
        ('root_cause_attribution.lightgbm_classifier', 'LightGBMRootCauseClassifier'),
        ('hardware_interface', 'MockQKDHardware'),
    ]
    
    sys.path.insert(0, str(project_root))
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   [OK] {module_name}.{class_name}")
        except Exception as e:
            errors.append(f"Import failed: {module_name}.{class_name} - {str(e)}")
            print(f"   [FAIL] {module_name}.{class_name} (FAILED)")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if not errors and not warnings:
        print("[OK] All checks passed! Installation is complete and ready.")
        print("\nNext steps:")
        print("  1. Train models: python scripts/train_attribution_models.py")
        print("  2. Run tests: python -m pytest tests/ -v")
        print("  3. Launch dashboard: streamlit run operations_dashboard/streamlit_app.py")
        return True
    
    if warnings:
        print(f"\n[WARN] {len(warnings)} Warning(s):")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print(f"\n[FAIL] {len(errors)} Error(s):")
        for e in errors:
            print(f"  - {e}")
        print("\nPlease fix errors before proceeding.")
        return False
    else:
        print("\n[OK] Installation OK (with warnings)")
        print("\nNext steps:")
        print("  1. Train models: python scripts/train_attribution_models.py")
        print("  2. Run tests: python -m pytest tests/ -v")
        return True

if __name__ == "__main__":
    success = validate_installation()
    sys.exit(0 if success else 1)
