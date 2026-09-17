# VECTOR Q Troubleshooting Guide

This guide covers common issues and solutions.

---

## Installation Issues

### Issue: "ModuleNotFoundError" when running scripts

**Cause**: Python can't find the VECTOR Q modules

**Solutions**:
```bash
# Solution 1: Run from project root
cd path/to/vector-q
python scripts/train_attribution_models.py

# Solution 2: Use pytest (automatically adds project to path)
python -m pytest tests/run_basic_test.py -v -s

# Solution 3: Add to PYTHONPATH (Windows)
$env:PYTHONPATH = "path\to\vector-q"

# Solution 3: Add to PYTHONPATH (Linux/Mac)
export PYTHONPATH="/path/to/vector-q:$PYTHONPATH"
```

---

### Issue: "numpy.dtype size changed" warning

**Cause**: Version mismatch between numpy and other packages

**Solution**:
```bash
# Reinstall with compatible versions
pip uninstall numpy pandas scipy scikit-learn -y
pip install numpy==1.26.4 pandas scipy scikit-learn
```

---

### Issue: Training script fails with "No module named 'shap'"

**Cause**: Missing SHAP package

**Solution**:
```bash
pip install shap
```

---

## Model Training Issues

### Issue: "AttributeError: 'LightGBMRootCauseClassifier' object has no attribute 'scaler'"

**Cause**: Outdated training script

**Solution**: Already fixed in latest version. If still occurs:
```bash
# Re-download train_attribution_models.py from repository
# Or update manually - the classifier doesn't use a scaler
```

---

### Issue: Training takes too long (>5 minutes)

**Cause**: Large dataset generation or slow CPU

**Solutions**:
```bash
# Reduce training samples (edit scripts/train_attribution_models.py)
# Change: n_training_runs=70 → n_training_runs=30
# Change: n_test_runs=15 → n_test_runs=10

# Or use faster settings
python scripts/train_attribution_models.py --quick
```

---

## Testing Issues

### Issue: Tests fail with "TypeError: __init__() got an unexpected keyword argument"

**Cause**: Incorrect parameter names in test files

**Solution**: All test files have been updated with correct parameters:
- `QuantumTelemetryEmulator(random_seed=42)` ✓
- `QKDNetworkOrchestrator(emulator=emulator)` ✓
- `MockQKDHardware(link_id="test")` ✓

If error persists, verify you're using the latest test files.

---

### Issue: "collected 0 items" when running pytest

**Cause**: Test file doesn't have test functions or proper structure

**Solution**:
```bash
# Use -s flag to see output from print statements
python -m pytest tests/run_basic_test.py -v -s

# The test files now have proper test functions
# They should work with both pytest and direct execution:
python tests/run_basic_test.py
```

---

### Issue: Physics validation test fails with "ML Prediction Uncertain"

**Cause**: Test uses unrealistic telemetry values

**Solution**: Already fixed. The test now uses realistic values:
- `raw_counts_hz=2000000.0` (2 Mcps for 25km link)
- `signal_to_noise_ratio=4500.0` (realistic SNR)

---

## Runtime Issues

### Issue: "SQLite database is locked"

**Cause**: Multiple processes accessing audit database

**Solution**:
```bash
# Close all VECTOR Q processes
# Delete the lock file
rm vector_q_audit.db-journal

# Or use a new database
rm vector_q_audit.db
```

---

### Issue: High memory usage (>4GB)

**Cause**: Large telemetry buffers or model loading

**Solutions**:
```python
# Reduce buffer size in config/qkd_system_parameters.py
TELEMETRY_WINDOW_SIZE = 25  # Change to 15

# Or clear buffers periodically
orchestrator.feature_extractor.clear_buffers()
```

---

### Issue: Warnings about "pgmpy not installed"

**Cause**: Optional packages for advanced features not installed

**Solution**: These are **optional** - system works with fallback modes:
```bash
# Install optional packages if you want full features
pip install pgmpy lifelines nonconformist torch torch-geometric

# Or ignore warnings - fallback modes work fine
```

---

## Dashboard Issues

### Issue: "streamlit: command not found"

**Cause**: Streamlit not installed or not in PATH

**Solution**:
```bash
pip install streamlit
streamlit run operations_dashboard/streamlit_app.py
```

---

### Issue: Dashboard shows "Failed to load models"

**Cause**: Models not trained yet

**Solution**:
```bash
# Train models first
python scripts/train_attribution_models.py

# Then launch dashboard
streamlit run operations_dashboard/streamlit_app.py
```

---

## Hardware Interface Issues

### Issue: "pysnmp not installed" warning

**Cause**: SNMP library not available

**Solution**: This is **optional** for real hardware:
```bash
# For real ID Quantique hardware, install:
pip install pysnmp

# For testing, use mock hardware (no installation needed)
from hardware_interface import MockQKDHardware
```

---

## Performance Issues

### Issue: Slow inference (<10 Hz)

**Cause**: Feature extraction or model prediction overhead

**Solutions**:
```python
# 1. Reduce feature window
TELEMETRY_WINDOW_SIZE = 15  # Instead of 25

# 2. Use faster model
# Edit scripts/train_attribution_models.py:
n_estimators=60  # Instead of 120

# 3. Disable advanced features if not needed
orchestrator = AdvancedQKDOrchestrator(
    enable_causal_inference=False,
    enable_survival_analysis=False
)
```

---

## Data Issues

### Issue: "data_splits.json not found"

**Cause**: Data manifest not generated

**Solution**:
```bash
# Generate data splits
python -c "from validation_framework.data_split_manifest import create_and_export_data_splits_manifest; create_and_export_data_splits_manifest()"
```

---

## Python Version Issues

### Issue: "Python 3.14 installed but need 3.10"

**Cause**: Multiple Python versions

**Solution**:
```bash
# Windows: Use specific Python version
C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac: Use specific Python version
python3.10 -m venv .venv
source .venv/bin/activate

# Verify version
python --version  # Should show Python 3.10.x
```

---

## Still Having Issues?

### Run the validation script:
```bash
python validate_installation.py
```

This will check:
- Python version
- Required packages
- Project structure
- Trained models
- Module imports

### Check logs:
```bash
# View audit log
sqlite3 vector_q_audit.db "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 10;"

# Check model files
ls -lh models/
```

### Get system info:
```python
import sys
import numpy as np
import pandas as pd
import sklearn
import lightgbm

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Import errors | Run from project root or use `python -m pytest` |
| Training fails | Install missing packages: `pip install -r requirements.txt` |
| Tests fail | Use updated test files with correct parameter names |
| No models | Run: `python scripts/train_attribution_models.py` |
| Warnings | Optional packages - ignore or install for full features |
| Slow performance | Reduce buffer size or use fewer estimators |
| Database locked | Close all processes, delete `.db-journal` file |
| Wrong Python | Use Python 3.10 explicitly in venv creation |

---

**Need more help?** Run `python validate_installation.py` for detailed diagnostics.
