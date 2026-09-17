"""
VECTOR-Q: 3-Minute Live Incident Demonstration
Fault -> Detection -> Diagnosis -> Forecast -> Optimization -> Recovery

Usage:
  python scripts/run_incident_demo.py

Author: Senior Quantum Systems & Applied ML Engineering Team
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_killer_demonstration import run_killer_demo

if __name__ == "__main__":
    run_killer_demo()
