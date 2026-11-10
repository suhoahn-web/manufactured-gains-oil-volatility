"""Stage 2 of the v4 re-run: external baselines, then the upgraded
expanding-window evaluation (M1-M5, 5 seeds), MCS, and stress/economic value.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    (ROOT / "experiments" / "02_baselines" / "external_baselines_vol.py",
     "external baselines (ARIMA/XGBoost/LSTM/TCN)"),
    (ROOT / "experiments" / "06_robustness" / "expanding_window.py",
     "expanding window (M1-M5, 5 seeds)"),
    (ROOT / "experiments" / "06_robustness" / "mcs_test.py", "MCS"),
    (ROOT / "experiments" / "07_final" / "stress_and_econ_value.py",
     "inventory regimes + economic value"),
]

for script, label in STAGES:
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable, str(script)],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(1)
print("\nSTAGE 2 COMPLETE", flush=True)
