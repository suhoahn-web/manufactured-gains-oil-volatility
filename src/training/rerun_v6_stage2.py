"""v6 stage 2: pooled evaluation only (ladder + external baselines already done).

Resumes after the interrupted run: expanding window with the architecture
control, then MCS, subsample robustness, and inventory/economic analysis.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    (ROOT / "experiments" / "06_robustness" / "expanding_window.py",
     "expanding window (7 configs, 5 seeds)"),
    (ROOT / "experiments" / "06_robustness" / "mcs_test.py", "MCS"),
    (ROOT / "experiments" / "06_robustness" / "subsample_robustness.py",
     "subsample robustness"),
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
print("\nV6 STAGE 2 COMPLETE", flush=True)
