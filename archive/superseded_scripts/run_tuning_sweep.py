"""Validation-driven hyperparameter probe around the default config.

One-factor-at-a-time from defaults (lookback 12, lambda_F 0.3, lambda_B 0.01
for M5): vary lookback {8, 26} for M1/M4/M5, vary lambda_F {0.1, 1.0} for
M4/M5. Selection will use validation metrics only.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    (["--models", "M1", "M4", "M5", "--lookback", "8",
      "--lambda-f", "0.3", "--lambda-b", "0.01", "--tag", "_L8"]),
    (["--models", "M1", "M4", "M5", "--lookback", "26",
      "--lambda-f", "0.3", "--lambda-b", "0.01", "--tag", "_L26"]),
    (["--models", "M4", "M5", "--lookback", "12",
      "--lambda-f", "0.1", "--lambda-b", "0.01", "--tag", "_lf0p1"]),
    (["--models", "M4", "M5", "--lookback", "12",
      "--lambda-f", "1.0", "--lambda-b", "0.01", "--tag", "_lf1p0"]),
]

for args in RUNS:
    print(f"=== {' '.join(args)} ===", flush=True)
    r = subprocess.run(
        [sys.executable, str(HERE / "run_ladder.py"), "--seeds", "5"] + args,
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(1)
print("TUNING SWEEP DONE", flush=True)
