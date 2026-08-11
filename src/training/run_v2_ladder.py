"""v2 iteration driver: change-based aux targets, horizons 1/2/4/8/12.

lambda_F=0.1 (validation-selected in v1 probe), lambda_B default 0.01 with a
mini-sweep {0.01, 0.1, 0.5} for validation-based selection.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ["--models", "M0", "M1", "M2", "M3", "--tag", "_v2"],
    ["--models", "M4", "M5", "--lambda-f", "0.1", "--lambda-b", "0.01",
     "--tag", "_v2"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.1",
     "--tag", "_v2lb0p1"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.5",
     "--tag", "_v2lb0p5"],
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
print("V2 LADDER DONE", flush=True)
