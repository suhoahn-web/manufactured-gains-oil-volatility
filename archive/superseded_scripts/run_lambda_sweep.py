"""Lambda_B sweep for M5 (design doc section 12), sequential background run.

Runs M5 at lambda_B in {0.01, 0.05, 0.1, 0.5, 1.0} (lambda_B=0 equals M4),
5 seeds each, writing tagged results with validation metrics so lambda is
selected on validation only.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for lam in ("0.01", "0.05", "0.1", "0.5", "1.0"):
    tag = f"_lb{lam.replace('.', 'p')}"
    print(f"=== lambda_B = {lam} ===", flush=True)
    r = subprocess.run(
        [sys.executable, str(HERE / "run_ladder.py"),
         "--models", "M5", "--seeds", "5",
         "--lambda-f", "0.3", "--lambda-b", lam, "--tag", tag],
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(1)
print("SWEEP DONE", flush=True)
