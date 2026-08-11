"""v3 volatility-pivot driver: full ladder on log-RV targets."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ["--models", "M0", "M1", "M2", "M3", "--tag", "_v3"],
    ["--models", "M4", "M5", "--lambda-f", "0.1", "--lambda-b", "0.01",
     "--tag", "_v3"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.1",
     "--tag", "_v3lb0p1"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.5",
     "--tag", "_v3lb0p5"],
]

for args in RUNS:
    print(f"=== {' '.join(args)} ===", flush=True)
    r = subprocess.run(
        [sys.executable, str(HERE / "run_ladder.py"), "--seeds", "5",
         "--target", "vol"] + args,
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(1)
print("V3 VOL LADDER DONE", flush=True)
