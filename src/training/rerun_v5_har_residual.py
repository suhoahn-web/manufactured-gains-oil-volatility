"""v5: HAR-residual ladder + full re-evaluation."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

LADDER = [
    ["--models", "M0", "M1", "M2", "M3", "--tag", "_v5"],
    ["--models", "M4", "M5", "--lambda-f", "0.1", "--lambda-b", "0.01",
     "--tag", "_v5"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.1",
     "--tag", "_v5lb0p1"],
    ["--models", "M5", "--lambda-f", "0.1", "--lambda-b", "0.5",
     "--tag", "_v5lb0p5"],
]


def run(cmd, label):
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        sys.exit(1)


for args in LADDER:
    run([sys.executable, str(HERE / "run_ladder.py"), "--seeds", "5",
         "--target", "vol", "--har-residual"] + args, " ".join(args))

run([sys.executable, str(ROOT / "experiments" / "06_robustness" / "expanding_window.py")],
    "expanding window (HAR-residual, 5 seeds)")
run([sys.executable, str(ROOT / "experiments" / "06_robustness" / "mcs_test.py")], "MCS")
run([sys.executable, str(ROOT / "experiments" / "06_robustness" / "subsample_robustness.py")],
    "subsample robustness")
run([sys.executable, str(ROOT / "experiments" / "07_final" / "stress_and_econ_value.py")],
    "inventory regimes + economic value")
print("\nV5 COMPLETE", flush=True)
