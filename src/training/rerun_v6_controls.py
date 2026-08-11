"""v6: address the review's structural objections.

Adds the architecture control (dual-branch, lambda_F = 0), re-runs the external
baselines HAR-anchored with the ARIMA look-ahead fixed, and re-runs the whole
pooled evaluation so that every comparison is architecture-matched and every
promised metric is recorded.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

LADDER = [
    # architecture control: dual branch, no auxiliary task
    ["--models", "M4", "--lambda-f", "0", "--lambda-b", "0", "--tag", "_v6arch"],
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

run([sys.executable, str(ROOT / "experiments" / "02_baselines"
                        / "external_baselines_vol.py")],
    "external baselines (HAR-anchored, ARIMA fixed)")
run([sys.executable, str(ROOT / "experiments" / "06_robustness"
                        / "expanding_window.py")],
    "expanding window (7 configs incl. architecture control, 5 seeds)")
run([sys.executable, str(ROOT / "experiments" / "06_robustness" / "mcs_test.py")],
    "MCS")
run([sys.executable, str(ROOT / "experiments" / "06_robustness"
                        / "subsample_robustness.py")], "subsample robustness")
run([sys.executable, str(ROOT / "experiments" / "07_final"
                        / "stress_and_econ_value.py")],
    "inventory regimes + economic value")
print("\nV6 COMPLETE", flush=True)
