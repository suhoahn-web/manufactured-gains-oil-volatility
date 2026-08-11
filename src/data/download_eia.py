"""Download and cache EIA weekly petroleum series (raw .xls) and parse to CSV.

Sources: EIA petroleum navigator bulk history files
  https://www.eia.gov/dnav/pet/hist_xls/<SERIES>w.xls

Raw files are cached in data/raw/eia/ untouched; parsed tidy CSVs go to
data/interim/. A metadata JSON records series id, name, unit, URL, download
timestamp, and date coverage.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eia"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
META_DIR = PROJECT_ROOT / "data" / "metadata"

BASE_URL = "https://www.eia.gov/dnav/pet/hist_xls/{sid}w.xls"

# series id -> (short name, unit)
SERIES = {
    "WCRFPUS2": ("production", "thousand barrels per day"),
    "WCRIMUS2": ("imports", "thousand barrels per day"),
    "WCREXUS2": ("exports", "thousand barrels per day"),
    "WCRRIUS2": ("refinery_inputs", "thousand barrels per day"),
    "WCESTUS1": ("stocks_commercial_ex_spr", "thousand barrels"),
    "WCRSTUS1": ("stocks_total_incl_spr", "thousand barrels"),
    "WCSSTUS1": ("stocks_spr", "thousand barrels"),
    "WPULEUS3": ("refinery_utilization", "percent of operable capacity"),
    "W_EPC0_SAX_YCUOK_MBBL": ("stocks_cushing", "thousand barrels"),
    "RWTC": ("wti_spot", "dollars per barrel"),
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download; academic use)"}


def download(sid: str, dest: Path) -> None:
    url = BASE_URL.format(sid=sid)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    if b"<html" in resp.content[:200].lower():
        raise RuntimeError(f"{sid}: got HTML instead of xls from {url}")
    dest.write_bytes(resp.content)


def parse(sid: str, path: Path) -> pd.DataFrame:
    """EIA hist xls: sheet 'Data 1', row 0 sourcekey-ish, header on row 2."""
    df = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "value"] + list(df.columns[2:])
    df = df[["date", "value"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}
    failures = []
    for sid, (name, unit) in SERIES.items():
        raw_path = RAW_DIR / f"{sid}w.xls"
        try:
            download(sid, raw_path)
            df = parse(sid, raw_path)
            out = INTERIM_DIR / f"{name}.csv"
            df.to_csv(out, index=False)
            metadata[sid] = {
                "name": name,
                "unit": unit,
                "url": BASE_URL.format(sid=sid),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "n_obs": int(len(df)),
                "first_date": str(df["date"].min().date()),
                "last_date": str(df["date"].max().date()),
            }
            print(f"OK   {sid:<24} {name:<26} {len(df):>5} obs  "
                  f"{df['date'].min().date()} -> {df['date'].max().date()}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(sid)
            print(f"FAIL {sid:<24} {exc}")
        time.sleep(1.0)  # be polite to EIA servers

    (META_DIR / "eia_series_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\n{len(metadata)}/{len(SERIES)} series downloaded; "
          f"metadata written to {META_DIR / 'eia_series_metadata.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
