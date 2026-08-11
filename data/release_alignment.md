# Release Alignment (design doc section 13)

## Timing facts

- EIA Weekly Petroleum Status Report (WPSR) covers a week ending **Friday** (week `t`).
- It is published the **following Wednesday** (i.e., during week `t+1`), 10:30 a.m. ET
  (Thursday if a federal holiday intervenes).
- EIA weekly WTI spot (RWTC weekly) is also a week-ending-Friday series.
- Therefore the physical data for week `t` were NOT known during week `t` -> using them to
  predict returns realized during week `t+1`-before-Wednesday would still be borderline.
  We adopt the conservative alignment below.

## Conservative alignment adopted (primary)

Let week `t` = the latest physical week available in the WPSR at the forecast origin.

- **Forecast origin** = Friday of week `t+1` (two days after the Wednesday release, so the
  release is fully public and the origin price is observed after it).
- **Information set at origin**: physical variables up to week `t`; prices up to Friday `t+1`.
- **Targets** (cumulative log returns from the origin price):
  - `h=1`: `log P_{t+2} - log P_{t+1}` — first fully future week (~1-week-ahead)
  - `h=2`: `log P_{t+3} - log P_{t+1}` (~2-week-ahead)
  - `h=4`: `log P_{t+5} - log P_{t+1}` (~4-week-ahead)

## Implementation mapping (`src/data/build_dataset.py`)

Each dataset row is indexed by the physical week-ending date `t`:

| column group          | uses data up to | rationale                          |
|-----------------------|-----------------|------------------------------------|
| physical features     | week `t`        | released Wednesday of week `t+1`   |
| price features        | Friday `t+1`    | market data, observed in real time |
| target `y_h{1,2,4}`   | weeks `t+2..t+5`| strictly after origin              |

Rows are formed only where all weekly dates align on the regular Friday grid; weeks where the
week-ending date deviates (holiday shifts) keep their EIA-reported dates, matched by exact
calendar merge on the common Friday grid, and gaps are left as NaN (never forward-filled across
release boundaries).

## Automated leakage tests (`src/data/test_leakage.py`)

1. For every row: max physical date used <= t, max price date used <= t+1's Friday.
2. For every row: min target date >= t+2's Friday.
3. Shuffling test target columns must not change any feature column (no shared memory).
4. Scalers are fit on train split only (checked in training code).
