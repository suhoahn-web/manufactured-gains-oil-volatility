# Balance Audit Summary

Common balance sample: 1991-02-08 -> 2026-07-31 (1852 weeks)

## Correlation: observed stock change vs reconstructed net flow (no adjustment)
- total stocks (incl SPR):      r = 0.7180
- commercial stocks (ex-SPR):   r = 0.6557
- typical |weekly stock change| (total): 3,664 kb/week

## Implied adjustment (kb/d), TOTAL-stock definition
- mean 263.0, std 544.8, median 212.3
- p05 -539.5, p95 1227.6, mean|.| 445.5

## Implied adjustment (kb/d), COMMERCIAL-stock definition
- mean 284.7, std 592.3, median 212.3
- p05 -544.8, p95 1384.9, mean|.| 477.3

## Methodology-era comparison (total-stock implied adjustment, kb/d)
- pre 2016-07 (exports monthly-based):  mean 189.4, std 495.3, n=1325
- post 2016-07 (exports weekly Census): mean 448.0, std 615.6, n=527

## Notes
- Commercial-definition residual absorbs SPR flows; total-definition residual
  corresponds to EIA's official 'crude oil adjustment' (unaccounted-for crude).
- EIA states components are independently estimated, so a nonzero adjustment
  is expected -> justifies SOFT balance constraint.