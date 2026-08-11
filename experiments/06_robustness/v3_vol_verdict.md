# v3 Volatility Verdict (QLIKE, seed-ensemble preds)

Common test weeks: 305

## Mean QLIKE by model/horizon

```
           h1      h2      h4      h8     h12
model                                        
M1     0.3101  0.1670  0.1635  0.1858  0.2006
M4     0.3283  0.1717  0.1669  0.1732  0.1804
M5s    0.3193  0.1650  0.1637  0.1771  0.1840
M5x    0.3048  0.1543  0.1444  0.1501  0.1507
HAR    0.3145  0.1565  0.1591  0.1943  0.2049
```

## DM tests on QLIKE differentials (negative DM = first better)

```
  h1: M4vsM1 +3.54(p=0.000) | M4vsHAR +0.84(p=0.403) | M5svsM4 -2.23(p=0.026) | M5svsHAR +0.28(p=0.777) | M5xvsM4 -3.57(p=0.000) | M5xvsHAR -0.61(p=0.543)
  h2: M4vsM1 +0.76(p=0.446) | M4vsHAR +0.89(p=0.372) | M5svsM4 -1.15(p=0.251) | M5svsHAR +0.49(p=0.626) | M5xvsM4 -1.76(p=0.079) | M5xvsHAR -0.14(p=0.885)
  h4: M4vsM1 +0.41(p=0.684) | M4vsHAR +0.29(p=0.773) | M5svsM4 -0.45(p=0.656) | M5svsHAR +0.17(p=0.866) | M5xvsM4 -1.29(p=0.198) | M5xvsHAR -0.52(p=0.601)
  h8: M4vsM1 -1.58(p=0.115) | M4vsHAR -0.69(p=0.493) | M5svsM4 +0.72(p=0.469) | M5svsHAR -0.51(p=0.609) | M5xvsM4 -0.80(p=0.426) | M5xvsHAR -0.84(p=0.403)
 h12: M4vsM1 -1.51(p=0.132) | M4vsHAR -0.85(p=0.394) | M5svsM4 +1.19(p=0.234) | M5svsHAR -0.66(p=0.513) | M5xvsM4 -0.80(p=0.423) | M5xvsHAR -0.89(p=0.376)
```
