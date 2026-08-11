# Forecast combination: HAR + physical-state models (single split)

## Test QLIKE (2020-2026)

```
                                 h1      h2      h4      h8     h12
model                                                              
HAR+M4                       0.1445  0.1210  0.1323  0.1624  0.1716
HAR+M4 (val-weighted)        0.1445  0.1210  0.1323  0.1624  0.1716
HAR+M5_lb0p1                 0.1425  0.1199  0.1312  0.1628  0.1716
HAR+M5_lb0p1 (val-weighted)  0.1425  0.1199  0.1312  0.1628  0.1716
HAR+M5_lb0p5                 0.1343  0.1119  0.1205  0.1470  0.1553
HAR+M5_lb0p5 (val-weighted)  0.1343  0.1119  0.1205  0.1470  0.1553
HAR                          0.1332  0.1129  0.1274  0.1692  0.1822
M4                           0.1692  0.1399  0.1479  0.1627  0.1681
M5_lb0p1                     0.1638  0.1370  0.1446  0.1633  0.1680
M5_lb0p5                     0.1454  0.1195  0.1227  0.1366  0.1405
```

Validation-estimated weights on the NN forecast:
```
                             w_h1  w_h2  w_h4  w_h8  w_h12
model                                                     
HAR+M4 (val-weighted)         0.5   0.5   0.5   0.5    0.5
HAR+M5_lb0p1 (val-weighted)   0.5   0.5   0.5   0.5    0.5
HAR+M5_lb0p5 (val-weighted)   0.5   0.5   0.5   0.5    0.5
```

## DM tests for HAR+M5_lb0p1 (equal weight)

```
  h1: comb vs HAR +2.06(p=0.040) | comb vs M5_lb0p1 -3.56(p=0.000)
  h2: comb vs HAR +1.19(p=0.234) | comb vs M5_lb0p1 -2.22(p=0.027)
  h4: comb vs HAR +0.41(p=0.681) | comb vs M5_lb0p1 -1.24(p=0.215)
  h8: comb vs HAR -0.54(p=0.587) | comb vs M5_lb0p1 -0.05(p=0.964)
 h12: comb vs HAR -0.87(p=0.385) | comb vs M5_lb0p1 +0.34(p=0.733)
```
