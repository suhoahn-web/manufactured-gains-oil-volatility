# Expanding-Window Robustness (pooled QLIKE, vol ladder)

Pooled test weeks: 487

## Mean QLIKE

```
              h1      h2      h4      h8     h12
model                                           
HAR       0.1206  0.0939  0.0974  0.1211  0.1418
M1        0.1207  0.0953  0.0998  0.1253  0.1464
M2        0.1223  0.1012  0.1117  0.1459  0.1630
M3        0.1194  0.0953  0.1011  0.1256  0.1364
M4arch    0.1193  0.0957  0.1002  0.1234  0.1409
M4        0.1190  0.0953  0.0999  0.1237  0.1415
M5_lb0p1  0.1184  0.0955  0.0980  0.1208  0.1380
M5_lb0p5  0.1152  0.0921  0.0914  0.1077  0.1224
```

## DM tests on pooled QLIKE differentials (negative = first better)

```
  h1: M2vsM1 +0.73(p=0.464) | M3vsM1 -0.69(p=0.492) | M4vsM2 -2.41(p=0.016) | M4vsM3 -0.23(p=0.815) | M4vsM1 -0.94(p=0.350) | M4vsHAR -0.59(p=0.553) | M5_lb0p1vsM4 -0.42(p=0.674) | M5_lb0p1vsHAR -0.59(p=0.554) | M5_lb0p1vsM1 -1.07(p=0.287) | M5_lb0p5vsM4 -1.03(p=0.304) | M5_lb0p5vsHAR -0.97(p=0.335)
  h2: M2vsM1 +1.53(p=0.127) | M3vsM1 -0.01(p=0.991) | M4vsM2 -2.85(p=0.005) | M4vsM3 -0.00(p=1.000) | M4vsM1 -0.01(p=0.991) | M4vsHAR +0.36(p=0.717) | M5_lb0p1vsM4 +0.11(p=0.915) | M5_lb0p1vsHAR +0.30(p=0.764) | M5_lb0p1vsM1 +0.05(p=0.963) | M5_lb0p5vsM4 -0.55(p=0.584) | M5_lb0p5vsHAR -0.20(p=0.842)
  h4: M2vsM1 +1.84(p=0.066) | M3vsM1 +0.34(p=0.732) | M4vsM2 -2.83(p=0.005) | M4vsM3 -0.28(p=0.780) | M4vsM1 +0.01(p=0.993) | M4vsHAR +0.50(p=0.614) | M5_lb0p1vsM4 -0.55(p=0.581) | M5_lb0p1vsHAR +0.09(p=0.932) | M5_lb0p1vsM1 -0.35(p=0.726) | M5_lb0p5vsM4 -0.72(p=0.474) | M5_lb0p5vsHAR -0.39(p=0.697)
  h8: M2vsM1 +1.99(p=0.047) | M3vsM1 +0.03(p=0.976) | M4vsM2 -3.18(p=0.002) | M4vsM3 -0.25(p=0.806) | M4vsM1 -0.23(p=0.819) | M4vsHAR +0.30(p=0.761) | M5_lb0p1vsM4 -0.55(p=0.584) | M5_lb0p1vsHAR -0.02(p=0.984) | M5_lb0p1vsM1 -0.50(p=0.617) | M5_lb0p5vsM4 -0.75(p=0.453) | M5_lb0p5vsHAR -0.46(p=0.642)
 h12: M2vsM1 +1.43(p=0.153) | M3vsM1 -0.65(p=0.519) | M4vsM2 -2.58(p=0.010) | M4vsM3 +0.47(p=0.638) | M4vsM1 -0.62(p=0.538) | M4vsHAR -0.03(p=0.974) | M5_lb0p1vsM4 -0.61(p=0.543) | M5_lb0p1vsHAR -0.26(p=0.796) | M5_lb0p1vsM1 -0.72(p=0.470) | M5_lb0p5vsM4 -0.79(p=0.431) | M5_lb0p5vsHAR -0.59(p=0.555)
```
