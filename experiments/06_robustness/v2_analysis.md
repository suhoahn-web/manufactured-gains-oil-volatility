# v2 Final Analysis

## DM tests (squared error, seed-ensemble; negative DM = first model better)

```
  h1: M5vsM4 DM=-0.253 p=0.800 | M4vsM1 DM=+1.165 p=0.245
  h2: M5vsM4 DM=+0.170 p=0.865 | M4vsM1 DM=+0.673 p=0.501
  h4: M5vsM4 DM=-0.477 p=0.633 | M4vsM1 DM=-0.794 p=0.428
  h8: M5vsM4 DM=-0.783 p=0.434 | M4vsM1 DM=+0.065 p=0.948
 h12: M5vsM4 DM=+0.098 p=0.922 | M4vsM1 DM=+0.325 p=0.745
```

## Return MAE, test subsample 2021-01 onward (ex-COVID)

```
           h1      h2      h4      h8     h12
model                                        
M1_v2  0.0318  0.0453  0.0670  0.1003  0.1184
M2_v2  0.0367  0.0463  0.0713  0.1075  0.1273
M3_v2  0.0322  0.0524  0.0702  0.1167  0.1203
M4_v2  0.0323  0.0458  0.0679  0.1019  0.1182
M5_v2  0.0323  0.0460  0.0681  0.1021  0.1190
```

## Return MAE by inventory regime, h1

```
       normal   build    draw
model                        
M1_v2  0.0357  0.1014  0.0317
M2_v2  0.0399  0.1073  0.0412
M3_v2  0.0363  0.1024  0.0333
M4_v2  0.0361  0.1023  0.0331
M5_v2  0.0361  0.1023  0.0331
```

## Return MAE by inventory regime, h12

```
       normal   build    draw
model                        
M1_v2  0.1647  0.2085  0.1072
M2_v2  0.1766  0.2294  0.1178
M3_v2  0.1701  0.2249  0.1074
M4_v2  0.1658  0.2109  0.1063
M5_v2  0.1664  0.2108  0.1075
```
