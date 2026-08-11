# Lambda selection + Inventory Stress Test

## Validation mean return-MAE by lambda_B (lower = better)

```
model
M5_lb0p01    0.060521
M5_lb0p05    0.060863
M5_lb0p1     0.061259
M5_lb0p5     0.067337
M5_lb1p0     0.086986
```

**Selected lambda_B* = 0.01** (validation only; test untouched)

## Stress thresholds (train quantiles of weekly commercial stock change, kb)

```
q10_90: draw < -4,630, build > 4,314
q05_95: draw < -6,338, build > 5,962
q15_85: draw < -3,781, build > 3,679
```

## Return MAE by period, horizon h1

```
period        all  normal   build    draw
model                                    
M1         0.0468  0.0359  0.1011  0.0336
M2         0.0524  0.0402  0.1076  0.0427
M3         0.0481  0.0368  0.1025  0.0362
M4         0.0482  0.0369  0.1031  0.0357
M5_lb0p01  0.0484  0.0370  0.1035  0.0362
```

### Directional accuracy by period, horizon h1

```
period        all  normal   build    draw
model                                    
M1         0.5664  0.5417  0.5593  0.6562
M2         0.4366  0.4583  0.4237  0.3750
M3         0.4543  0.4630  0.4746  0.4062
M4         0.4808  0.4769  0.5085  0.4688
M5_lb0p01  0.4779  0.4815  0.4915  0.4531
```

## Return MAE by period, horizon h2

```
period        all  normal   build    draw
model                                    
M1         0.0673  0.0527  0.1334  0.0555
M2         0.0699  0.0544  0.1367  0.0608
M3         0.0747  0.0592  0.1403  0.0667
M4         0.0714  0.0562  0.1372  0.0622
M5_lb0p01  0.0716  0.0558  0.1378  0.0638
```

### Directional accuracy by period, horizon h2

```
period        all  normal   build    draw
model                                    
M1         0.5782  0.5509  0.5932  0.6562
M2         0.4720  0.4907  0.4746  0.4062
M3         0.4720  0.4907  0.4746  0.4062
M4         0.5074  0.5185  0.5085  0.4688
M5_lb0p01  0.4926  0.5093  0.4915  0.4375
```

## Return MAE by period, horizon h4

```
period        all  normal   build    draw
model                                    
M1         0.0965  0.0815  0.1636  0.0851
M2         0.1024  0.0873  0.1647  0.0959
M3         0.1005  0.0850  0.1645  0.0938
M4         0.1035  0.0884  0.1644  0.0983
M5_lb0p01  0.1031  0.0879  0.1638  0.0982
```

### Directional accuracy by period, horizon h4

```
period        all  normal   build    draw
model                                    
M1         0.5457  0.5278  0.5424  0.6094
M2         0.4926  0.4907  0.5593  0.4375
M3         0.4867  0.4907  0.5424  0.4219
M4         0.5221  0.5185  0.5932  0.4688
M5_lb0p01  0.5133  0.5185  0.5763  0.4375
```

## Diebold-Mariano tests (squared-error loss, seed-ensemble preds)

```
h1: M5 vs M4  DM=+0.178 p=0.859 (negative = M5 better) | M5 vs M1  DM=+0.332 p=0.740
h2: M5 vs M4  DM=-0.184 p=0.854 (negative = M5 better) | M5 vs M1  DM=+0.563 p=0.574
h4: M5 vs M4  DM=-0.115 p=0.908 (negative = M5 better) | M5 vs M1  DM=+1.114 p=0.266
```
