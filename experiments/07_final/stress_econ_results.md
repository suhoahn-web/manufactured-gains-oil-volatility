# Inventory-Regime + Economic-Value Results (pooled EW weeks)

## Mean QLIKE by target-window inventory regime, h1
(train q10/q90 of the 1-week average weekly stock change: -4,630 / 4,356 kb)

```
          normal   build    draw  n_build  n_draw
model                                            
HAR       0.1304  0.1341  0.0792       89      95
M1        0.1336  0.1333  0.0704       89      95
M2        0.1314  0.1381  0.0812       89      95
M3        0.1283  0.1320  0.0818       89      95
M4arch    0.1302  0.1285  0.0785       89      95
M4        0.1299  0.1282  0.0784       89      95
M5_lb0p1  0.1292  0.1262  0.0795       89      95
M5_lb0p5  0.1234  0.1237  0.0833       89      95
```

## Mean QLIKE by target-window inventory regime, h4
(train q10/q90 of the 4-week average weekly stock change: -2,979 / 3,196 kb)

```
          normal   build    draw  n_build  n_draw
model                                            
HAR       0.0799  0.2414  0.0498       73      97
M1        0.0850  0.2358  0.0494       73      97
M2        0.0945  0.2414  0.0748       73      97
M3        0.0846  0.2357  0.0571       73      97
M4arch    0.0848  0.2277  0.0585       73      97
M4        0.0846  0.2272  0.0581       73      97
M5_lb0p1  0.0840  0.2153  0.0598       73      97
M5_lb0p5  0.0794  0.1851  0.0634       73      97
```

## Mean QLIKE by target-window inventory regime, h12
(train q10/q90 of the 12-week average weekly stock change: -2,054 / 2,352 kb)

```
          normal   build    draw  n_build  n_draw
model                                            
HAR       0.0521  0.8379  0.0477       54      98
M1        0.0590  0.8180  0.0486       54      98
M2        0.0768  0.8147  0.0795       54      98
M3        0.0638  0.6940  0.0580       54      98
M4arch    0.0620  0.7674  0.0554       54      98
M4        0.0624  0.7705  0.0549       54      98
M5_lb0p1  0.0637  0.7264  0.0545       54      98
M5_lb0p5  0.0651  0.5646  0.0579       54      98
```

## Economic value: volatility targeting, h=1, single-split test 2020+

(sigma_target = train median weekly RV = 0.0366; position w = min(sigma_target/forecast, 3); no transaction costs)

```
          ann_ret  ann_vol  sharpe
model                             
M1         0.0318   0.3225  0.0986
M4         0.0240   0.3276  0.0733
M5_lb0p1   0.0261   0.3247  0.0805
M5_lb0p5   0.0257   0.3204  0.0803
buy&hold   0.0859   1.0330  0.0831
```
