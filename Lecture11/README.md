STAT406 - Lecture 11 notes
================
Matias Salibian-Barrera
2019-10-09

#### LICENSE

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

Lecture slides are [here](STAT406-18-lecture-11.pdf).

Pruning regression trees with `rpart`
-------------------------------------

***Important note**: As discussed in class, the K-fold CV methodology implemented in the package `rpart` seems to consider a sequence of trees (or, equivalently, of complexity parameters) based on the full training set. For more details refer to the corresponding documentation: pages 12 and ff of the package vignette, which can be accessed from `R` using the command `vignette('longintro', package='rpart')`. For an alternative implementation of CV-based pruning, please see also the Section **"Pruning regression trees with `tree`"** below.*

The stopping criteria generally used when fitting regression trees do not take into account explicitly the complexity of the tree. Hence, we may end up with either an overfitting tree, or a very simple one, which typically results in a decline in the quality of the corresponding predictions. As discussed in class, one solution is to purposedly grow / train a very large overfitting tree, and then prune it. One can also estimate the corresponding MSPE of each tree in the prunning sequence and choose an optimal one. The function `rpart` implements this approach, and we illustrate it below.

We force `rpart` to build a very large tree via the arguments of the function `rpart.control`. At the same time, to obtain a good picture of the evolution of MSPE for different subtrees, we set the smallest complexity parameter to be considered by the cross-validation experiment to a very low value (here we use `1e-8`).

``` r
library(rpart)
data(Boston, package='MASS')
# split data into a training and
# a test set
set.seed(123456) 
n <- nrow(Boston)
ii <- sample(n, floor(n/4))
dat.te <- Boston[ ii, ]
dat.tr <- Boston[ -ii, ]

myc <- rpart.control(minsplit=3, cp=1e-8, xval=10)
set.seed(123)
bos.to <- rpart(medv ~ ., data=dat.tr, method='anova',
                control=myc)
plot(bos.to, compress=TRUE) # type='proportional')
```

![](README_files/figure-markdown_github/prune-1.png)

Not surprisingly, the predictions of this large tree are not very good:

``` r
# predictions are poor, unsurprisingly
pr.to <- predict(bos.to, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.to)^2) )
```

    ## [1] 19.36147

To prune we explore the *CP table* returned in the `rpart` object to find the value of the complexity parameter with optimal estimated prediction error. The estimated prediction error of each subtree (corresponding to each value of `CP`) is contained in the column `xerror`, and the associated standard deviation is in column `xstd`. We would like to find the value of `CP` that yields a corresponding pruned tree with smallest estimated prediction error. The function `printcp` shows the CP table corresponding to an `rpart` object:

``` r
printcp(bos.to)
```

    ## 
    ## Regression tree:
    ## rpart(formula = medv ~ ., data = dat.tr, method = "anova", control = myc)
    ## 
    ## Variables actually used in tree construction:
    ##  [1] age     black   chas    crim    dis     indus   lstat   nox    
    ##  [9] ptratio rad     rm      tax     zn     
    ## 
    ## Root node error: 32946/380 = 86.7
    ## 
    ## n= 380 
    ## 
    ##             CP nsplit rel error  xerror     xstd
    ## 1   4.7150e-01      0 1.0000000 1.00707 0.094276
    ## 2   1.5701e-01      1 0.5285006 0.61687 0.066834
    ## 3   7.9798e-02      2 0.3714954 0.43020 0.056551
    ## 4   5.7540e-02      3 0.2916970 0.39297 0.056761
    ## 5   3.4802e-02      4 0.2341575 0.37417 0.058539
    ## 6   2.0424e-02      5 0.1993555 0.26854 0.044695
    ## 7   1.9408e-02      6 0.1789313 0.25933 0.042290
    ## 8   1.6414e-02      7 0.1595235 0.25874 0.042326
    ## 9   1.1118e-02      8 0.1431095 0.25272 0.042338
    ## 10  9.6449e-03      9 0.1319911 0.26244 0.049504
    ## 11  7.7292e-03     10 0.1223462 0.24910 0.048253
    ## 12  6.5545e-03     11 0.1146170 0.26541 0.055816
    ## 13  5.7344e-03     12 0.1080625 0.26188 0.055868
    ## 14  5.3955e-03     14 0.0965937 0.26314 0.055985
    ## 15  4.6018e-03     15 0.0911983 0.25807 0.054774
    ## 16  3.7390e-03     16 0.0865964 0.26002 0.054764
    ## 17  3.2170e-03     17 0.0828574 0.25896 0.054124
    ## 18  2.5445e-03     18 0.0796404 0.26272 0.054301
    ## 19  2.3205e-03     20 0.0745514 0.27475 0.054715
    ## 20  2.1485e-03     21 0.0722309 0.27232 0.054753
    ## 21  2.1316e-03     22 0.0700824 0.27543 0.054776
    ## 22  2.0477e-03     23 0.0679508 0.27582 0.054774
    ## 23  2.0283e-03     24 0.0659031 0.27356 0.054777
    ## 24  1.9878e-03     25 0.0638748 0.27561 0.054818
    ## 25  1.9781e-03     26 0.0618870 0.27598 0.054815
    ## 26  1.9686e-03     27 0.0599089 0.27707 0.054815
    ## 27  1.6400e-03     28 0.0579403 0.27869 0.054931
    ## 28  1.6357e-03     29 0.0563003 0.28049 0.054943
    ## 29  1.6212e-03     30 0.0546646 0.28043 0.054943
    ## 30  1.5386e-03     31 0.0530435 0.27775 0.054922
    ## 31  1.4205e-03     32 0.0515048 0.27753 0.054917
    ## 32  1.3390e-03     33 0.0500843 0.27690 0.054911
    ## 33  1.2731e-03     34 0.0487453 0.27704 0.055064
    ## 34  1.2294e-03     35 0.0474723 0.27640 0.055029
    ## 35  1.1693e-03     36 0.0462429 0.27821 0.055112
    ## 36  1.1587e-03     37 0.0450736 0.28100 0.055187
    ## 37  1.1306e-03     38 0.0439149 0.28096 0.055191
    ## 38  1.1235e-03     39 0.0427842 0.28092 0.055190
    ## 39  1.1117e-03     40 0.0416607 0.28216 0.055189
    ## 40  1.0183e-03     41 0.0405490 0.28095 0.055153
    ## 41  1.0016e-03     42 0.0395307 0.27614 0.055057
    ## 42  9.8001e-04     43 0.0385291 0.27542 0.055061
    ## 43  9.5959e-04     45 0.0365691 0.27544 0.055060
    ## 44  9.5612e-04     47 0.0346499 0.27553 0.055059
    ## 45  8.9091e-04     48 0.0336937 0.27556 0.055073
    ## 46  8.8600e-04     49 0.0328028 0.27855 0.055096
    ## 47  8.7103e-04     50 0.0319168 0.27858 0.055093
    ## 48  8.4075e-04     51 0.0310458 0.27801 0.055099
    ## 49  8.2287e-04     52 0.0302051 0.27787 0.055101
    ## 50  8.2159e-04     53 0.0293822 0.27231 0.054581
    ## 51  7.9802e-04     54 0.0285606 0.27227 0.054582
    ## 52  7.7379e-04     55 0.0277626 0.27259 0.054579
    ## 53  7.6674e-04     56 0.0269888 0.27296 0.054577
    ## 54  7.4051e-04     57 0.0262220 0.27443 0.054600
    ## 55  6.5174e-04     58 0.0254815 0.27595 0.054865
    ## 56  6.4506e-04     59 0.0248298 0.27740 0.054862
    ## 57  6.1748e-04     60 0.0241847 0.27719 0.054865
    ## 58  5.7918e-04     61 0.0235673 0.27834 0.054882
    ## 59  5.6590e-04     62 0.0229881 0.27932 0.054887
    ## 60  5.3958e-04     63 0.0224222 0.27950 0.054881
    ## 61  5.2778e-04     64 0.0218826 0.28190 0.055245
    ## 62  5.2595e-04     65 0.0213548 0.28194 0.055242
    ## 63  4.9608e-04     66 0.0208289 0.28482 0.055642
    ## 64  4.9581e-04     67 0.0203328 0.28585 0.055648
    ## 65  4.6477e-04     68 0.0198370 0.28834 0.055642
    ## 66  4.5562e-04     69 0.0193722 0.28888 0.055638
    ## 67  4.3208e-04     71 0.0184610 0.29007 0.055644
    ## 68  4.2934e-04     73 0.0175968 0.29015 0.055645
    ## 69  4.0512e-04     75 0.0167381 0.29001 0.055641
    ## 70  4.0437e-04     76 0.0163330 0.29072 0.055637
    ## 71  3.8959e-04     77 0.0159286 0.29103 0.055635
    ## 72  3.3745e-04     78 0.0155390 0.29148 0.055596
    ## 73  3.2839e-04     79 0.0152016 0.28724 0.055463
    ## 74  3.1358e-04     80 0.0148732 0.28740 0.055462
    ## 75  3.0960e-04     81 0.0145596 0.28814 0.055459
    ## 76  2.8639e-04     82 0.0142500 0.28687 0.055458
    ## 77  2.7607e-04     83 0.0139636 0.28725 0.055459
    ## 78  2.7189e-04     85 0.0134115 0.28647 0.055465
    ## 79  2.6958e-04     86 0.0131396 0.28645 0.055465
    ## 80  2.6552e-04     87 0.0128700 0.28641 0.055464
    ## 81  2.6115e-04     88 0.0126045 0.28588 0.055459
    ## 82  2.5749e-04     89 0.0123434 0.28580 0.055459
    ## 83  2.5578e-04     90 0.0120859 0.28494 0.055467
    ## 84  2.5257e-04     91 0.0118301 0.28419 0.055474
    ## 85  2.2556e-04     92 0.0115775 0.28395 0.055471
    ## 86  2.2386e-04     93 0.0113519 0.28294 0.055475
    ## 87  2.1854e-04     94 0.0111281 0.28332 0.055477
    ## 88  2.1012e-04     95 0.0109095 0.28319 0.055477
    ## 89  2.0946e-04     96 0.0106994 0.28424 0.055481
    ## 90  2.0488e-04     97 0.0104900 0.28369 0.055487
    ## 91  2.0296e-04     98 0.0102851 0.28385 0.055485
    ## 92  2.0035e-04     99 0.0100821 0.28413 0.055482
    ## 93  1.9446e-04    100 0.0098818 0.28394 0.055478
    ## 94  1.9166e-04    101 0.0096873 0.28383 0.055485
    ## 95  1.8824e-04    102 0.0094957 0.28383 0.055485
    ## 96  1.8713e-04    103 0.0093074 0.28413 0.055481
    ## 97  1.7808e-04    104 0.0091203 0.28435 0.055481
    ## 98  1.7610e-04    105 0.0089422 0.28428 0.055480
    ## 99  1.7325e-04    106 0.0087661 0.28411 0.055478
    ## 100 1.7018e-04    107 0.0085929 0.28410 0.055478
    ## 101 1.5789e-04    108 0.0084227 0.28405 0.055476
    ## 102 1.5735e-04    109 0.0082648 0.28422 0.055461
    ## 103 1.4751e-04    110 0.0081074 0.28416 0.055462
    ## 104 1.4632e-04    111 0.0079599 0.28468 0.055460
    ## 105 1.3986e-04    112 0.0078136 0.28415 0.055464
    ## 106 1.3925e-04    113 0.0076737 0.28389 0.055466
    ## 107 1.3479e-04    116 0.0072560 0.28430 0.055467
    ## 108 1.3357e-04    117 0.0071212 0.28430 0.055467
    ## 109 1.3245e-04    118 0.0069876 0.28431 0.055468
    ## 110 1.3171e-04    119 0.0068552 0.28419 0.055469
    ## 111 1.2728e-04    120 0.0067235 0.28431 0.055469
    ## 112 1.2691e-04    121 0.0065962 0.28447 0.055467
    ## 113 1.2493e-04    122 0.0064693 0.28464 0.055466
    ## 114 1.1699e-04    123 0.0063444 0.28527 0.055470
    ## 115 1.1655e-04    125 0.0061104 0.28484 0.055472
    ## 116 1.1542e-04    126 0.0059938 0.28484 0.055472
    ## 117 1.0244e-04    127 0.0058784 0.28480 0.055471
    ## 118 1.0244e-04    128 0.0057760 0.28425 0.055465
    ## 119 1.0205e-04    129 0.0056735 0.28425 0.055465
    ## 120 9.8401e-05    130 0.0055715 0.28462 0.055464
    ## 121 9.7938e-05    131 0.0054731 0.28458 0.055465
    ## 122 9.7938e-05    132 0.0053751 0.28458 0.055465
    ## 123 9.7128e-05    133 0.0052772 0.28458 0.055465
    ## 124 9.4118e-05    134 0.0051801 0.28462 0.055464
    ## 125 9.3663e-05    135 0.0050860 0.28462 0.055464
    ## 126 9.3243e-05    136 0.0049923 0.28462 0.055464
    ## 127 8.2635e-05    137 0.0048991 0.28481 0.055468
    ## 128 8.2635e-05    138 0.0048164 0.28584 0.055470
    ## 129 7.3547e-05    139 0.0047338 0.28636 0.055477
    ## 130 7.3049e-05    140 0.0046602 0.28600 0.055471
    ## 131 6.8395e-05    141 0.0045872 0.28630 0.055474
    ## 132 6.5562e-05    142 0.0045188 0.28633 0.055470
    ## 133 5.8916e-05    143 0.0044532 0.28611 0.055473
    ## 134 5.6726e-05    145 0.0043354 0.28561 0.055466
    ## 135 5.6471e-05    146 0.0042787 0.28556 0.055462
    ## 136 5.5090e-05    147 0.0042222 0.28556 0.055462
    ## 137 5.4263e-05    149 0.0041120 0.28548 0.055463
    ## 138 5.1296e-05    150 0.0040578 0.28532 0.055464
    ## 139 5.1296e-05    151 0.0040065 0.28531 0.055464
    ## 140 5.1053e-05    152 0.0039552 0.28531 0.055464
    ## 141 5.1003e-05    153 0.0039041 0.28531 0.055464
    ## 142 4.9576e-05    154 0.0038531 0.28516 0.055464
    ## 143 4.9308e-05    155 0.0038035 0.28518 0.055464
    ## 144 4.8615e-05    156 0.0037542 0.28518 0.055464
    ## 145 4.8615e-05    157 0.0037056 0.28520 0.055464
    ## 146 4.5354e-05    158 0.0036570 0.28522 0.055463
    ## 147 4.2544e-05    159 0.0036116 0.28532 0.055463
    ## 148 4.2519e-05    160 0.0035691 0.28541 0.055462
    ## 149 4.1488e-05    161 0.0035266 0.28540 0.055462
    ## 150 4.0759e-05    163 0.0034436 0.28540 0.055462
    ## 151 4.0675e-05    166 0.0033213 0.28576 0.055464
    ## 152 4.0141e-05    167 0.0032807 0.28587 0.055462
    ## 153 3.9661e-05    168 0.0032405 0.28587 0.055462
    ## 154 3.9133e-05    169 0.0032009 0.28584 0.055463
    ## 155 3.6878e-05    170 0.0031617 0.28583 0.055460
    ## 156 3.6524e-05    171 0.0031248 0.28583 0.055460
    ## 157 3.4197e-05    172 0.0030883 0.28586 0.055460
    ## 158 3.2895e-05    173 0.0030541 0.28561 0.055464
    ## 159 3.2781e-05    174 0.0030212 0.28561 0.055464
    ## 160 3.2438e-05    175 0.0029884 0.28561 0.055464
    ## 161 2.9503e-05    177 0.0029236 0.28594 0.055471
    ## 162 2.9381e-05    178 0.0028941 0.28573 0.055476
    ## 163 2.9381e-05    179 0.0028647 0.28573 0.055476
    ## 164 2.9139e-05    180 0.0028353 0.28575 0.055476
    ## 165 2.8420e-05    181 0.0028062 0.28566 0.055476
    ## 166 2.6761e-05    182 0.0027777 0.28567 0.055476
    ## 167 2.4484e-05    183 0.0027510 0.28550 0.055474
    ## 168 2.4282e-05    184 0.0027265 0.28485 0.055305
    ## 169 2.3311e-05    185 0.0027022 0.28507 0.055306
    ## 170 2.3083e-05    186 0.0026789 0.28508 0.055306
    ## 171 2.2309e-05    187 0.0026558 0.28508 0.055306
    ## 172 2.1930e-05    188 0.0026335 0.28531 0.055304
    ## 173 2.1409e-05    190 0.0025897 0.28521 0.055300
    ## 174 2.0325e-05    191 0.0025682 0.28531 0.055299
    ## 175 2.0235e-05    192 0.0025479 0.28502 0.055301
    ## 176 2.0235e-05    193 0.0025277 0.28508 0.055300
    ## 177 2.0235e-05    194 0.0025075 0.28508 0.055300
    ## 178 2.0235e-05    195 0.0024872 0.28508 0.055300
    ## 179 1.8439e-05    196 0.0024670 0.28507 0.055300
    ## 180 1.8262e-05    197 0.0024485 0.28511 0.055300
    ## 181 1.7099e-05    198 0.0024303 0.28501 0.055301
    ## 182 1.7099e-05    199 0.0024132 0.28518 0.055303
    ## 183 1.6390e-05    200 0.0023961 0.28518 0.055303
    ## 184 1.6390e-05    201 0.0023797 0.28526 0.055305
    ## 185 1.4620e-05    202 0.0023633 0.28548 0.055306
    ## 186 1.4620e-05    203 0.0023487 0.28557 0.055305
    ## 187 1.4610e-05    204 0.0023341 0.28557 0.055305
    ## 188 1.3380e-05    205 0.0023195 0.28551 0.055306
    ## 189 1.3380e-05    206 0.0023061 0.28551 0.055306
    ## 190 1.2950e-05    207 0.0022927 0.28561 0.055305
    ## 191 1.2950e-05    208 0.0022797 0.28562 0.055304
    ## 192 1.1382e-05    209 0.0022668 0.28552 0.055308
    ## 193 1.1382e-05    210 0.0022554 0.28562 0.055310
    ## 194 1.0927e-05    211 0.0022440 0.28583 0.055309
    ## 195 1.0118e-05    212 0.0022331 0.28589 0.055309
    ## 196 1.0118e-05    213 0.0022230 0.28589 0.055308
    ## 197 9.9152e-06    214 0.0022129 0.28589 0.055308
    ## 198 9.9152e-06    215 0.0022029 0.28589 0.055308
    ## 199 9.4852e-06    216 0.0021930 0.28589 0.055308
    ## 200 9.1817e-06    217 0.0021835 0.28589 0.055309
    ## 201 8.0283e-06    218 0.0021744 0.28577 0.055309
    ## 202 8.0283e-06    219 0.0021663 0.28584 0.055311
    ## 203 7.2846e-06    220 0.0021583 0.28584 0.055311
    ## 204 7.2846e-06    222 0.0021437 0.28584 0.055311
    ## 205 6.1211e-06    223 0.0021365 0.28552 0.055306
    ## 206 5.8277e-06    225 0.0021242 0.28573 0.055311
    ## 207 5.0588e-06    226 0.0021184 0.28580 0.055310
    ## 208 4.2747e-06    227 0.0021133 0.28595 0.055309
    ## 209 4.0976e-06    228 0.0021091 0.28601 0.055308
    ## 210 3.7182e-06    229 0.0021050 0.28607 0.055308
    ## 211 3.7182e-06    230 0.0021012 0.28607 0.055308
    ## 212 3.6423e-06    231 0.0020975 0.28607 0.055308
    ## 213 3.2376e-06    232 0.0020939 0.28613 0.055307
    ## 214 3.2376e-06    233 0.0020906 0.28613 0.055307
    ## 215 3.2376e-06    234 0.0020874 0.28613 0.055307
    ## 216 2.7317e-06    235 0.0020842 0.28609 0.055308
    ## 217 2.4788e-06    236 0.0020814 0.28604 0.055308
    ## 218 2.4788e-06    238 0.0020765 0.28604 0.055308
    ## 219 1.8970e-06    239 0.0020740 0.28604 0.055308
    ## 220 1.8212e-06    240 0.0020721 0.28594 0.055309
    ## 221 1.8212e-06    241 0.0020703 0.28594 0.055309
    ## 222 1.2647e-06    242 0.0020685 0.28593 0.055309
    ## 223 8.0940e-07    243 0.0020672 0.28598 0.055309
    ## 224 6.8293e-07    244 0.0020664 0.28596 0.055309
    ## 225 4.5529e-07    245 0.0020657 0.28596 0.055309
    ## 226 2.0235e-07    247 0.0020648 0.28597 0.055309
    ## 227 2.0235e-07    248 0.0020646 0.28597 0.055309
    ## 228 2.0235e-07    249 0.0020644 0.28597 0.055309
    ## 229 1.0000e-08    251 0.0020640 0.28597 0.055309

It is probably better and easier to find this optimal value *programatically* as follows:

``` r
( b <- bos.to$cptable[which.min(bos.to$cptable[,"xerror"]),"CP"] )
```

    ## [1] 0.007729171

> **R coding digression**: Note that above we could also have used the following:
>
> ``` r
> tmp <- bos.to$cptable[,"xerror"]
> (b <- bos.to$cptable[ max( which(tmp == min(tmp)) ), "CP"] )
> ```
>
>     ## [1] 0.007729171
>
> What is the difference between `which.min(a)` and `max( which( a == min(a) ) )`?

We can now use the function `prune` on the `rpart` object setting the complexity parameter to the estimated optimal value found above:

``` r
bos.t3 <- prune(bos.to, cp=b)
```

This is how the optimally pruned tree looks:

``` r
plot(bos.t3, uniform=FALSE, margin=0.01)
text(bos.t3, pretty=TRUE)
```

![](README_files/figure-markdown_github/prune4.5-1.png)

Finally, we can verify that the predictions of the pruned tree on the test set are better than before:

``` r
# predictions are better
pr.t3 <- predict(bos.t3, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.t3)^2) )
```

    ## [1] 19.59567

Again, it would be a **very good exercise** for you to compare the MSPE of the pruned tree with that of several of the alternative methods we have seen in class so far, **without using a training / test split**.

#### Why is the pruned tree not a subtree of the "default" one?

Note that the pruned tree above is not a subtree of the one constructed using the default stopping criteria. In particular, note that the node to the right of the cut "lstat &gt;= 14.4" is split with the cut "dis &gt;= 1.385", whereas in the original tree, the corresponding node was split using "lstat &gt;= 4.91":

``` r
set.seed(123)
bos.t <- rpart(medv ~ ., data=dat.tr, method='anova')
plot(bos.t, uniform=FALSE, margin=0.01)
text(bos.t, pretty=TRUE)
```

![](README_files/figure-markdown_github/prune6-1.png)

Although "intuitively" one may say that building an overfitting tree means "running the tree algorithm longer" (in other words, relaxing the stopping rules will just make the splitting algorithm run longer), this is not the case. The reason for this difference is that one of the default "stopping" criteria is to set a limit on the minimum size of a child node. This default limit in `rpart` is 7 (`round(20/3)`). When we relaxed the tree building criteria this limit was reduced (to 1) and thus the "default" tree is not in fact a subtree of the large tree (that is later pruned). In particular, note that the split "dis &gt;= 1.38485" leaves a node with only 4 observations, which means that this split would not have been considered when building the "default" tree. You can verify this by inspecting the pruned tree

``` r
bos.t3
```

    ## n= 380 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##  1) root 380 32946.0800 22.42789  
    ##    2) rm< 6.825 310 12121.2900 19.38968  
    ##      4) lstat>=14.4 134  2591.6610 14.70821  
    ##        8) crim>=7.006285 59   605.7939 11.41017 *
    ##        9) crim< 7.006285 75   839.2795 17.30267 *
    ##      5) lstat< 14.4 176  4356.9170 22.95398  
    ##       10) dis>=1.38485 172  2094.9080 22.45349  
    ##         20) rm< 6.5255 140  1112.9610 21.50786 *
    ##         21) rm>=6.5255 32   309.0472 26.59062 *
    ##       11) dis< 1.38485 4   366.3075 44.47500  
    ##         22) rm< 5.7415 1     0.0000 27.90000 *
    ##         23) rm>=5.7415 3     0.0000 50.00000 *
    ##    3) rm>=6.825 70  5290.7390 35.88286  
    ##      6) rm< 7.437 47  1706.3390 31.59574  
    ##       12) nox>=0.659 2    27.3800 14.10000 *
    ##       13) nox< 0.659 45  1039.5480 32.37333  
    ##         26) dis>=1.33395 44   721.7873 31.97273 *
    ##         27) dis< 1.33395 1     0.0000 50.00000 *
    ##      7) rm>=7.437 23   955.3565 44.64348  
    ##       14) crim>=2.654025 1     0.0000 21.90000 *
    ##       15) crim< 2.654025 22   414.5786 45.67727 *

<!-- Note that pruning doesn't always improve a tree. For example,  -->
<!-- if we prune the first tree we fit in this example: -->
<!-- ```{r prune8, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- # what if we prune the original tree? -->
<!-- set.seed(123) -->
<!-- bos.t <- rpart(medv ~ ., data=dat.tr, method='anova') -->
<!-- b <- bos.t$cptable[which.min(bos.t$cptable[,"xerror"]),"CP"] -->
<!-- bos.t4 <- prune(bos.t, cp=b) -->
<!-- ``` -->
<!-- We obtain the same tree as before: -->
<!-- ```{r prune10, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- plot(bos.t4, uniform=FALSE, margin=0.01) -->
<!-- text(bos.t4, pretty=TRUE) -->
<!-- ``` -->
<!-- Below is the original tree: -->
<!-- ```{r prune6, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- plot(bos.t, uniform=FALSE, margin=0.01) -->
<!-- text(bos.t, pretty=TRUE) -->
<!-- ``` -->
#### Pruning regression trees with `tree`

The implementation of trees in the `R` package `tree` follows the original CV-based pruning strategy, as discussed in Section 3.4 of the book

> Breiman, Leo. (1984). Classification and regression trees. Wadsworth International Group

or Section 7.2 of:

> Ripley, Brian D. (1996). Pattern recognition and neural networks. Cambridge University Press

Both books are available in electronic form from the UBC Library.

We now use the function `tree::tree()` to fit the same regression tree as above. Note that the default stopping criteria in this implementation of regression trees is different from the one in `rpart::rpart()`, hence to obtain the same results as above we need to modify the default stopping criteria using the argument `control`:

``` r
library(tree)
bos.t2 <- tree(medv ~ ., data=dat.tr, control=tree.control(nobs=nrow(dat.tr), mincut=6, minsize=20))
```

We plot the resulting tree

``` r
plot(bos.t2); text(bos.t2)
```

![](README_files/figure-markdown_github/prunetree1-1.png)

As discussed before, we now fit a very large tree, which will be pruned later:

``` r
set.seed(123)
bos.to2 <- tree(medv ~ ., data=dat.tr, control=tree.control(nobs=nrow(dat.tr), mincut=1, minsize=2, mindev=1e-5))
plot(bos.to2)
```

![](README_files/figure-markdown_github/prunetree2-1.png)

We now use the function `tree:cv.tree()` to estimate the MSPE of the subtrees of `bos.to2`, using 5-fold CV, and plot the estimated MSPE (here labeled as "deviance") as a function of the complexity parameter (or, equivalently, the size of the tree):

``` r
set.seed(123)
tt <- cv.tree(bos.to2, K = 5)
plot(tt)
```

![](README_files/figure-markdown_github/prunetree3-1.png)

Finally, we use the function `prune.tree` to prune the larger tree at the "optimal" size, as estimated by `cv.tree` above:

``` r
bos.pr2 <- prune.tree(bos.to2, k = tt$k[ max( which(tt$dev == min(tt$dev)) ) ])
plot(bos.pr2); text(bos.pr2)
```

![](README_files/figure-markdown_github/prunetree3.2-1.png)

Compare this pruned tree with the one obtained with the regression trees implementation in `rpart`.

Instability of regression trees
-------------------------------

Trees can be rather unstable, in the sense that small changes in the training data set may result in relatively large differences in the fitted trees. As a simple illustration we randomly split the `Boston` data used before into two halves and fit a regression tree to each portion. We then display both trees.

``` r
# Instability of trees...
library(rpart)
data(Boston, package='MASS')
set.seed(654321)
n <- nrow(Boston)
ii <- sample(n, floor(n/2))
dat.t1 <- Boston[ -ii, ]
bos.t1 <- rpart(medv ~ ., data=dat.t1, method='anova')
plot(bos.t1, uniform=FALSE, margin=0.01)
text(bos.t1, pretty=TRUE, cex=.8)
```

![](README_files/figure-markdown_github/inst1-1.png)

``` r
dat.t2 <- Boston[ ii, ]
bos.t2 <- rpart(medv ~ ., data=dat.t2, method='anova')
plot(bos.t2, uniform=FALSE, margin=0.01)
text(bos.t2, pretty=TRUE, cex=.8)
```

![](README_files/figure-markdown_github/inst2-1.png)

Although we would expect both random halves of the same (moderately large) training set to beat least qualitatively similar, Note that the two trees are rather different. To compare with a more stable predictor, we fit a linear regression model to each half, and look at the two sets of estimated coefficients side by side:

``` r
# bos.lmf <- lm(medv ~ ., data=Boston)
bos.lm1 <- lm(medv ~ ., data=dat.t1)
bos.lm2 <- lm(medv ~ ., data=dat.t2)
cbind(round(coef(bos.lm1),2),
round(coef(bos.lm2),2))
```

    ##               [,1]   [,2]
    ## (Intercept)  27.10  46.32
    ## crim         -0.13  -0.05
    ## zn            0.05   0.04
    ## indus        -0.02   0.00
    ## chas          2.42   3.36
    ## nox         -10.56 -22.07
    ## rm            4.38   2.77
    ## age           0.01   0.00
    ## dis          -1.62  -1.31
    ## rad           0.30   0.30
    ## tax          -0.01  -0.01
    ## ptratio      -0.75  -1.08
    ## black         0.01   0.01
    ## lstat        -0.62  -0.46

Note that most of the estimated regression coefficients are similar, and all of them are at least qualitatively comparable.

Bagging
-------

One strategy to obtain more stable predictors is called **Bootstrap AGGregatING** (bagging). It can be applied to many predictors (not only trees), and it generally results in larger improvements in prediction quality when it is used with predictors that are flexible (low bias), but highly variable.

The justification and motivation were discussed in class. Intuitively we are averaging the predictions obtained from an estimate of the "average prediction" we would have computed had we had access to several (many?) independent training sets (samples).

There are several (many?) `R` packages implementing bagging for different predictors, with varying degrees of flexibility (the implementations) and user-friendliness. However, for pedagogical and illustrative purposes, in these notes I will *bagg* by hand.

<!-- ### Bagging by hand -->
<!-- Again, to simplify the discussion and presentation, in order to evaluate  -->
<!-- prediction quality I will split the  -->
<!-- data (`Boston`) into a training and a test set. We do this now: -->
<!-- ```{r bag1, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- set.seed(123456) -->
<!-- n <- nrow(Boston) -->
<!-- ii <- sample(n, floor(n/4)) -->
<!-- dat.te <- Boston[ ii, ] -->
<!-- dat.tr <- Boston[ -ii, ] -->
<!-- ``` -->
<!-- I will now train $N = 5$ trees and average their predictions.  -->
<!-- Note that, in order to illustrate the process more -->
<!-- clearly, I will compute and store the $N \times n_e$ -->
<!-- predictions, where $n_e$ denotes the number of observations in  -->
<!-- the test set. This is not the best (most efficient) way of implementing *bagging*, -->
<!-- but the main purpose here is to understand **what** we are doing. Also note that -->
<!-- an alternative (better in terms of reusability of the -->
<!-- ensamble, but maybe still not the most efficient option) would be -->
<!-- to store the $N$ trees directly. This would also allow for -->
<!-- more elegant and easy to read code. Once again, this approach  -->
<!-- will be sacrificed in the altar of clarity of presentation and  -->
<!-- pedagogy (but do try it yourself!) -->
<!-- First create an array where we will store all the predictions: -->
<!-- ```{r bag2, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- N <- 5 -->
<!-- myps <- array(NA, dim=c(nrow(dat.te), N)) -->
<!-- con <- rpart.control(minsplit=3, cp=1e-3, xval=1) -->
<!-- ``` -->
<!-- The last object (`con`) contains my options to train large -->
<!-- (potentially overfitting) trees.  -->
<!-- ```{r bag3, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- n.tr <- nrow(dat.tr) -->
<!-- set.seed(123456) -->
<!-- for(j in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con) -->
<!--   myps[,j] <- predict(tmp, newdata=dat.te, type='vector') -->
<!-- } -->
<!-- pr.bagg <- rowMeans(myps) -->
<!-- with(dat.te, mean( (medv - pr.bagg)^2 ) ) -->
<!-- ``` -->
<!-- And compare with predictions from the pruned tree, and the -->
<!-- ones from other predictors discussed in the previous note: -->
<!-- ```{r bag4, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- myc <- rpart.control(minsplit=3, cp=1e-8, xval=10) -->
<!-- set.seed(123) -->
<!-- bos.to <- rpart(medv ~ ., data=dat.tr, method='anova', -->
<!--                 control=myc) -->
<!-- b <- bos.to$cptable[which.min(bos.to$cptable[,"xerror"]),"CP"] -->
<!-- bos.t3 <- prune(bos.to, cp=b) -->
<!-- pr.t3 <- predict(bos.t3, newdata=dat.te, type='vector') -->
<!-- with(dat.te, mean((medv - pr.t3)^2) ) -->
<!-- ``` -->
<!-- What if we *bagg* $N = 10$ trees?  -->
<!-- ```{r bag10, fig.width=5, fig.height=5, message=FALSE, warning=FALSE, echo=FALSE} -->
<!-- N <- 10 -->
<!-- myps <- array(NA, dim=c(nrow(dat.te), N)) -->
<!-- n.tr <- nrow(dat.tr) -->
<!-- set.seed(123456) -->
<!-- for(j in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con) -->
<!--   myps[,j] <- predict(tmp, newdata=dat.te, type='vector') -->
<!-- } -->
<!-- pr.bagg <- rowMeans(myps) -->
<!-- with(dat.te, mean( (medv - pr.bagg)^2 ) ) -->
<!-- ``` -->
<!-- or $N = 100$ trees?  -->
<!-- ```{r bag100, fig.width=5, fig.height=5, message=FALSE, warning=FALSE, echo=FALSE} -->
<!-- N <- 100 -->
<!-- myps <- array(NA, dim=c(nrow(dat.te), N)) -->
<!-- n.tr <- nrow(dat.tr) -->
<!-- set.seed(123456) -->
<!-- for(j in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con) -->
<!--   myps[,j] <- predict(tmp, newdata=dat.te, type='vector') -->
<!-- } -->
<!-- pr.bagg <- rowMeans(myps) -->
<!-- with(dat.te, mean( (medv - pr.bagg)^2 ) ) -->
<!-- ``` -->
<!-- or $N = 1000$ trees?  -->
<!-- ```{r bag1000, fig.width=5, fig.height=5, message=FALSE, warning=FALSE, echo=FALSE} -->
<!-- N <- 1000 -->
<!-- myps <- array(NA, dim=c(nrow(dat.te), N)) -->
<!-- n.tr <- nrow(dat.tr) -->
<!-- set.seed(123456) -->
<!-- for(j in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con) -->
<!--   myps[,j] <- predict(tmp, newdata=dat.te, type='vector') -->
<!-- } -->
<!-- pr.bagg <- rowMeans(myps) -->
<!-- with(dat.te, mean( (medv - pr.bagg)^2 ) ) -->
<!-- ``` -->
<!-- Should we consider higher values of $N$? How about other -->
<!-- training / test splits? Should we use CV instead?  -->
<!-- Another split: -->
<!-- ```{r anothersplit, fig.width=5, fig.height=5, message=FALSE, warning=FALSE, echo=FALSE} -->
<!-- set.seed(123) -->
<!-- n <- nrow(Boston) -->
<!-- ii <- sample(n, floor(n/4)) -->
<!-- dat.te <- Boston[ ii, ] -->
<!-- dat.tr <- Boston[ -ii, ] -->
<!-- for(N in c(5, 10, 100, 1000)) { -->
<!-- myps <- array(NA, dim=c(nrow(dat.te), N)) -->
<!-- n.tr <- nrow(dat.tr) -->
<!-- set.seed(123456) -->
<!-- for(j in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   tmp <- rpart(medv ~ ., data=dat.tr[ii, ], method='anova', control=con) -->
<!--   myps[,j] <- predict(tmp, newdata=dat.te, type='vector') -->
<!-- } -->
<!-- pr.bagg <- rowMeans(myps) -->
<!-- print(c(N, with(dat.te, mean( (medv - pr.bagg)^2 ) ))) -->
<!-- } -->
<!-- ``` -->
<!-- Similar conclusion: increasing $N$ helps, but the improvement  -->
<!-- becomes smaller, while the computational cost keeps increasing.  -->
<!-- ### Bagging a regression spline -->
<!-- Bagging does not provide much of an advantage when applied to linear -->
<!-- predictors (can you explain why?) Nevertheless, let us try it on the `lidar` data,  -->
<!-- which, as we did before, we randomly split into a training and test set: -->
<!-- ```{r bagsplines, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- data(lidar, package='SemiPar') -->
<!-- set.seed(123456) -->
<!-- n <- nrow(lidar) -->
<!-- ii <- sample(n, floor(n/5)) -->
<!-- lid.te <- lidar[ ii, ] -->
<!-- lid.tr <- lidar[ -ii, ] -->
<!-- ``` -->
<!-- Now fit a cubic spline, and estimate the MSPE using the test set: -->
<!-- ```{r bagsplines2, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- library(splines) -->
<!-- a <- lm(logratio ~ bs(x=range, df=10, degree=3), data=lid.tr)  -->
<!-- oo <- order(lid.tr$range) -->
<!-- pr.of <- predict(a, newdata=lid.te) -->
<!-- mean( (lid.te$logratio - pr.of)^2 ) -->
<!-- ``` -->
<!-- We build an ensemble of 10 fits and estimate the corresponding -->
<!-- MSPE using the test set: -->
<!-- ```{r bagsplines3, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- N <- 10 # 5 500 1500 -->
<!-- myps <- matrix(NA, nrow(lid.te), N) -->
<!-- set.seed(123456) -->
<!-- n.tr <- nrow(lid.tr) -->
<!-- for(i in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   a.b <- lm(logratio ~ bs(x=range, df=10, degree=3), data=lid.tr[ii,])  -->
<!--   myps[,i] <- predict(a.b, newdata=lid.te) -->
<!-- } -->
<!-- pr.ba <- rowMeans(myps)# , na.rm=TRUE) -->
<!-- mean( (lid.te$logratio - pr.ba)^2 ) -->
<!-- ``` -->
<!-- Using smoothing splines? -->
<!-- ```{r bagsmooth, fig.width=5, fig.height=5, message=FALSE, warning=FALSE} -->
<!-- a <- smooth.spline(x = lid.tr$range, y = lid.tr$logratio, cv = TRUE, all.knots = TRUE) -->
<!-- pr.of <- predict(a, x=lid.te$range)$y -->
<!-- mean( (lid.te$logratio - pr.of)^2 ) -->
<!-- ``` -->
<!-- Using ensemble of 10: -->
<!-- ```{r bagsmooth2, fig.width=5, fig.height=5, message=FALSE, warning=FALSE, echo=FALSE} -->
<!-- N <- 10 # 5 500 1500 -->
<!-- myps <- matrix(NA, nrow(lid.te), N) -->
<!-- set.seed(123456) -->
<!-- n.tr <- nrow(lid.tr) -->
<!-- for(i in 1:N) { -->
<!--   ii <- sample(n.tr, replace=TRUE) -->
<!--   a.b <- smooth.spline(x = lid.tr$range[ii], y = lid.tr$logratio[ii], cv = TRUE, all.knots = TRUE) -->
<!--   myps[,i] <- predict(a.b, x=lid.te$range)$y -->
<!-- } -->
<!-- pr.ba <- rowMeans(myps)# , na.rm=TRUE) -->
<!-- mean( (lid.te$logratio - pr.ba)^2 ) -->
<!-- ``` -->
