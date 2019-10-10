STAT406 - Lecture 11 notes
================
Matias Salibian-Barrera
2019-10-09

#### LICENSE

These notes are released under the “Creative Commons
Attribution-ShareAlike 4.0 International” license. See the
**human-readable version**
[here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real
thing**
[here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

## Lecture slides

Lecture slides are [here](STAT406-18-lecture-11.pdf).

## Pruning regression trees with `rpart`

***Important note**: As discussed in class, the K-fold CV methodology
implemented in the package `rpart` seems to consider a sequence of trees
(or, equivalently, of complexity parameters) based on the full training
set. For more details refer to the corresponding documentation: pages 12
and ff of the package vignette, which can be accessed from `R` using the
command `vignette('longintro', package='rpart')`. For an alternative
implementation of CV-based pruning, please see also the Section
**“Pruning regression trees with `tree`”** below.*

The stopping criteria generally used when fitting regression trees do
not take into account explicitly the complexity of the tree. Hence, we
may end up with either an overfitting tree, or a very simple one, which
typically results in a decline in the quality of the corresponding
predictions. As discussed in class, one solution is to purposedly grow /
train a very large overfitting tree, and then prune it. One can also
estimate the corresponding MSPE of each tree in the prunning sequence
and choose an optimal one. The function `rpart` implements this
approach, and we illustrate it below.

We force `rpart` to build a very large tree via the arguments of the
function `rpart.control`. At the same time, to obtain a good picture of
the evolution of MSPE for different subtrees, we set the smallest
complexity parameter to be considered by the cross-validation experiment
to a very low value (here we use `1e-8`).

``` r
library(rpart)
data(Boston, package='MASS')
# split data into a training and
# a test set
set.seed(123) 
n <- nrow(Boston)
ii <- sample(n, floor(n/4))
dat.te <- Boston[ ii, ]
dat.tr <- Boston[ -ii, ]

myc <- rpart.control(minsplit=2, cp=1e-5, xval=5)
set.seed(123)
bos.to <- rpart(medv ~ ., data=dat.tr, method='anova',
                control=myc)
plot(bos.to, compress=TRUE) # type='proportional')
```

![](README_files/figure-gfm/prune-1.png)<!-- -->

Not surprisingly, the predictions of this large tree are not very good:

``` r
# predictions are poor, unsurprisingly
pr.to <- predict(bos.to, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.to)^2) )
```

    ## [1] 15.11998

To prune we explore the *CP table* returned in the `rpart` object to
find the value of the complexity parameter with optimal estimated
prediction error. The estimated prediction error of each subtree
(corresponding to each value of `CP`) is contained in the column
`xerror`, and the associated standard deviation is in column `xstd`. We
would like to find the value of `CP` that yields a corresponding pruned
tree with smallest estimated prediction error. The function `printcp`
shows the CP table corresponding to an `rpart` object:

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
    ## Root node error: 34400/380 = 90.526
    ## 
    ## n= 380 
    ## 
    ##             CP nsplit  rel error  xerror     xstd
    ## 1   4.6902e-01      0 1.00000000 1.00552 0.093762
    ## 2   1.6021e-01      1 0.53097703 0.61081 0.058489
    ## 3   6.8838e-02      2 0.37076479 0.42296 0.047729
    ## 4   5.0662e-02      3 0.30192661 0.37303 0.049837
    ## 5   2.9168e-02      4 0.25126467 0.28434 0.035232
    ## 6   2.7593e-02      5 0.22209691 0.26489 0.034998
    ## 7   2.2040e-02      6 0.19450414 0.25798 0.035025
    ## 8   1.7409e-02      7 0.17246395 0.24649 0.035193
    ## 9   1.1351e-02      8 0.15505502 0.25006 0.035307
    ## 10  1.1016e-02      9 0.14370379 0.24015 0.036519
    ## 11  9.1490e-03     10 0.13268765 0.23448 0.036408
    ## 12  7.9746e-03     11 0.12353869 0.22547 0.036025
    ## 13  6.9037e-03     12 0.11556406 0.21868 0.035847
    ## 14  5.6087e-03     13 0.10866039 0.22123 0.036836
    ## 15  5.5546e-03     15 0.09744295 0.21755 0.036613
    ## 16  5.5466e-03     16 0.09188837 0.21755 0.036613
    ## 17  4.0935e-03     17 0.08634172 0.21932 0.036598
    ## 18  3.5102e-03     18 0.08224824 0.21385 0.035870
    ## 19  3.3512e-03     19 0.07873805 0.21372 0.035898
    ## 20  2.9423e-03     20 0.07538687 0.20865 0.034586
    ## 21  2.6512e-03     21 0.07244456 0.20849 0.034898
    ## 22  2.5154e-03     22 0.06979332 0.20280 0.034684
    ## 23  2.2085e-03     23 0.06727797 0.20538 0.034898
    ## 24  2.0593e-03     24 0.06506942 0.21023 0.037624
    ## 25  1.9810e-03     25 0.06301010 0.20986 0.037639
    ## 26  1.9511e-03     27 0.05904803 0.20895 0.037628
    ## 27  1.8277e-03     28 0.05709694 0.20899 0.037622
    ## 28  1.7458e-03     29 0.05526921 0.20856 0.037564
    ## 29  1.6559e-03     30 0.05352340 0.21167 0.037647
    ## 30  1.6497e-03     31 0.05186753 0.21379 0.037695
    ## 31  1.5424e-03     32 0.05021779 0.21123 0.037611
    ## 32  1.5100e-03     33 0.04867536 0.21185 0.037619
    ## 33  1.5019e-03     34 0.04716535 0.21185 0.037619
    ## 34  1.4570e-03     35 0.04566342 0.21172 0.037621
    ## 35  1.4274e-03     36 0.04420640 0.21430 0.037875
    ## 36  1.3582e-03     37 0.04277899 0.21280 0.037797
    ## 37  1.2051e-03     38 0.04142078 0.21257 0.037749
    ## 38  1.0909e-03     39 0.04021566 0.21861 0.038153
    ## 39  1.0566e-03     40 0.03912474 0.21933 0.038148
    ## 40  1.0193e-03     41 0.03806815 0.21962 0.038144
    ## 41  9.9963e-04     43 0.03602963 0.21915 0.038181
    ## 42  9.9828e-04     46 0.03303075 0.21916 0.038180
    ## 43  9.8063e-04     47 0.03203246 0.21917 0.038177
    ## 44  9.2797e-04     48 0.03105184 0.21729 0.038153
    ## 45  9.0584e-04     50 0.02919590 0.21873 0.037608
    ## 46  8.1100e-04     51 0.02829006 0.21672 0.037324
    ## 47  7.9078e-04     52 0.02747906 0.21559 0.037062
    ## 48  7.7292e-04     53 0.02668829 0.21574 0.037061
    ## 49  7.2626e-04     54 0.02591537 0.21633 0.037064
    ## 50  6.4954e-04     55 0.02518911 0.21853 0.037306
    ## 51  6.3314e-04     57 0.02389003 0.22133 0.037335
    ## 52  6.0300e-04     58 0.02325689 0.22039 0.037306
    ## 53  5.4493e-04     59 0.02265389 0.22385 0.037354
    ## 54  5.4168e-04     60 0.02210895 0.22398 0.037351
    ## 55  5.4084e-04     61 0.02156728 0.22398 0.037351
    ## 56  5.3382e-04     62 0.02102643 0.22398 0.037351
    ## 57  5.1617e-04     64 0.01995880 0.22426 0.037347
    ## 58  4.7365e-04     65 0.01944263 0.22510 0.037587
    ## 59  4.5337e-04     66 0.01896898 0.22486 0.037601
    ## 60  4.5089e-04     68 0.01806224 0.22487 0.037599
    ## 61  4.4651e-04     70 0.01716047 0.22496 0.037598
    ## 62  4.4032e-04     71 0.01671395 0.22496 0.037598
    ## 63  4.1008e-04     72 0.01627363 0.22569 0.037593
    ## 64  4.0747e-04     73 0.01586356 0.22610 0.037646
    ## 65  4.0200e-04     74 0.01545609 0.22610 0.037646
    ## 66  3.9682e-04     75 0.01505409 0.22578 0.037646
    ## 67  3.7175e-04     76 0.01465727 0.22680 0.037595
    ## 68  3.5854e-04     77 0.01428552 0.22680 0.037595
    ## 69  3.5622e-04     78 0.01392698 0.22673 0.037587
    ## 70  3.3377e-04     80 0.01321454 0.22647 0.037590
    ## 71  3.2491e-04     81 0.01288077 0.22450 0.036125
    ## 72  3.1063e-04     82 0.01255585 0.22443 0.036126
    ## 73  2.9237e-04     83 0.01224522 0.22303 0.036060
    ## 74  2.7716e-04     85 0.01166048 0.22316 0.036107
    ## 75  2.7715e-04     86 0.01138332 0.22292 0.036106
    ## 76  2.6174e-04     87 0.01110617 0.22289 0.036107
    ## 77  2.5250e-04     88 0.01084443 0.22331 0.036118
    ## 78  2.5117e-04     89 0.01059193 0.22349 0.036114
    ## 79  2.3740e-04     90 0.01034076 0.22283 0.036309
    ## 80  2.3280e-04     91 0.01010335 0.22301 0.036305
    ## 81  2.3143e-04     92 0.00987055 0.22314 0.036307
    ## 82  2.3130e-04     93 0.00963913 0.22314 0.036307
    ## 83  2.2388e-04     94 0.00940783 0.22295 0.036305
    ## 84  2.2108e-04     95 0.00918396 0.22419 0.036350
    ## 85  2.1415e-04     96 0.00896288 0.22387 0.036354
    ## 86  2.1225e-04     97 0.00874873 0.22375 0.036356
    ## 87  2.0652e-04     98 0.00853648 0.22453 0.036391
    ## 88  2.0504e-04     99 0.00832996 0.22447 0.036389
    ## 89  2.0470e-04    100 0.00812493 0.22447 0.036389
    ## 90  1.9845e-04    101 0.00792023 0.22448 0.036390
    ## 91  1.8624e-04    102 0.00772178 0.22526 0.036400
    ## 92  1.8028e-04    103 0.00753553 0.22503 0.036384
    ## 93  1.7399e-04    104 0.00735525 0.22518 0.036388
    ## 94  1.6689e-04    105 0.00718126 0.22505 0.036404
    ## 95  1.5721e-04    106 0.00701438 0.22641 0.036395
    ## 96  1.5603e-04    107 0.00685717 0.22643 0.036406
    ## 97  1.5476e-04    108 0.00670113 0.22654 0.036405
    ## 98  1.5194e-04    109 0.00654637 0.22641 0.036400
    ## 99  1.4899e-04    110 0.00639443 0.22654 0.036401
    ## 100 1.4717e-04    111 0.00624544 0.22615 0.036397
    ## 101 1.4363e-04    112 0.00609827 0.22645 0.036394
    ## 102 1.4128e-04    113 0.00595464 0.22715 0.036391
    ## 103 1.3722e-04    114 0.00581336 0.22722 0.036390
    ## 104 1.2843e-04    115 0.00567614 0.22752 0.036387
    ## 105 1.1939e-04    116 0.00554771 0.22819 0.036389
    ## 106 1.1939e-04    117 0.00542832 0.22812 0.036388
    ## 107 1.1533e-04    118 0.00530893 0.22812 0.036388
    ## 108 1.1463e-04    119 0.00519360 0.22858 0.036383
    ## 109 1.0642e-04    120 0.00507897 0.22863 0.036393
    ## 110 1.0552e-04    121 0.00497254 0.22819 0.036398
    ## 111 1.0501e-04    122 0.00486702 0.22839 0.036425
    ## 112 1.0494e-04    123 0.00476201 0.22839 0.036425
    ## 113 1.0367e-04    124 0.00465706 0.22858 0.036423
    ## 114 1.0081e-04    125 0.00455339 0.22858 0.036423
    ## 115 9.9225e-05    126 0.00445258 0.22845 0.036422
    ## 116 9.7533e-05    127 0.00435335 0.22842 0.036422
    ## 117 9.6148e-05    129 0.00415829 0.22827 0.036422
    ## 118 9.5027e-05    130 0.00406214 0.22806 0.036424
    ## 119 9.0843e-05    132 0.00387208 0.22799 0.036423
    ## 120 8.7452e-05    133 0.00378124 0.22788 0.036420
    ## 121 8.5620e-05    134 0.00369379 0.22818 0.036421
    ## 122 8.3721e-05    135 0.00360817 0.22861 0.036420
    ## 123 8.3721e-05    136 0.00352445 0.22870 0.036416
    ## 124 8.1396e-05    137 0.00344073 0.22892 0.036425
    ## 125 7.9593e-05    138 0.00335933 0.22926 0.036457
    ## 126 7.8878e-05    139 0.00327974 0.22882 0.036457
    ## 127 7.7520e-05    140 0.00320086 0.22873 0.036460
    ## 128 7.5349e-05    141 0.00312334 0.22868 0.036461
    ## 129 7.3703e-05    142 0.00304799 0.22830 0.036468
    ## 130 6.5504e-05    143 0.00297429 0.22800 0.036472
    ## 131 6.4099e-05    144 0.00290879 0.22764 0.036473
    ## 132 6.2791e-05    145 0.00284469 0.22766 0.036468
    ## 133 6.2791e-05    146 0.00278190 0.22778 0.036449
    ## 134 5.7689e-05    147 0.00271910 0.22717 0.036416
    ## 135 5.5881e-05    148 0.00266142 0.22719 0.036417
    ## 136 5.4506e-05    149 0.00260553 0.22681 0.036414
    ## 137 5.4084e-05    150 0.00255103 0.22700 0.036416
    ## 138 5.3513e-05    151 0.00249694 0.22700 0.036416
    ## 139 5.1746e-05    152 0.00244343 0.22702 0.036417
    ## 140 5.0233e-05    153 0.00239169 0.22695 0.036414
    ## 141 4.9613e-05    154 0.00234145 0.22696 0.036414
    ## 142 4.9409e-05    155 0.00229184 0.22709 0.036413
    ## 143 4.8866e-05    157 0.00219302 0.22705 0.036413
    ## 144 4.7481e-05    158 0.00214416 0.22700 0.036414
    ## 145 4.7224e-05    159 0.00209668 0.22693 0.036408
    ## 146 4.7093e-05    160 0.00204945 0.22693 0.036408
    ## 147 4.6899e-05    161 0.00200236 0.22693 0.036408
    ## 148 4.5582e-05    162 0.00195546 0.22845 0.037208
    ## 149 4.4057e-05    163 0.00190988 0.22845 0.037208
    ## 150 4.3605e-05    164 0.00186582 0.22875 0.037208
    ## 151 4.3605e-05    165 0.00182222 0.22875 0.037208
    ## 152 4.2384e-05    166 0.00177861 0.22875 0.037208
    ## 153 4.2209e-05    167 0.00173623 0.22875 0.037208
    ## 154 4.2209e-05    168 0.00169402 0.22875 0.037208
    ## 155 4.2006e-05    169 0.00165181 0.22874 0.037206
    ## 156 3.7985e-05    170 0.00160980 0.22824 0.037173
    ## 157 3.7248e-05    171 0.00157182 0.22869 0.037175
    ## 158 3.7209e-05    172 0.00153457 0.22869 0.037175
    ## 159 3.6846e-05    174 0.00146015 0.22841 0.037174
    ## 160 3.6846e-05    175 0.00142330 0.22841 0.037174
    ## 161 3.6555e-05    176 0.00138646 0.22838 0.037175
    ## 162 3.5320e-05    178 0.00131335 0.22808 0.037175
    ## 163 3.2704e-05    179 0.00127803 0.22730 0.037163
    ## 164 2.9433e-05    181 0.00121262 0.22723 0.037162
    ## 165 2.9187e-05    182 0.00118319 0.22709 0.037144
    ## 166 2.7907e-05    184 0.00112481 0.22704 0.037145
    ## 167 2.6381e-05    185 0.00109691 0.22706 0.037146
    ## 168 2.6163e-05    186 0.00107053 0.22719 0.037149
    ## 169 2.5630e-05    188 0.00101820 0.22742 0.037151
    ## 170 2.4564e-05    189 0.00099257 0.22749 0.037151
    ## 171 2.2108e-05    190 0.00096801 0.22751 0.037152
    ## 172 2.1366e-05    191 0.00094590 0.22747 0.037153
    ## 173 2.1366e-05    192 0.00092453 0.22747 0.037153
    ## 174 2.0930e-05    193 0.00090317 0.22747 0.037153
    ## 175 2.0930e-05    194 0.00088224 0.22747 0.037153
    ## 176 2.0373e-05    196 0.00084038 0.22747 0.037153
    ## 177 1.9622e-05    197 0.00082000 0.22746 0.037154
    ## 178 1.9380e-05    198 0.00080038 0.22732 0.037156
    ## 179 1.9380e-05    199 0.00078100 0.22732 0.037156
    ## 180 1.7660e-05    200 0.00076162 0.22732 0.037156
    ## 181 1.7587e-05    201 0.00074396 0.22731 0.037155
    ## 182 1.7587e-05    203 0.00070879 0.22731 0.037155
    ## 183 1.7490e-05    204 0.00069120 0.22731 0.037155
    ## 184 1.7490e-05    205 0.00067371 0.22745 0.037157
    ## 185 1.6376e-05    206 0.00065622 0.22743 0.037157
    ## 186 1.6097e-05    207 0.00063984 0.22743 0.037157
    ## 187 1.5698e-05    209 0.00060765 0.22743 0.037157
    ## 188 1.5504e-05    210 0.00059195 0.22748 0.037156
    ## 189 1.4884e-05    211 0.00057645 0.22750 0.037156
    ## 190 1.4535e-05    212 0.00056156 0.22722 0.037157
    ## 191 1.4244e-05    216 0.00050342 0.22724 0.037157
    ## 192 1.4244e-05    217 0.00048918 0.22724 0.037157
    ## 193 1.4002e-05    218 0.00047493 0.22724 0.037157
    ## 194 1.3081e-05    220 0.00044693 0.22739 0.037156
    ## 195 1.2815e-05    221 0.00043385 0.22755 0.037157
    ## 196 1.2403e-05    222 0.00042103 0.22755 0.037157
    ## 197 1.2282e-05    223 0.00040863 0.22755 0.037157
    ## 198 1.2282e-05    224 0.00039635 0.22755 0.037157
    ## 199 1.1773e-05    225 0.00038407 0.22758 0.037157
    ## 200 1.1725e-05    226 0.00037229 0.22770 0.037149
    ## 201 1.1725e-05    227 0.00036057 0.22770 0.037149
    ## 202 1.0901e-05    228 0.00034884 0.22789 0.037150
    ## 203 1.0901e-05    229 0.00033794 0.22792 0.037149
    ## 204 1.0683e-05    230 0.00032704 0.22792 0.037149
    ## 205 1.0000e-05    231 0.00031636 0.22792 0.037149

It is probably better and easier to find this optimal value
*programatically* as follows:

``` r
( b <- bos.to$cptable[which.min(bos.to$cptable[,"xerror"]),"CP"] )
```

    ## [1] 0.002515355

<!-- > **R coding digression**: Note that above we could also have used the following: -->

<!-- > ```{r prune4.alt, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->

<!-- > tmp <- bos.to$cptable[,"xerror"] -->

<!-- > (b <- bos.to$cptable[ max( which(tmp == min(tmp)) ), "CP"] ) -->

<!-- > ``` -->

<!-- > What is the difference between `which.min(a)` and `max( which( a == min(a) ) )`? -->

We can now use the function `prune` on the `rpart` object setting the
complexity parameter to the estimated optimal value found above:

``` r
bos.t3 <- prune(bos.to, cp=b)
```

This is how the optimally pruned tree looks:

``` r
plot(bos.t3, uniform=FALSE, margin=0.01)
text(bos.t3, pretty=FALSE)
```

![](README_files/figure-gfm/prune4.5-1.png)<!-- -->

Finally, we can verify that the predictions of the pruned tree on the
test set are better than before:

``` r
# predictions are better
pr.t3 <- predict(bos.t3, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.t3)^2) )
```

    ## [1] 10.94228

Again, it would be a **very good exercise** for you to compare the MSPE
of the pruned tree with that of several of the alternative methods we
have seen in class so far, **without using a training / test split**.

<!-- #### Why is the pruned tree not a subtree of the "default" one? -->

<!-- Note that the pruned tree above is not a subtree of the one -->

<!-- constructed using the default stopping criteria.  -->

<!-- This description is outdated. Will be updated soon. In particular, note  -->

<!-- that the node to the right of the cut "lstat >= 14.4" is split  -->

<!-- with the cut "dis >= 1.385", whereas in the original tree,  -->

<!-- the corresponding node was split using "lstat >= 4.91": -->

<!-- ```{r prune6, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->

<!-- set.seed(123) -->

<!-- bos.t <- rpart(medv ~ ., data=dat.tr, method='anova') -->

<!-- plot(bos.t, uniform=FALSE, margin=0.01) -->

<!-- text(bos.t, pretty=TRUE) -->

<!-- ``` -->

<!-- Although "intuitively" one may say that building an overfitting tree -->

<!-- means "running the tree algorithm longer" (in other words, relaxing the  -->

<!-- stopping rules will just make the splitting algorithm run longer), this  -->

<!-- is not the case. The reason for this difference is that one  -->

<!-- of the default "stopping" criteria is to set a limit on the minimum  -->

<!-- size of a child node. This default limit in `rpart` is 7 -->

<!-- (`round(20/3)`). When we relaxed the tree building criteria this limit was reduced  -->

<!-- (to 1) and thus the "default" tree is not in fact a subtree of the large tree -->

<!-- (that is later pruned). In particular, note that  -->

<!-- the split "dis >= 1.38485" leaves a node with only 4 observations,  -->

<!-- which means that this split would not have been considered -->

<!-- when building the "default" tree. You can verify this by inspecting the  -->

<!-- pruned tree -->

<!-- ```{r prunecheck, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->

<!-- bos.t3 -->

<!-- ``` -->

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

The implementation of trees in the `R` package `tree` follows the
original CV-based pruning strategy, as discussed in Section 3.4 of the
book

> Breiman, Leo. (1984). Classification and regression trees. Wadsworth
> International Group

or Section 7.2 of:

> Ripley, Brian D. (1996). Pattern recognition and neural networks.
> Cambridge University Press

Both books are available in electronic form from the UBC Library.

We now use the function `tree::tree()` to fit the same regression tree
as above. Note that the default stopping criteria in this implementation
of regression trees is different from the one in `rpart::rpart()`, hence
to obtain the same results as above we need to modify the default
stopping criteria using the argument `control`:

``` r
library(tree)
bos.t2 <- tree(medv ~ ., data=dat.tr, control=tree.control(nobs=nrow(dat.tr), mincut=6, minsize=20))
```

We plot the resulting tree

``` r
plot(bos.t2); text(bos.t2)
```

![](README_files/figure-gfm/prunetree1-1.png)<!-- -->

As discussed before, we now fit a very large tree, which will be pruned
later:

``` r
set.seed(123)
bos.to2 <- tree(medv ~ ., data=dat.tr, control=tree.control(nobs=nrow(dat.tr), mincut=1, minsize=2, mindev=1e-5))
plot(bos.to2)
```

![](README_files/figure-gfm/prunetree2-1.png)<!-- -->

We now use the function `tree:cv.tree()` to estimate the MSPE of the
subtrees of `bos.to2`, using 5-fold CV, and plot the estimated MSPE
(here labeled as “deviance”) as a function of the complexity parameter
(or, equivalently, the size of the tree):

``` r
set.seed(123)
tt <- cv.tree(bos.to2, K = 5)
plot(tt)
```

![](README_files/figure-gfm/prunetree3-1.png)<!-- -->

Finally, we use the function `prune.tree` to prune the larger tree at
the “optimal” size, as estimated by `cv.tree` above:

``` r
bos.pr2 <- prune.tree(bos.to2, k = tt$k[ max( which(tt$dev == min(tt$dev)) ) ])
plot(bos.pr2); text(bos.pr2)
```

![](README_files/figure-gfm/prunetree3.2-1.png)<!-- -->

Compare this pruned tree with the one obtained with the regression trees
implementation in `rpart`. In particular, we can compare the predictions
of this other pruned tree on the test set:

``` r
# predictions are worse than the rpart-pruned tree
pr.tree <- predict(bos.pr2, newdata=dat.te, type='vector')
with(dat.te, mean((medv - pr.tree)^2) )
```

    ## [1] 12.3009

Note that the predictions of the tree pruned with the `tree` package
seem to be worse than those of the tree pruned with the `rpart` package.
**Does this mean that `rpart` gives trees with better predictions than
`tree` for data coming from the process than generated our training
set?** **Or could it all be an artifact of the specific test set we
used?** **Can you think of an experiment to verify this claim?** Again,
it would be a **very good exercise** for you to check which fit (`tree`
or `rpart`) gives pruned trees with better prediction properties in this
case.

## Instability of regression trees

Trees can be rather unstable, in the sense that small changes in the
training data set may result in relatively large differences in the
fitted trees. As a simple illustration we randomly split the `Boston`
data used before into two halves and fit a regression tree to each
portion. We then display both trees.

``` r
# Instability of trees...
library(rpart)
data(Boston, package='MASS')
set.seed(123)
n <- nrow(Boston)
ii <- sample(n, floor(n/2))
dat.t1 <- Boston[ -ii, ]
bos.t1 <- rpart(medv ~ ., data=dat.t1, method='anova')
plot(bos.t1, uniform=FALSE, margin=0.01)
text(bos.t1, pretty=TRUE, cex=.8)
```

![](README_files/figure-gfm/inst1-1.png)<!-- -->

``` r
dat.t2 <- Boston[ ii, ]
bos.t2 <- rpart(medv ~ ., data=dat.t2, method='anova')
plot(bos.t2, uniform=FALSE, margin=0.01)
text(bos.t2, pretty=TRUE, cex=.8)
```

![](README_files/figure-gfm/inst2-1.png)<!-- -->

Although we would expect both random halves of the same (moderately
large) training set to beat least qualitatively similar, Note that the
two trees are rather different. To compare with a more stable predictor,
we fit a linear regression model to each half, and look at the two sets
of estimated coefficients side by side:

``` r
# bos.lmf <- lm(medv ~ ., data=Boston)
bos.lm1 <- lm(medv ~ ., data=dat.t1)
bos.lm2 <- lm(medv ~ ., data=dat.t2)
cbind(round(coef(bos.lm1),2),
round(coef(bos.lm2),2))
```

    ##               [,1]   [,2]
    ## (Intercept)  35.47  32.35
    ## crim         -0.12  -0.09
    ## zn            0.04   0.05
    ## indus         0.01   0.03
    ## chas          0.90   3.98
    ## nox         -23.90 -12.33
    ## rm            5.01   3.39
    ## age          -0.01   0.00
    ## dis          -1.59  -1.41
    ## rad           0.33   0.28
    ## tax          -0.01  -0.01
    ## ptratio      -1.12  -0.72
    ## black         0.01   0.01
    ## lstat        -0.31  -0.66

Note that most of the estimated regression coefficients are similar, and
all of them are at least qualitatively comparable.

## Bagging

One strategy to obtain more stable predictors is called **Bootstrap
AGGregatING** (bagging). It can be applied to many predictors (not only
trees), and it generally results in larger improvements in prediction
quality when it is used with predictors that are flexible (low bias),
but highly variable.

The justification and motivation were discussed in class. Intuitively we
are averaging the predictions obtained from an estimate of the “average
prediction” we would have computed had we had access to several (many?)
independent training sets (samples).

There are several (many?) `R` packages implementing bagging for
different predictors, with varying degrees of flexibility (the
implementations) and user-friendliness. However, for pedagogical and
illustrative purposes, in these notes I will *bagg* by hand.

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
