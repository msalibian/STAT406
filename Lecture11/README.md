STAT406 - Lecture 11 notes
================
Matias Salibian-Barrera
2018-10-17

#### LICENSE

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

Lecture slides are [here](STAT406-18-lecture-11.pdf).

Pruning regression trees with `rpart`
-------------------------------------

***Important note**: As discussed in class, the K-fold CV methodology implemented in the package `rpart` seems to consider a sequence of trees (or, equivalently, of complexity parameters) based on the full training set. For more details refer to the corresponding documentation: pages 12 and ff of the package vignette, which can be accessed from `R` using the command `vignette('longintro', package='rpart')`. For an alternative implementation of CV-based pruning that follows the spirit of K-fold CV more closely, please see also the Section **"Pruning regression trees with `tree`"** below.*

The stopping criteria generally used when fitting regression trees do not take into account explicitly the complexity of the tree. Hence, we may end up with an overfitting tree, which typically results in a decline in the quality of the corresponding predictions. As discussed in class, one solution is to purposedly grow / train a very large overfitting tree, and then prune it. One can also estimate the corresponding MSPE of each tree in the prunning sequence and choose an optimal one. The function `rpart` implements this approach, and we illustrate it below.

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

    ## [1] 36.51097

To prune we explore the *CP table* returned in the `rpart` object to find the value of the complexity parameter with optimal estimated prediction error:

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
    ## Root node error: 33010/380 = 86.869
    ## 
    ## n= 380 
    ## 
    ##             CP nsplit rel error  xerror     xstd
    ## 1   4.8583e-01      0 1.0000000 1.00717 0.093142
    ## 2   1.6116e-01      1 0.5141667 0.55925 0.057254
    ## 3   6.4179e-02      2 0.3530038 0.41102 0.046946
    ## 4   5.5119e-02      3 0.2888251 0.36320 0.052820
    ## 5   3.4229e-02      4 0.2337058 0.34482 0.050358
    ## 6   2.9005e-02      5 0.1994769 0.30971 0.049040
    ## 7   1.6086e-02      6 0.1704715 0.26839 0.047553
    ## 8   1.1097e-02      7 0.1543852 0.25793 0.044987
    ## 9   9.3796e-03      8 0.1432884 0.23574 0.037247
    ## 10  9.2703e-03      9 0.1339088 0.26114 0.047891
    ## 11  8.7692e-03     10 0.1246385 0.26001 0.047785
    ## 12  7.7928e-03     11 0.1158693 0.25402 0.047728
    ## 13  6.6572e-03     12 0.1080766 0.25062 0.047732
    ## 14  5.4505e-03     13 0.1014193 0.24476 0.046452
    ## 15  5.0968e-03     14 0.0959689 0.24633 0.047325
    ## 16  4.2783e-03     15 0.0908721 0.24511 0.047325
    ## 17  4.1533e-03     16 0.0865937 0.24437 0.047361
    ## 18  4.0140e-03     17 0.0824404 0.24385 0.047367
    ## 19  2.6191e-03     18 0.0784264 0.25009 0.048241
    ## 20  2.5594e-03     19 0.0758073 0.24588 0.046215
    ## 21  2.3391e-03     20 0.0732479 0.24720 0.046313
    ## 22  2.3363e-03     21 0.0709089 0.24684 0.046350
    ## 23  1.9136e-03     23 0.0662363 0.25078 0.046453
    ## 24  1.8273e-03     24 0.0643227 0.25661 0.046259
    ## 25  1.8004e-03     25 0.0624953 0.25608 0.046239
    ## 26  1.7997e-03     26 0.0606949 0.25608 0.046239
    ## 27  1.7983e-03     27 0.0588952 0.25608 0.046239
    ## 28  1.7521e-03     29 0.0552987 0.25551 0.046232
    ## 29  1.7456e-03     30 0.0535466 0.25534 0.046245
    ## 30  1.7035e-03     31 0.0518010 0.25396 0.046119
    ## 31  1.6719e-03     32 0.0500975 0.25901 0.046168
    ## 32  1.6417e-03     33 0.0484255 0.25892 0.046176
    ## 33  1.5552e-03     34 0.0467839 0.25724 0.046136
    ## 34  1.4516e-03     35 0.0452287 0.25641 0.046145
    ## 35  1.4414e-03     36 0.0437771 0.25704 0.046145
    ## 36  1.4377e-03     37 0.0423356 0.25704 0.046145
    ## 37  1.3450e-03     39 0.0394602 0.25594 0.046063
    ## 38  1.2749e-03     40 0.0381151 0.25664 0.046020
    ## 39  1.2215e-03     41 0.0368402 0.25787 0.046012
    ## 40  9.8996e-04     42 0.0356187 0.25597 0.044855
    ## 41  9.2997e-04     43 0.0346288 0.25436 0.045300
    ## 42  9.2095e-04     44 0.0336988 0.25424 0.045297
    ## 43  8.9900e-04     45 0.0327779 0.25474 0.045311
    ## 44  8.3153e-04     46 0.0318788 0.25330 0.045230
    ## 45  7.7598e-04     47 0.0310473 0.25327 0.045257
    ## 46  7.6105e-04     48 0.0302713 0.25466 0.045402
    ## 47  7.2813e-04     49 0.0295103 0.25404 0.045396
    ## 48  7.2476e-04     50 0.0287822 0.25402 0.045398
    ## 49  6.9801e-04     51 0.0280574 0.25410 0.045403
    ## 50  6.6252e-04     52 0.0273594 0.25563 0.045415
    ## 51  6.2526e-04     53 0.0266969 0.25617 0.045452
    ## 52  6.1499e-04     54 0.0260716 0.25370 0.045432
    ## 53  5.4165e-04     55 0.0254566 0.25793 0.045662
    ## 54  5.3418e-04     56 0.0249150 0.26016 0.045639
    ## 55  5.2347e-04     57 0.0243808 0.26163 0.045726
    ## 56  5.1389e-04     58 0.0238573 0.26162 0.045727
    ## 57  5.1165e-04     59 0.0233434 0.26140 0.045730
    ## 58  5.0244e-04     60 0.0228318 0.26172 0.045731
    ## 59  5.0225e-04     61 0.0223293 0.26255 0.045743
    ## 60  4.9246e-04     62 0.0218271 0.26255 0.045743
    ## 61  4.7451e-04     63 0.0213346 0.26253 0.045743
    ## 62  4.5859e-04     64 0.0208601 0.26176 0.045724
    ## 63  4.3688e-04     65 0.0204015 0.26176 0.045775
    ## 64  4.3625e-04     66 0.0199647 0.26092 0.045774
    ## 65  4.1909e-04     67 0.0195284 0.26070 0.045777
    ## 66  4.1564e-04     68 0.0191093 0.26102 0.045780
    ## 67  3.9315e-04     69 0.0186937 0.26116 0.045781
    ## 68  3.9232e-04     70 0.0183005 0.26030 0.045780
    ## 69  3.8884e-04     71 0.0179082 0.26048 0.045778
    ## 70  3.8215e-04     72 0.0175194 0.26102 0.045797
    ## 71  3.6565e-04     73 0.0171372 0.26017 0.045116
    ## 72  3.6474e-04     74 0.0167716 0.25964 0.045046
    ## 73  3.5279e-04     76 0.0160421 0.25979 0.045048
    ## 74  3.4059e-04     79 0.0149837 0.25973 0.045048
    ## 75  3.3386e-04     80 0.0146431 0.25986 0.045048
    ## 76  3.2990e-04     81 0.0143093 0.25974 0.045049
    ## 77  3.0830e-04     82 0.0139794 0.25986 0.045050
    ## 78  3.0718e-04     83 0.0136711 0.26067 0.045046
    ## 79  2.9368e-04     84 0.0133639 0.26089 0.045018
    ## 80  2.8903e-04     85 0.0130702 0.26043 0.045020
    ## 81  2.8429e-04     86 0.0127812 0.26043 0.045020
    ## 82  2.8335e-04     87 0.0124969 0.26067 0.045018
    ## 83  2.7721e-04     88 0.0122136 0.26058 0.045019
    ## 84  2.7305e-04     89 0.0119364 0.26036 0.045021
    ## 85  2.6665e-04     90 0.0116633 0.26038 0.045020
    ## 86  2.6539e-04     91 0.0113967 0.25950 0.045014
    ## 87  2.6265e-04     92 0.0111313 0.25950 0.045014
    ## 88  2.5655e-04     93 0.0108686 0.25956 0.045016
    ## 89  2.5556e-04     94 0.0106121 0.25949 0.045017
    ## 90  2.2912e-04     96 0.0101009 0.26173 0.045018
    ## 91  2.2306e-04     98 0.0096427 0.26365 0.045070
    ## 92  2.1408e-04     99 0.0094196 0.26300 0.045064
    ## 93  2.1328e-04    101 0.0089915 0.26226 0.045046
    ## 94  1.9765e-04    102 0.0087782 0.26236 0.045046
    ## 95  1.9678e-04    104 0.0083829 0.26206 0.045048
    ## 96  1.9510e-04    106 0.0079893 0.26128 0.045047
    ## 97  1.7138e-04    107 0.0077942 0.26188 0.045044
    ## 98  1.5672e-04    108 0.0076229 0.26363 0.045059
    ## 99  1.5642e-04    109 0.0074661 0.26438 0.045064
    ## 100 1.5586e-04    111 0.0071533 0.26438 0.045064
    ## 101 1.4662e-04    112 0.0069974 0.26439 0.045062
    ## 102 1.4541e-04    113 0.0068508 0.26513 0.045058
    ## 103 1.3824e-04    114 0.0067054 0.26779 0.046019
    ## 104 1.3787e-04    115 0.0065672 0.26739 0.046017
    ## 105 1.3712e-04    116 0.0064293 0.26739 0.046017
    ## 106 1.3695e-04    117 0.0062922 0.26733 0.046018
    ## 107 1.2442e-04    118 0.0061552 0.26737 0.046020
    ## 108 1.2269e-04    119 0.0060308 0.26765 0.046019
    ## 109 1.1346e-04    120 0.0059081 0.26754 0.046022
    ## 110 1.1045e-04    122 0.0056812 0.26825 0.046032
    ## 111 1.0997e-04    123 0.0055708 0.26830 0.046032
    ## 112 1.0386e-04    124 0.0054608 0.26810 0.046027
    ## 113 1.0340e-04    125 0.0053569 0.26908 0.046028
    ## 114 1.0259e-04    126 0.0052535 0.26908 0.046028
    ## 115 1.0224e-04    127 0.0051509 0.26908 0.046028
    ## 116 1.0185e-04    128 0.0050487 0.26898 0.046025
    ## 117 9.5993e-05    129 0.0049468 0.26938 0.046022
    ## 118 9.3355e-05    130 0.0048509 0.26945 0.046022
    ## 119 8.4873e-05    131 0.0047575 0.26889 0.046026
    ## 120 8.3440e-05    132 0.0046726 0.26761 0.045989
    ## 121 8.2229e-05    134 0.0045057 0.26761 0.045989
    ## 122 7.8362e-05    135 0.0044235 0.26818 0.046021
    ## 123 7.8315e-05    137 0.0042668 0.26819 0.046021
    ## 124 7.8013e-05    138 0.0041885 0.26819 0.046021
    ## 125 7.4782e-05    139 0.0041105 0.26842 0.046018
    ## 126 7.4219e-05    140 0.0040357 0.26870 0.046015
    ## 127 7.3614e-05    141 0.0039615 0.26884 0.046014
    ## 128 7.2907e-05    142 0.0038878 0.26884 0.046014
    ## 129 7.0912e-05    143 0.0038149 0.26929 0.046022
    ## 130 7.0692e-05    144 0.0037440 0.26929 0.046022
    ## 131 6.9075e-05    145 0.0036733 0.26933 0.046021
    ## 132 6.1850e-05    146 0.0036043 0.26936 0.046021
    ## 133 6.1850e-05    147 0.0035424 0.26975 0.046261
    ## 134 6.0118e-05    148 0.0034806 0.27000 0.046262
    ## 135 5.9870e-05    149 0.0034204 0.26988 0.046260
    ## 136 5.9376e-05    150 0.0033606 0.27002 0.046259
    ## 137 5.8164e-05    151 0.0033012 0.27000 0.046259
    ## 138 5.8164e-05    152 0.0032430 0.27000 0.046259
    ## 139 5.3812e-05    153 0.0031849 0.27031 0.046260
    ## 140 4.8874e-05    154 0.0031311 0.27113 0.046376
    ## 141 4.8520e-05    155 0.0030822 0.27063 0.046357
    ## 142 4.8076e-05    156 0.0030337 0.27095 0.046356
    ## 143 4.5440e-05    157 0.0029856 0.27093 0.046357
    ## 144 4.1472e-05    158 0.0029401 0.27101 0.046356
    ## 145 3.9625e-05    159 0.0028987 0.27131 0.046357
    ## 146 3.8397e-05    161 0.0028194 0.27118 0.046357
    ## 147 3.6807e-05    162 0.0027810 0.27146 0.046355
    ## 148 3.6453e-05    163 0.0027442 0.27139 0.046357
    ## 149 3.6064e-05    164 0.0027078 0.27139 0.046357
    ## 150 3.4560e-05    165 0.0026717 0.27139 0.046357
    ## 151 3.4560e-05    166 0.0026371 0.27113 0.046356
    ## 152 3.4560e-05    167 0.0026026 0.27114 0.046356
    ## 153 3.4415e-05    168 0.0025680 0.27114 0.046356
    ## 154 3.4131e-05    169 0.0025336 0.27114 0.046356
    ## 155 3.1556e-05    170 0.0024995 0.27059 0.046348
    ## 156 3.1556e-05    171 0.0024679 0.27059 0.046348
    ## 157 3.1556e-05    172 0.0024364 0.27059 0.046348
    ## 158 3.1246e-05    173 0.0024048 0.27081 0.046369
    ## 159 3.0925e-05    174 0.0023736 0.27081 0.046369
    ## 160 3.0925e-05    175 0.0023426 0.27081 0.046369
    ## 161 3.0672e-05    176 0.0023117 0.27081 0.046369
    ## 162 3.0294e-05    177 0.0022810 0.27076 0.046370
    ## 163 3.0294e-05    178 0.0022507 0.27076 0.046370
    ## 164 2.9082e-05    179 0.0022205 0.27079 0.046369
    ## 165 2.4343e-05    180 0.0021914 0.27070 0.046363
    ## 166 2.4235e-05    181 0.0021670 0.27097 0.046363
    ## 167 2.3266e-05    182 0.0021428 0.27095 0.046363
    ## 168 2.0448e-05    183 0.0021195 0.27139 0.046361
    ## 169 2.0196e-05    184 0.0020991 0.27115 0.046364
    ## 170 2.0196e-05    185 0.0020789 0.27108 0.046364
    ## 171 1.8671e-05    186 0.0020587 0.27100 0.046365
    ## 172 1.8284e-05    187 0.0020400 0.27109 0.046364
    ## 173 1.7065e-05    188 0.0020217 0.27098 0.046365
    ## 174 1.6359e-05    189 0.0020047 0.27098 0.046365
    ## 175 1.6157e-05    191 0.0019720 0.27098 0.046365
    ## 176 1.4844e-05    192 0.0019558 0.27112 0.046364
    ## 177 1.2925e-05    193 0.0019410 0.27115 0.046364
    ## 178 1.2925e-05    194 0.0019280 0.27130 0.046363
    ## 179 1.1360e-05    195 0.0019151 0.27093 0.046214
    ## 180 1.1042e-05    196 0.0019037 0.27094 0.046214
    ## 181 1.1042e-05    197 0.0018927 0.27094 0.046214
    ## 182 1.0239e-05    198 0.0018817 0.27089 0.046214
    ## 183 9.8959e-06    199 0.0018714 0.27092 0.046214
    ## 184 8.5327e-06    202 0.0018417 0.27085 0.046213
    ## 185 8.1793e-06    203 0.0018332 0.27081 0.046212
    ## 186 7.2957e-06    204 0.0018250 0.27114 0.046221
    ## 187 7.2957e-06    205 0.0018177 0.27114 0.046221
    ## 188 7.2957e-06    206 0.0018104 0.27114 0.046221
    ## 189 7.2705e-06    207 0.0018031 0.27114 0.046221
    ## 190 7.2705e-06    208 0.0017959 0.27114 0.046221
    ## 191 7.2705e-06    209 0.0017886 0.27114 0.046221
    ## 192 6.3112e-06    210 0.0017813 0.27112 0.046223
    ## 193 6.1092e-06    211 0.0017750 0.27113 0.046223
    ## 194 6.1092e-06    212 0.0017689 0.27113 0.046223
    ## 195 5.6801e-06    213 0.0017628 0.27113 0.046223
    ## 196 5.0489e-06    214 0.0017571 0.27126 0.046222
    ## 197 5.0489e-06    215 0.0017521 0.27126 0.046222
    ## 198 5.0489e-06    217 0.0017420 0.27126 0.046222
    ## 199 4.9480e-06    218 0.0017369 0.27126 0.046222
    ## 200 4.2664e-06    219 0.0017320 0.27126 0.046222
    ## 201 4.2664e-06    220 0.0017277 0.27125 0.046222
    ## 202 4.0896e-06    221 0.0017234 0.27125 0.046222
    ## 203 4.0896e-06    222 0.0017193 0.27123 0.046222
    ## 204 3.7110e-06    223 0.0017153 0.27126 0.046223
    ## 205 3.6352e-06    224 0.0017115 0.27126 0.046223
    ## 206 3.2313e-06    225 0.0017079 0.27141 0.046228
    ## 207 2.7264e-06    226 0.0017047 0.27138 0.046228
    ## 208 2.4740e-06    228 0.0016992 0.27140 0.046228
    ## 209 2.4740e-06    229 0.0016967 0.27140 0.046228
    ## 210 2.4740e-06    230 0.0016943 0.27140 0.046228
    ## 211 2.4740e-06    232 0.0016893 0.27140 0.046228
    ## 212 1.8176e-06    234 0.0016844 0.27136 0.046228
    ## 213 1.8176e-06    235 0.0016826 0.27140 0.046228
    ## 214 1.2622e-06    237 0.0016789 0.27144 0.046228
    ## 215 1.2622e-06    238 0.0016777 0.27147 0.046227
    ## 216 1.2622e-06    241 0.0016739 0.27147 0.046227
    ## 217 1.2622e-06    242 0.0016726 0.27147 0.046227
    ## 218 8.0783e-07    243 0.0016714 0.27147 0.046227
    ## 219 8.0783e-07    245 0.0016697 0.27147 0.046227
    ## 220 4.5440e-07    246 0.0016689 0.27147 0.046227
    ## 221 4.5440e-07    248 0.0016680 0.27149 0.046227
    ## 222 2.0196e-07    249 0.0016676 0.27151 0.046227
    ## 223 1.0000e-08    250 0.0016674 0.27153 0.046227

It is probably better and easier to find this optimal value *programatically* as follows:

``` r
( b <- bos.to$cptable[which.min(bos.to$cptable[,"xerror"]),"CP"] )
```

    ## [1] 0.00937962

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

    ## [1] 18.96988

Again, it would be a **very good exercise** for you to compare the MSPE of the pruned tree with that of several of the alternative methods we have seen in class so far, **without using a training / test split**.

Note that pruning doesn't always improve a tree. For example, if we prune the first tree we fit in this example:

``` r
# what if we prune the original tree?
set.seed(123)
bos.t <- rpart(medv ~ ., data=dat.tr, method='anova')
b <- bos.t$cptable[which.min(bos.t$cptable[,"xerror"]),"CP"]
bos.t4 <- prune(bos.t, cp=b)
```

We obtain the same tree as before:

``` r
plot(bos.t4, uniform=FALSE, margin=0.01)
text(bos.t4, pretty=TRUE)
```

![](README_files/figure-markdown_github/prune10-1.png)

Below is the original tree:

``` r
plot(bos.t, uniform=FALSE, margin=0.01)
text(bos.t, pretty=TRUE)
```

![](README_files/figure-markdown_github/prune6-1.png)

#### Why is the pruned tree not a subtree of the "default" one?

Note that the pruned tree above is not a subtree of the one constructed using the default stopping criteria. The reason for this difference is that one of the default criteria is a limit on the minimum size of a child node. When we relaxed the criteria this limit was reduced and thus the "default" tree is not in fact a subtree of the large tree (that is later pruned). For example: ...

Pruning regression trees with `tree`
------------------------------------

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
bos.to2 <- tree(medv ~ ., data=dat.tr, control=tree.control(nobs=nrow(dat.tr), mincut=1, minsize=2, mindev=1e-10))
plot(bos.to2)
```

![](README_files/figure-markdown_github/prunetree2-1.png)

We now use the function `tree:cv.tree()` to estimate the MSPE of the subtrees of `bos.to2`, using 5-fold CV:

``` r
set.seed(123)
tt <- cv.tree(bos.to2, K = 5)
bos.pr2 <- prune.tree(bos.to2, k = tt$k[ which.min(tt$dev) ])
plot(bos.pr2); text(bos.pr2)
```

![](README_files/figure-markdown_github/prunetree3-1.png)

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
    ## (Intercept)  39.21  33.12
    ## crim         -0.13  -0.10
    ## zn            0.04   0.05
    ## indus         0.04  -0.01
    ## chas          2.72   2.80
    ## nox         -20.07 -14.18
    ## rm            3.45   4.15
    ## age           0.00   0.00
    ## dis          -1.44  -1.46
    ## rad           0.28   0.34
    ## tax          -0.01  -0.02
    ## ptratio      -1.01  -0.90
    ## black         0.01   0.01
    ## lstat        -0.56  -0.50

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
