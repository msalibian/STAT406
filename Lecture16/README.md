STAT406 - Lecture 16 notes
================
Matias Salibian-Barrera
2019-10-31

LICENSE
-------

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

The lecture slides are [here](STAT406-19-lecture-16.pdf).

#### Instability of trees

Just like in the regression case, classification trees can be highly unstable (specifically: relatively small changes in the training set may result in comparably large changes in the corresponding tree). We illustrate the problem on the very simple graduate school admissions example (3-class 2-dimensional covariates) we used in class. First read the data:

``` r
mm <- read.table('../Lecture14/T11-6.DAT', header=FALSE)
```

We transform the response variable `V3` into a factor (which is how class labels are represented in `R`, and what `rpart()` expects as the values in the response variable to build a classifier):

``` r
mm$V3 <- as.factor(mm$V3)
```

To obtain better looking plots later, we now re-scale one of the features (so both explanatory variables have similar ranges):

``` r
mm[,2] <- mm[,2] / 150
```

We use the function `rpart` to train a classification tree on these data, using deviance-based (`information`) splits:

``` r
library(rpart)
a.t <- rpart(V3~V1+V2, data=mm, method='class', parms=list(split='information'))
```

To illustrate the instability of this tree (i.e. how the tree changes when the data are perturbed slightly), we create a new training set (`mm2`) that is identical to the original one (in `mm`), except for two observations where we change their responses from class `1` to class `2`:

``` r
mm2 <- mm
mm2[1,3] <- 2
mm2[7,3] <- 2
```

The following plot contains the new training set, with the two changed observations (you can find them around the point `(GPA, GMAT) = (3, 4)`) highlighted with a blue dot (their new class) and a red ring around them (their old class was "red"):

``` r
plot(mm2[,1:2], pch=19, cex=1.5, col=c("red", "blue", "green")[mm2[,3]],
     xlab='GPA', 'GMAT', xlim=c(2,5), ylim=c(2,5))
points(mm[c(1,7),-3], pch='O', cex=1.1, col=c("red", "blue", "green")[mm[c(1,7),3]])
```

![](README_files/figure-markdown_github/inst2-1.png)

As we did above, we now train a classification tree on the perturbed data in `mm2`:

``` r
a2.t <- rpart(V3~V1+V2, data=mm2, method='class', parms=list(split='information'))
```

To visualize the differences between the two trees we build a fine grid of points and compare the predicted probabilities of each class on each point on the grid. First, construct the grid:

``` r
aa <- seq(2, 5, length=200)
bb <- seq(2, 5, length=200)
dd <- expand.grid(aa, bb)
names(dd) <- names(mm)[1:2]
```

Now, compute the estimated conditional probabilities of each of the 3 classes on each of the 40,000 points on the grid `dd`:

``` r
p.t <- predict(a.t, newdata=dd, type='prob')
p2.t <- predict(a2.t, newdata=dd, type='prob')
```

The next figures show the estimated probabilities of class "red" with each of the two trees:

``` r
filled.contour(aa, bb, matrix(p.t[,1], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)}, 
panel.last={points(mm[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm[,3]])})
```

![](README_files/figure-markdown_github/inst2.2-1.png)

``` r
filled.contour(aa, bb, matrix(p2.t[,1], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)},
panel.last={points(mm2[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm2[,3]]);
points(mm[c(1,7),-3], pch='O', cex=1.1, col=c("red", "blue", "green")[mm[c(1,7),3]])
})
```

![](README_files/figure-markdown_github/inst2.2-2.png)

Similarly, the estimated conditional probabilities for class "blue" at each point of the grid are:

``` r
# blues
filled.contour(aa, bb, matrix(p.t[,2], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)}, 
panel.last={ points(mm[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm[,3]])})
```

![](README_files/figure-markdown_github/kk-1.png)

``` r
filled.contour(aa, bb, matrix(p2.t[,2], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)},
pane.last={points(mm2[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm2[,3]]);
points(mm[c(1,7),-3], pch='O', cex=1.1, col=c("red", "blue", "green")[mm[c(1,7),3]])
})
```

![](README_files/figure-markdown_github/kk-2.png)

And finally, for class "green":

``` r
# greens
filled.contour(aa, bb, matrix(p.t[,3], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)}, panel.last={ points(mm[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm[,3]])})
```

![](README_files/figure-markdown_github/kk2-1.png)

``` r
filled.contour(aa, bb, matrix(p2.t[,3], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
plot.axes={axis(1); axis(2)},
pane.last={points(mm2[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm2[,3]]);
points(mm[c(1,7),-3], pch='O', cex=1.1, col=c("red", "blue", "green")[mm[c(1,7),3]])
})
```

![](README_files/figure-markdown_github/kk2-2.png)

Note that, for example, the regions of the feature space (the explanatory variables) that would be classified as "red" or "green" for the trees trained with the original and the slightly changed training sets change quite noticeably, even though the difference in the training sets is relatively small. Below we show how an ensemble of classifiers constructed via bagging can provide a more stable classifier.

Bagging
=======

Just as we did for regression, bagging consists of building an ensemble of predictors (in this case, classifiers) using bootstrap samples. If we using *B* bootstrap samples, we will construct *B* classifiers, and given a point **x**, we now have *B* estimated conditional probabilities for each of the possible *K* classes. Unlike what happens with regression problems, we now have a choice to make when deciding how to combine the *B* outputs for each point. We can take either: a majority vote over the *B* separate decisions, or we can average the *B* estimated probabilities for the *K* classes, to obtain bagged estimated conditional probabilities. As discussed and illustrated in class, the latter approach is usually preferred.

To illustrate the increased stability of bagged classification trees, we repeat the experiment above: we build an ensemble of 1000 classification trees trained on the original data, and a second ensemble (also of 1000 trees) using the slightly modified data. Each ensemble is constructed in exactly the same way we did in the regression case. For the first ensemble we train `NB = 1000` trees and store them in a list (called `ts`) for future use:

``` r
my.c <- rpart.control(minsplit=3, cp=1e-6, xval=10)
NB <- 1000
ts <- vector('list', NB)
set.seed(123)
n <- nrow(mm)
for(j in 1:NB) {
  ii <- sample(1:n, replace=TRUE)
  ts[[j]] <- rpart(V3~V1+V2, data=mm[ii,], method='class', parms=list(split='information'), control=my.c)
}
```

### Using the ensemble

As discussed in class, there are two possible ways to use this ensemble given a new observation: we can classify it to the class with most votes among the *B* bagged classifiers, or we can compute the average conditional probabilities over the *B* classifiers, and use this average as our esimated conditional probability. We illustrate both of these with the point `(GPA, GMAT) = (3.3, 3.0)`.

#### Majority vote

The simplest, but less elegant way to compute the votes for each class across the *B* trees in the ensemble is to loop over them and count:

``` r
x0 <- t( c(V1=3.3, V2=3.0) )
votes <- vector('numeric', 3)
names(votes) <- 1:3
for(j in 1:NB) {
  k <- predict(ts[[j]], newdata=data.frame(x0), type='class')
  votes[k] <- votes[k] + 1
}
(votes)
```

    ##   1   2   3 
    ## 909   0  91

And we see that the class most voted is 1.

The above calculation can be made more elegantly with the function `sapply` (or `lapply`):

``` r
votes2 <- sapply(ts, FUN=function(a, newx) predict(a, newdata=newx, type='class'), newx=data.frame(x0) )
table(votes2)
```

    ## votes2
    ##   1   2   3 
    ## 909   0  91

#### Average probabilities (over the ensemble)

If we wanted to compute the average of the conditional probabilities across the *B* different estimates, we could do it in a very similar way. Here I show how to do it using `sapply`. You are strongly encouraged to verify these calculations by computing the average of the conditional probabilities using a for-loop.

``` r
votes2 <- sapply(ts, FUN=function(a, newx) predict(a, newdata=newx, type='prob'), newx=data.frame(x0) )
( rowMeans(votes2) )
```

    ## [1] 0.90881555 0.00000000 0.09118445

And again, we see that class `1` has a much higher probability of occuring for this point.

### Increased stability of ensembles

To illustrate that ensembles of tree-based classifiers tend to be more stable than a single tree, we construct another example, but this time using the slightly modified data. The ensemble is stored in the list `ts2`:

``` r
mm2 <- mm
mm2[1,3] <- 2
mm2[7,3] <- 2
NB <- 1000
ts2 <- vector('list', NB)
set.seed(123)
n <- nrow(mm)
for(j in 1:NB) {
  ii <- sample(1:n, replace=TRUE)
  ts2[[j]] <- rpart(V3~V1+V2, data=mm2[ii,], method='class', parms=list(split='information'), control=my.c)
}
```

We use the same fine grid as before to show the estimated conditional probabilities, this time obtained with the two ensembles.

``` r
aa <- seq(2, 5, length=200)
bb <- seq(2, 5, length=200)
dd <- expand.grid(aa, bb)
names(dd) <- names(mm)[1:2]
```

To combine (average) the `NB = 1000` estimated probabilities of each of the 3 classes for each of the 40,000 points in the grid `dd` I use the function `vapply` and store the result in a 3-dimensional array. The averaged probabilities over the 1000 bagged trees can then obtained by averaging across the 3rd dimension. This approach may not be intuitively very clear at first sight. *You are strongly encouraged to ignore my code below and compute the bagged conditional probabilites for the 3 classes for each point in the grid in a way that is clear to you*. The main goal is to understand the method and be able to do it on your own. Efficient and / or elegant code can be written later, but it is not the focus of this course. The ensemble of trees trained with the original data:

``` r
pp0 <- vapply(ts, FUN=predict, FUN.VALUE=matrix(0, 200*200, 3), newdata=dd, type='prob')
pp <- apply(pp0, c(1, 2), mean)
```

And the ensemble of trees trained with the slightly modified data:

``` r
pp02 <- vapply(ts2, FUN=predict, FUN.VALUE=matrix(0, 200*200, 3), newdata=dd, type='prob')
pp2 <- apply(pp02, c(1, 2), mean)
```

The plots below show the estimated conditional probabilities for class "red" in each point of the grid, with each of the two ensembles. Note how similar they are (and contrast this with the results obtained before without bagging):

``` r
filled.contour(aa, bb, matrix(pp[,3], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
               plot.axes={axis(1); axis(2)},
                 panel.last={points(mm[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm[,3]])})
```

![](README_files/figure-markdown_github/bag1.plot-1.png)

``` r
filled.contour(aa, bb, matrix(pp2[,3], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT',
               plot.axes={axis(1); axis(2)},
                 panel.last={points(mm2[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm2[,3]]);
                 points(mm[c(1,7),-3], pch='O', cex=1.2, col=c("red", "blue", "green")[mm[c(1,7),3]])
               })
```

![](README_files/figure-markdown_github/bag1.plot-2.png)

You are strongly encouraged to obtain the corresponding plots comparing the estimated conditional probabilities with both ensembles for each of the other 2 classes ("blue" and "green").

Random Forests
==============

Even though using a *bagged* ensemble of trees usually results in a more stable predictor / classifier, a better ensemble can be improved by training each of its members in a careful way. The main idea is to try to reduce the (conditional) potential correlation among the predictions of the bagged trees, as discussed in class. Each of the bootstrap trees in the ensemble is grown using only a randomly selected set of features when partitioning each node. More specifically, at each node only a random subset of explanatory variables is considered to determine the optimal split. These randomly chosen features are selected independently at each node as the tree is being constructed.

To train a Random Forest in `R` we use the funtion `randomForest` from the package with the same name. The syntax is the same as that of `rpart`, but the tuning parameters for each of the *trees* in the *forest* are different from `rpart`. Refer to the help page if you need to modify them.

We load and prepare the admissions data as before:

``` r
mm <- read.table('../Lecture14/T11-6.DAT', header=FALSE)
mm$V3 <- as.factor(mm$V3)
mm[,2] <- mm[,2] / 150
```

and train a Random Forest with 500 trees and using all the default tuning parameters:

``` r
library(randomForest)
a.rf <- randomForest(V3~V1+V2, data=mm, ntree=500) 
```

Predictions can be obtained using the `predict` method, as usual, when you specify the `newdata` argument. Refer to the help page of `predict.randomForest` for details on the different behaviour of `predict` for Random Forest objects when the argument `newdata` is either present or missing.

To visualize the predicted classes obtained with a Random Forest on our example data, we compute the corresponding predicted conditional class probabilities on the same grid used before:

``` r
aa <- seq(2, 5, length=200)
bb <- seq(2, 5, length=200)
dd <- expand.grid(aa, bb)
names(dd) <- names(mm)[1:2]
```

The estimated conditional probabilities for class *red* are shown in the plot below (how are these estimated conditional probabilities computed exactly?)

``` r
pp.rf <- predict(a.rf, newdata=dd, type='prob')
filled.contour(aa, bb, matrix(pp.rf[,1], 200, 200), col=terrain.colors(20), xlab='GPA', ylab='GMAT', 
               plot.axes={axis(1); axis(2)},
               panel.last={points(mm[,-3], pch=19, cex=1.5, col=c("red", "blue", "green")[mm[,3]])}
               )
```

![](README_files/figure-markdown_github/rf1.1-1.png)

And the predicted conditional probabilities for the rest of the classes are: ![](README_files/figure-markdown_github/rf2-1.png)![](README_files/figure-markdown_github/rf2-2.png)

A very interesting exercise would be to train a Random Forest on the perturbed data (in `mm2`) and verify that the predicted conditional probabilities do not change much, as was the case for the bagged classifier.

### Another example

We will now use a more interesting example. The ISOLET data, available here: <http://archive.ics.uci.edu/ml/datasets/ISOLET>, contains data on sound recordings of 150 speakers saying each letter of the alphabet (twice). See the original source for more details. Since the full data set is rather large, here we only use the subset corresponding to the observations for the letters **C** and **Z**.

We first load the training and test data sets, and force the response variable to be categorical, so that the `R` implementations of the different predictors we will use below will build classifiers and not their regression counterparts:

``` r
xtr <- read.table('../Lecture15/isolet-train-c-z.data', sep=',')
xte <- read.table('../Lecture15/isolet-test-c-z.data', sep=',') 
xtr$V618 <- as.factor(xtr$V618)
xte$V618 <- as.factor(xte$V618)
```

To train a Random Forest we use the function `randomForest` in the package of the same name. The code underlying this package was originally written by Leo Breiman. We first train a Random Forest, using all the default parameters

``` r
library(randomForest)
set.seed(123)
( a.rf <- randomForest(V618 ~ ., data=xtr, ntree=500) )
```

    ## 
    ## Call:
    ##  randomForest(formula = V618 ~ ., data = xtr, ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 24
    ## 
    ##         OOB estimate of  error rate: 2.29%
    ## Confusion matrix:
    ##      3  26 class.error
    ## 3  234   6  0.02500000
    ## 26   5 235  0.02083333

We now check its performance on the test set:

``` r
p.rf <- predict(a.rf, newdata=xte, type='response')
table(p.rf, xte$V618)
```

    ##     
    ## p.rf  3 26
    ##   3  60  1
    ##   26  0 59

Note that the Random Forest only makes one mistake out of 120 (approx 0.8%) observations in the test set. However, the OOB error rate estimate is slightly over 2%. The next plot shows the evolution of the OOB error rate estimate as a function of the number of classifiers in the ensemble (trees in the forest). Note that 500 trees appears to be a reasonable forest size, in the sense thate the OOB error rate estimate is stable.

``` r
plot(a.rf, lwd=3, lty=1)
```

![](README_files/figure-markdown_github/rf0.oob-1.png)

<!-- Consider again the ISOLET data, available  -->
<!-- here:  -->
<!-- [http://archive.ics.uci.edu/ml/datasets/ISOLET](http://archive.ics.uci.edu/ml/datasets/ISOLET).  -->
<!-- Here we only use a subset  -->
<!-- corresponding to the observations for the letters **C** and **Z**.  -->
<!-- We first load the training and test data sets, and force the response  -->
<!-- variable to be categorical, so that the `R` implementations of the -->
<!-- different predictors we will use below will build  -->
<!-- classifiers and not their regression counterparts: -->
<!-- ```{r rf.isolet, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- xtr <- read.table('../Lecture15/isolet-train-c-z.data', sep=',') -->
<!-- xte <- read.table('../Lecture15/isolet-test-c-z.data', sep=',')  -->
<!-- xtr$V618 <- as.factor(xtr$V618) -->
<!-- xte$V618 <- as.factor(xte$V618) -->
<!-- ``` -->
<!-- To train a Random Forest we use the function `randomForest` in the -->
<!-- package of the same name. The code underlying this package was originally  -->
<!-- written by Leo Breiman. We train a RF leaving all -->
<!-- paramaters at their default values, and check  -->
<!-- its performance on the test set: -->
<!-- ```{r rf.isolet2, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- library(randomForest) -->
<!-- set.seed(123) -->
<!-- a.rf <- randomForest(V618 ~ ., data=xtr, ntree=500)  -->
<!-- p.rf <- predict(a.rf, newdata=xte, type='response') -->
<!-- table(p.rf, xte$V618) -->
<!-- ``` -->
<!-- Note that the Random Forest only makes one mistake out of 120 observations -->
<!-- in the test set. The OOB error rate estimate is slightly over 2%,  -->
<!-- and we see that 500 trees is a reasonable forest size: -->
<!-- ```{r rf.oob, fig.width=6, fig.height=6, message=FALSE, warning=FALSE} -->
<!-- plot(a.rf, lwd=3, lty=1) -->
<!-- a.rf -->
<!-- ``` -->
#### Using a test set instead of OBB

Given that in this case we do have a test set, we can use it to monitor the error rate (instead of using the OOB error estimates):

``` r
x.train <- model.matrix(V618 ~ ., data=xtr)
y.train <- xtr$V618
x.test <- model.matrix(V618 ~ ., data=xte)
y.test <- xte$V618
set.seed(123)
a.rf <- randomForest(x=x.train, y=y.train, xtest=x.test, ytest=y.test, ntree=500) 
test.err <- a.rf$test$err.rate
ma <- max(c(test.err))
plot(test.err[, 2], lwd=2, lty=1, col='red', type='l', ylim=c(0, max(c(0, ma))))
lines(test.err[, 3], lwd=2, lty=1, col='green')
lines(test.err[, 1], lwd=2, lty=1, col='black')
```

![](README_files/figure-markdown_github/rf.isolet.test-1.png)

According to the help page for the `plot` method for objects of class `randomForest`, the following plot should show both error rates (OOB plus those on the test set):

``` r
plot(a.rf, lwd=2)
```

![](README_files/figure-markdown_github/rf.isolet.test.plot-1.png)

#### Feature sequencing / Variable ranking

To explore which variables were used in the forest, and also, their importance rank as discussed in class, we can use the function `varImpPlot`:

``` r
varImpPlot(a.rf, n.var=20)
```

![](README_files/figure-markdown_github/rf.isolet3-1.png)

#### Comparing RF with other classifiers

We now compare the Random Forest with some of the other classifiers we saw in class, using their classification error rate on the test set as our comparison measure. We first start with K-NN:

``` r
library(class)
u1 <- knn(train=xtr[, -618], test=xte[, -618], cl=xtr[, 618], k = 1)
table(u1, xte$V618)
```

    ##     
    ## u1    3 26
    ##   3  57  9
    ##   26  3 51

``` r
u5 <- knn(train=xtr[, -618], test=xte[, -618], cl=xtr[, 618], k = 5)
table(u5, xte$V618)
```

    ##     
    ## u5    3 26
    ##   3  58  5
    ##   26  2 55

``` r
u10 <- knn(train=xtr[, -618], test=xte[, -618], cl=xtr[, 618], k = 10)
table(u10, xte$V618)
```

    ##     
    ## u10   3 26
    ##   3  58  6
    ##   26  2 54

``` r
u20 <- knn(train=xtr[, -618], test=xte[, -618], cl=xtr[, 618], k = 20)
table(u20, xte$V618)
```

    ##     
    ## u20   3 26
    ##   3  58  5
    ##   26  2 55

``` r
u50 <- knn(train=xtr[, -618], test=xte[, -618], cl=xtr[, 618], k = 50)
table(u50, xte$V618)
```

    ##     
    ## u50   3 26
    ##   3  58  7
    ##   26  2 53

To use logistic regression we first create a new variable that is 1 for the letter **C** and 0 for the letter **Z**, and use it as our response variable.

``` r
xtr$V619 <- as.numeric(xtr$V618==3)
d.glm <- glm(V619 ~ . - V618, data=xtr, family=binomial)
pr.glm <- as.numeric( predict(d.glm, newdata=xte, type='response') >  0.5 )
table(pr.glm, xte$V618)
```

    ##       
    ## pr.glm  3 26
    ##      0 25 33
    ##      1 35 27

Question for the reader: why do you think this classifier's performance is so disappointing?

It is interesting to see how a simple LDA classifier does:

``` r
library(MASS)
xtr$V619 <- NULL
d.lda <- lda(V618 ~ . , data=xtr)
pr.lda <- predict(d.lda, newdata=xte)$class
table(pr.lda, xte$V618)
```

    ##       
    ## pr.lda  3 26
    ##     3  58  3
    ##     26  2 57

Finally, note that a carefully built classification tree performs remarkably well, only using 3 features:

``` r
library(rpart)
my.c <- rpart.control(minsplit=5, cp=1e-8, xval=10)
set.seed(987)
a.tree <- rpart(V618 ~ ., data=xtr, method='class', parms=list(split='information'), control=my.c)
cp <- a.tree$cptable[which.min(a.tree$cptable[,"xerror"]),"CP"]
a.tp <- prune(a.tree, cp=cp)
p.t <- predict(a.tp, newdata=xte, type='vector')
table(p.t, xte$V618)
```

    ##    
    ## p.t  3 26
    ##   1 59  0
    ##   2  1 60

Finally, note that if you train a single classification tree with the default values for the stopping criterion tuning parameters, the tree also uses only 3 features, but its classification error rate on the test set is larger than that of the pruned one:

``` r
set.seed(987)
a2.tree <- rpart(V618 ~ ., data=xtr, method='class', parms=list(split='information'))
p2.t <- predict(a2.tree, newdata=xte, type='vector')
table(p2.t, xte$V618)
```

    ##     
    ## p2.t  3 26
    ##    1 57  2
    ##    2  3 58
