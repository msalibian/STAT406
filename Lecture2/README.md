STAT406 - Lecture 2 notes
================
Matias Salibian-Barrera
2018-09-05

#### LICENSE

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

The lecture slides will be here.

Predictions using a linear model
--------------------------------

In these notes we continue looking at the problem of comparing different models based on their prediction properties. As in the previous lecture, we consider a **full** and a **reduced** model, and in all that follows we assume that the variables included in the **reduced** model were not selected using the training data. **This seemingly innocent assumption is in fact critical, and we will later come back to it.**

### Estimating the MSPE with a test set

One way to estimate the mean squared prediction error of a model or predictor is to use it on a test set (where the responses are known, but that was not used when training the predcitor or estimating the model), and the comparing the predictions with the actual responses.

First, we load the training set and fit both models:

``` r
x.tr <- read.table('../Lecture1/pollution-train.dat', header=TRUE, sep=',')
full <- lm(MORT ~ . , data=x.tr)
reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
```

Although the **full** model fits the data better than the reduced one (see Lecture 1), its predictions on the test set are better. First, compute the two vectors of test set predictions:

``` r
x.te <- read.table('../Lecture1/pollution-test.dat', header=TRUE, sep=',')
pr.full <- predict(full, newdata=x.te)
pr.reduced <- predict(reduced, newdata=x.te)
```

And now, use them to estimate the mean squared prediction error of each model:

``` r
with(x.te, mean( (MORT - pr.full)^2 ))
```

    ## [1] 4677.45

``` r
with(x.te, mean( (MORT - pr.reduced)^2 ))
```

    ## [1] 1401.571

In Lecture 1 we also saw that this is not just an artifact of the specific training / test split of the data. The **reduced** model generally produces better predictions, regardless of the specific training / test split we use. We can verify this repeating the procedure many times (100, say) and looking at the estimated mean squared prediction errors obtained each time for each model.

![](README_files/figure-markdown_github/testrain-1.png)

### Leave-one-out cross-validation

A different procedure to estimate the prediction power of a model or method is called **leave-one-out CV**. One advantage of using this method is that the model we fit can use a larger training set. We discussed the procedure in class. Here we apply it to estimate the mean squared prediction error of the **full** and **reduced** models. Again, we assume that the reduced model was chosen independently from the training set.

``` r
x <- read.csv('../Lecture1/rutgers-lib-30861_CSV-1.csv')
n <- nrow(x)
pr.full <- pr.reduced <- rep(0, n)
for(i in 1:n) {
  full <- lm(MORT ~ . , data=x[-i, ])
  reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x[-i, ])
  pr.full[i] <- predict(full, newdata = x[i, ])
  pr.reduced[i] <- predict(reduced, newdata = x[i, ])
}
```

Now we have the leave-one-out predictions for each model and can compute the corresponding estimated mean squared prediction errors:

``` r
mean( (x$MORT - pr.full)^2 )
```

    ## [1] 2136.785

``` r
mean( (x$MORT - pr.reduced)^2 )
```

    ## [1] 1848.375

Note that here again the reduced model seems to yield better prediction errors.

### K-fold cross-validation

Leave-one-out cross-validation can be computationally very demanding (or even unfeasible) when the sample size is large and training the predictor is relatively costly. One solution is called **K-fold CV**. We split the data into **K** folds, train the predictor on the data without a fold, and use it to predict the responses in the removed fold. We cycle through the folds, and use the average of the squared prediction errors as an estimate of the mean squared prediction error. The following script does **5-fold CV** for the `full` and `reduced` linear models on the pollution dataset, once again assuming that the reduced model was originally chosen without using the data.

``` r
n <- nrow(x)
k <- 5
pr.full <- pr.reduced <- rep(0, n)
# Create labels for the "folds"
inds <- (1:n) %% k + 1 
# shuffle the rows of x, this is bad coding!
set.seed(123)
xs <- x[ sample(n, repl=FALSE), ]
# loop through the folds
for(j in 1:k) {
  x.tr <- xs[inds != j, ]
  x.te <- xs[inds == j, ]
  full <- lm(MORT ~ . , data=x.tr)
  reduced <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x.tr)
  pr.full[ inds== j] <- predict(full, newdata=x.te)
  pr.reduced[ inds==j ] <- predict(reduced, newdata=x.te)
}
```

We now compute the estimated mean squared prediction error of each model:

``` r
mean( (xs$MORT - pr.full)^2 )
```

    ## [1] 2328.806

``` r
mean( (xs$MORT - pr.reduced)^2 )
```

    ## [1] 1854.591

This method is clearly faster than leave-one-out CV, but the results may depend on the specific fold partition, and on the number **K** of folds used.

-   One way to obtain more stable mean squared prediction errors using K-fold CV is to repeat the above procedure many times, and compare the distribution of the mean squared prediction errors for each estimator. First, fit the **full** and **reduced** models using the whole data set as training:

``` r
m.f <- lm(MORT ~ . , data=x)
m.r <- lm(MORT ~ POOR + HC + NOX + HOUS + NONW, data=x)
```

We will use 50 runs of 5-fold CV comparing the **full** and **reduced** models. Again, here we assume that the reduced model was not obtained using the training data.

``` r
N <- 50
mspe1 <- mspe2 <- vector('double', N)
ii <- (1:(n <- nrow(x))) %% 5 + 1
set.seed(327)
for(i in 1:N) {
  ii <- sample(ii)
  pr.f <- pr.r <- vector('double', n)
  for(j in 1:5) {
    pr.f[ ii == j ] <- predict(update(m.f, data=x[ii != j, ]), newdata=x[ii==j,])
    pr.r[ ii == j ] <- predict(update(m.r, data=x[ii != j, ]), newdata=x[ii==j,])
  }
  mspe1[i] <- with(x, mean( (MORT - pr.f)^2 ))
  mspe2[i] <- with(x, mean( (MORT - pr.r)^2 ))
}  
boxplot(mspe1, mspe2, names=c('Full', 'Reduced'), 
        col=c('gray80', 'tomato3'), 
        main='Air Pollution - 50 runs 5-fold CV', ylim=c(0, 5000))
mtext(expression(hat(MSPE)), side=2, line=2.5)
```

![](README_files/figure-markdown_github/cv10runs-1.png)

Note that the estimated mean squared prediction error of the **reduced** model has a smaller mean / median than that of the **full** one. This tells us that the conclusion we reached favouring the reduced model (in terms of its prediction mean squared error) does not depend on a particular choice of folds. In other words, this provides more evidence to conclude that the reduced model will produce better predictions than the full one.

-   A computationally simpler (albeit possibly less precise) way to account for the K-fold variability is to run K-fold CV once and use the sample standard error of the **K** *smaller* mean squared prediction errors to construct a rough *confidence interval* around the overall mean squared prediction error estimate (that is the average of the mean squared prediction errors over the K folds).

-   The dependency of this MSPE on **K** is more involved. We will discuss it later.
