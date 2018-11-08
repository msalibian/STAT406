STAT406 - Lecture 18 notes
================
Matias Salibian-Barrera
2018-11-08

LICENSE
-------

These notes are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

Lecture slides
--------------

Preliminary lecture slides will be here.

What is Adaboost doing, *really*?
---------------------------------

We have seen in class that Adaboost can be thought of as fitting an *additive model* in a stepwise (greedy) way, using an exponential loss. It is then easy to prove that Adaboost.M1 is computing an approximation to the *optimal classifier* G( x ) = log\[ P( Y = 1 | X = x ) / P( Y = -1 | X = x ) \] / 2. More specifically, Adaboost.M1 is fitting an additive model to that function, in other words is attempting to find functions *f*<sub>1</sub>, *f*<sub>2</sub>, ..., *f*<sub>*N*</sub> such that *G*(*x*)=∑<sub>*i*</sub>*f*<sub>*i*</sub>(*x*).

Knowing what function the boosting algorithm is approximating (albeit in a greedy and suboptimal way), allows us to
understand when the algorithm is expected to work well, and also when it may not work well. In particular, it provides one way to choose the complexity of the *weak lerners* used to construct the ensemble. For an example you can refer to the corresponding lab activity.

### A more challenging example, the `email spam` data

The email spam data set is a relatively classic data set containing 57 features (potentially explanatory variables) measured on 4601 email messages. The goal is to predict whether an email is *spam* or not. The 57 features are a mix of continuous and discrete variables. More information can be found at <https://archive.ics.uci.edu/ml/datasets/spambase>.

We first load the data and randomly separate it into a training and a test set. A more thorough analysis would be to use *full* K-fold cross-validation, but given the computational complexity, I decided to leave the rest of this 3-fold CV exercise to the reader.

``` r
data(spam, package='ElemStatLearn')
n <- nrow(spam)
set.seed(987)
ii <- sample(n, floor(n/3))
spam.te <- spam[ii, ]
spam.tr <- spam[-ii, ]
```

We now use Adaboost with 500 iterations, using *stumps* as our weak learners / classifiers, and check the performance on the test set:

``` r
library(adabag)
onesplit <- rpart.control(cp=-1, maxdepth=1, minsplit=0, xval=0)
bo1 <- boosting(spam ~ ., data=spam.tr, boos=FALSE, mfinal=500, control=onesplit)
pr1 <- predict(bo1, newdata=spam.te)
table(spam.te$spam, pr1$class) # (pr1$confusion)
```

    ##        
    ##         email spam
    ##   email   879   39
    ##   spam     55  560

The classification error rate on the test set is rather high. We now compare it with that of a Random Forest:

``` r
library(randomForest)
set.seed(123) 
a <- randomForest(spam ~ . , data=spam.tr) # , ntree=500)
plot(a)
```

![](README_files/figure-markdown_github/spam.3-1.png)

``` r
pr.rf <- predict(a, newdata=spam.te, type='response')
table(spam.te$spam, pr.rf)
```

    ##        pr.rf
    ##         email spam
    ##   email   888   30
    ##   spam     53  562

The number of trees in the random forest seems to be appropriate, and its performance on this test set is definitively better than that of boosting (the estimated classification error rate of the latter using this test set is 0.0613177, while for the Random Forest is 0.0541422 on the test set and 0.0488918 using OOB).

Is there *any room for improvement* for Adaboost? As we discussed in class, depending on the interactions that may be present in the *true classification function*, we might be able to improve our boosting classifier by slightly increasing the complexity of our base ensemble members. Here we try to use 3-split classification trees, instead of the 1-split ones used above:

``` r
threesplits <- rpart.control(cp=-1, maxdepth=3, minsplit=0, xval=0)
bo3 <- boosting(spam ~ ., data=spam.tr, boos=FALSE, mfinal=500, control=threesplits)
pr3 <- predict(bo3, newdata=spam.te)
(pr3$confusion)
```

    ##                Observed Class
    ## Predicted Class email spam
    ##           email   886   36
    ##           spam     32  579

The number of element on the boosting ensemble appears to be appropriate:

``` r
plot(errorevol(bo3, newdata=spam.te))
```

![](README_files/figure-markdown_github/spam.5-1.png)

There is, in fact, a noticeable improvement in performance on this test set. The estimated classification error rate of AdaBoost using 3-split trees on this test set is 0.0443575. Recall that the estimated classification error rate for the Random Forest was 0.0541422 (or 0.0488918 using OOB).

As mentioned above you are strongly encouraged to finish this analysis by doing a complete K-fold CV analysis in order to compare boosting with random forests on these data.

Gradient boosting
-----------------

Discussed in class.

Neural Networks
---------------

Discussed in class.

#### An example with a simple neural network

This example using the ISOLET data illustrates the use of simple neural networks, and also highlights some issues of which it may be important to be aware.

#### Additional resources for discussion (refer to the lecture for context)

-   <https://arxiv.org/abs/1412.6572>
-   <https://arxiv.org/abs/1312.6199>
-   <https://www.axios.com/ai-pioneer-advocates-starting-over-2485537027.html>
-   <https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085e>
