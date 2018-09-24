# STAT406 - "Elements of Statistical Learning"

Public repository for STAT406 @ UBC - "Elements of Statistical Learning".


#### LICENSE
The notes in this repository are released under the "Creative Commons Attribution-ShareAlike 4.0 International" license. See the **human-readable version** [here](https://creativecommons.org/licenses/by-sa/4.0/) and the **real thing** [here](https://creativecommons.org/licenses/by-sa/4.0/legalcode).

## Course outline
The course outline is available [here](STAT406-18-19-MSB.pdf).

## Tentative weekly schedule (including Quizzes and Midterms)
The tentative week-by-week schedule is [here](Weekly-schedule-18-19-detailed-2.pdf).

## PIAZZA
You can register in the course's [PIAZZA](https://www.piazza.com) page via
[Canvas](https://canvas.ubc.ca).

## WebWork
In order to complete the WebWork quizzes you need to register via
[Canvas](https://canvas.ubc.ca): go to the course Canvas page, click on
*Assignments*, then on *WebWork Link*, and finally click on *Load WebWork Link on a new window*.
This is a **necessary** step (don't shoot the messenger!) but you only
need to do this **once**.

## Weekly reading and other resources
This is a list of **strongly** recommended **pre-class** reading. **[JWHT13]**
and **[HTF09]** indicate two of the reference books listed below.

* Week 1 (L1): Review of Linear Regression
	* Sections 2.1, 2.1.1, 2.1.2, 2.1.3, 2.2, 2.2.1 from [JWHT13]
	* Sections 2.4 and 2.6 from [HTF09].
* Week 2 (L2/3): Goodness of Fit vs Prediction error, Cross Validation
	* Sections 5.1, 5.1.1, 5.1.2, 5.1.3 from [JWHT13]
	* Sections 7.1, 7.2, 7.3, 7.10 from [HTF09].
* Week 3 (L4/5): Correlated predictors, Feature selection, AIC
	* Sections 6.1, 6.1.1, 6.1.2, 6.1.3, 6.2 and 6.2.1 from [JWHT13]
	* Sections 7.4, 7.5 from [HTF09].
* Week 4 (L6/MT1): Ridge regression, LASSO, Elastic Net
	* Sections 6.2 (complete) from [JWHT13]
	* Sections 3.4, 3.8, 3.8.1, 3.8.2 from [HTF09]
* Week 5 (L7/8): Elastic Net, Smoothers (Local regression, Splines)
	* Sections 7.1, 7.3, 7.4, 7.5, 7.6 from [JWHT13]
	<!--
* Week 6 (L10/11): Curse of dimensionality, Regression Trees
	* Sections 8.1, 8.1.1, 8.1.3, 8.1.4 from [JWHT13]
* Week 7 (L12/13): Bagging, Classification, LDA, Logistic Regression
	* Sections 8.2, 8.2.1, 4.1, 4.2 from [JWHT13]
* Week 8 (L14/15): LDA, LQA, Nearest Neighbours, Trees
	* Section 4.4, 4.3, 2.2.3, 8.1.2 from [JWHT13]
* Week 9 (L16/17): Ensembles, Bagging, Random Forests
	* Sections 8.2.1 and 8.2.2 from [JWHT13]
* Week 10 (L18/19): Boosting, Neural Networks?
	* Sections 8.2.3 from [JWHT13]
	* Sections 10.1 - 10.10 (except 10.7), 11.3 - 11.5, 11.7 from [HTF09]
* Week 11 (L20/21): Unsupervised learning, K-means, model-based clustering
	* Sections 10.3 from [JWHT13]
	* Sections 13.2, 14.3 from [HTF09]
* Week 12 (L22/23): EM-algorith, Hierarchical clustering
	* Sections 10.3 from [JWHT13]
	* Sections 8.5, 14.3 from [HTF09]
* Week 13 (L24/25): Principal Components, Multidimensional Scaling
	* Sections 10.2 from [JWHT13]
	* Sections 14.5.1, 14.8, 14.9 from [HTF09] -->

## Reference books
* **[JWHT13]**: James, G., Witten, D., Hastie, T. and Tibshirani, R.
An Introduction to Statistical Learning. 2013. Springer-Verlag New York
	* [Book page](http://www-bcf.usc.edu/~gareth/ISL/), [Book PDF](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)

* **[HTF09]**: Hastie, T., Tibshirani, R. and Friedman, J.
The Elements of Statistical Learning. 2009. Second Edition. Springer-Verlag New York
	* [Book page](http://web.stanford.edu/~hastie/ElemStatLearn), [Book PDF](https://web.stanford.edu/~hastie/ElemStatLearn/download.html)

* **[MASS]**: Venables, W.N. and Ripley, B.D.
Modern Applied Statistics with S. 2002. Fourth edition, Springer, New York.
	* [Book page](https://www.stats.ox.ac.uk/pub/MASS4/), [Publisher page](http://link.springer.com/book/10.1007%2F978-0-387-21706-2)


## Useful tools
- [R](http://www.cran.r-project.org/): This is the software we will use in the course. I will assume that you are familiar with it (in particular, that you know how to write **your own functions** and **loops**). If needed, there are plenty of resources on line to learn R.
- [RStudio](https://www.rstudio.com/products/RStudio/): The IDE (integrated development environment) of choice for R. Not necessary, but helpful.
- [Jupyter Notebooks](https://jupyter.org). "The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text."
You can use these to interactively run and play with the lecture notes and the code to reproduce all the examples I use in class. This is not necessary, but may be helpful. There are two options to run notebooks: locally on your own computer or use a remote server:
  1. Follow the instructions
[here](https://jupyter.org/install.html) to install Jupyter on your laptop. You will also need to follow [these instructions](https://www.datacamp.com/community/blog/jupyter-notebook-r) to install the `R kernel` for Jupyter.
  2. Alternatively, you can run the notebooks on the [syzygy](https://ubc.syzygy.ca/) server. There are Julia, Python 2, Python 3, and R kernels available (although we will only use the R one). Sign in with your UBC CWL. Once you are logged in, use [this link](https://ubc.syzygy.ca/jupyter/user-redirect/git-pull?repo=https://github.com/msalibian/STAT406) to clone this repository (STAT406) (including all notebooks) directly onto your [syzygy](https://ubc.syzygy.ca/) home directory. You may need to do this regularly throughout the Term.
