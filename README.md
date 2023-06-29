# UBayFS <img src="vignettes/logo.png" align="right" width="200"/>



The UBayFS package implements the framework proposed in the article [Jenul et al. (2022)](https://link.springer.com/article/10.1007/s10994-022-06221-9), together with an interactive Shiny dashboard, which makes UBayFS applicable to R-users with different levels of expertise. UBayFS is an ensemble feature selection technique embedded in a Bayesian statistical framework. The method combines data and user knowledge, where the first is extracted via data-driven ensemble feature selection. The user can control the feature selection by assigning prior weights to features and penalizing specific feature combinations. In particular, the user can define a maximal number of selected features and must-link constraints (features must be selected together) or cannot-link constraints (features must not be selected together). Using relaxed constraints, a parameter $\rho$ regulates the penalty shape. Hence, violation of constraints can be valid but leads to a lower target value of the feature set that is derived from the violated constraints. UBayFS can be used for common feature selection and also for block feature selection.

Documentation and Structure
---------------------------

A [documentation](https://annajenul.github.io/UBayFSpy/) illustrates how UBayFS can be used for standard feature selection 
