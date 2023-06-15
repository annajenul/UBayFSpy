Quickstart
==========
The UBayFS package implements the framework proposed in the article [Jenul et al. (2022)](https://link.springer.com/article/10.1007/s10994-022-06221-9), 
together with an interactive Shiny dashboard, which makes UBayFS applicable to R-users with different levels of expertise. UBayFS is an ensemble feature selection technique 
embedded in a Bayesian statistical framework. The method combines data and user knowledge, where the first is extracted via data-driven ensemble feature selection. 
The user can control the feature selection by assigning prior weights to features and penalizing specific feature combinations. 
In particular, the user can define a maximal number of selected features and must-link constraints (features must be selected together) or 
cannot-link constraints (features must not be selected together). Using relaxed constraints, a parameter $\rho$ regulates the penalty shape. 
Hence, violation of constraints can be valid but leads to a lower target value of the feature set that is derived from the violated constraints. 
UBayFS can be used for common feature selection and also for block feature selection.


Documentation
-------------
The following Jupyter notebook provides a `classification example <https://github.com/annajenul/UbayFSpy/blob/main/examples/classification%20example.ipynb>`_ , illustrating the UBayFS workflow. 

Classification example
----------------------
The following python example illustrates UBayFS on the Wisconsin breast cancer (classification) dataset, available from scikit-learn.
First, we load and prepare the data. Then we initialize a UBayFS model. This example shows
how to select features with UBayFS. For more examples including block feature selection have a look at the 
example notebooks on the UBayFS GitHub repository.

.. code-block:: python

    import pandas as pd
    import numpy as np
    import UBayFS

    from UBaymodel import UBaymodel
    from UBayconstraint import UBayconstraint

    data = pd.read_csv("./data/data.csv")
    labels = pd.read_csv("./data/labels.csv").replace(("M","B"),(0,1)).astype(int)

    model = UBaymodel(data=data,
                 target = labels,
                 feat_names = data.columns,
                 weights = [0.01],
                 M = 100, random_state=10,
                 method=[ "mrmr"],
                 nr_features = 20)

    constraints = UBayconstraint(rho=np.array([np.Inf, 0.1, 1, 1]), 
                             constraint_types=["max_size", "must_link", "cannot_link", "cannot_link","jjj"], 
                             constraint_vars=[10, [0,10,20], [0,9], [19,22,23]], 
                             num_elements=data.shape[1])
    model.setConstraints(constraints)
    model.train()

    