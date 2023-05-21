# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:43:37 2023

@author: annaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from UBayFS import UBaymodel
from UBayFS_constraints import UBayconstraint
data, target = make_classification(random_state=1,  n_features=20)
#data = np.c_[ data, np.ones(100) ]  
#data = data +100
data = pd.DataFrame(data)

# regression
#data, target = make_regression(random_state=1)
#data = np.c_[ data, np.ones(100) ]  
#data = data +100
#data = pd.DataFrame(data)


#block_list = list()
#block_list.append(np.arange(0,50))
#block_list.append(np.arange(50,101))


A = np.array([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
A2 = np.array([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0], [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

b = np.array([1,2,3])
b1 = np.array([1])
rho = np.array([np.Inf, 0.5,np.Inf])
state = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
state1 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
state2 = np.ones(20)
c= UBayconstraint(A,b,rho)
c2 = UBayconstraint(rho=np.array([2]), constraint_types=["max_size", "must_link", "cannot_link", "jjj"], constraint_vars=[3, [1,3], [4,5,6], [7]], num_elements=10)


a = UBaymodel(data, target)
a.setConstraints(c)
a.setConstraints(c2, append=True)
a.sampleInitial(np.exp(a.posteriorExpectation()), 5)



features, feature_idx = a.train()

# sample initial
# a.sampleInitial(post_scores = np.exp(a.posteriorExpectation()), size=10)
