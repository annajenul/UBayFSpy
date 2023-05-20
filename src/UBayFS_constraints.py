# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:18:17 2023

@author: annaj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample
from sklearn.feature_selection import SelectKBest, chi2
from skfeature.function.similarity_based import fisher_score
from scipy.special import logsumexp
import mrmr
import sys

class UBayconstraint():
    
    def __init__(self, A, b, rho, block_matrix=None):
        
        
        if block_matrix is None:
            block_matrix = np.identity(np.shape(A)[1])
            
        self.A = A
        self.b = b
        self.rho = rho
        self.block_matrix = block_matrix
        
    def get_dimensions(self):
        return np.array([np.shape(self.A)[0], np.shape(self.block_matrix)[1]])
        
    def group_admissibility(self, state, log=True):
        
        if not len(state) == self.get_dimensions()[1]:
            sys.exit("Wrong size of state!")
            
        state = np.matmul(self.block_matrix, state) > 0
        
        ind_inf = np.where(self.rho == np.inf)[0]
        ind_non_inf = np.where(self.rho != np.inf)[0]
        
        # case 1: rho < Inf
        if len(ind_non_inf) > 0:
           const_not_fulfilled =  np.where(self.b[ind_non_inf] - \
           np.matmul(self.A[ind_non_inf,:], state) < 0)[0]
    
           z = self.b[ind_non_inf] - \
           np.matmul(self.A[ind_non_inf,:], state) * self.rho[ind_non_inf]
           
           lprob1 = np.log(2) + z - np.apply_along_axis(logsumexp, 1, np.column_stack((z,np.zeros(len(z)))))
           lprob1 = np.sum(lprob1[const_not_fulfilled])
        else:
           lprob1 = 0
           
        if len(ind_inf) > 0:
            z = self.b[ind_inf] - \
            np.matmul(self.A[ind_inf,:], state) >= 0
            lprob2 = np.sum(np.log(z))
        else:
            lprob2 = 0
           
        if log:
            return lprob1 + lprob2
        else:
            return np.exp(lprob1 + lprob2)
        
           
       