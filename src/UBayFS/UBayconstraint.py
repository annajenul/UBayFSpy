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
import math
import sys
from itertools import chain

class UBayconstraint():
    """
    This class initializes user-defined constraints.
    
    PARAMETERS
    -----
    rho: <numpy array> 
        Vector of regularization strenghts for the defined constraints.
    A: <numpy array> 
        Matrix describing constraints. Left side of the equation system. Default: ``A=None``
    b : <numpy array>
        1-d dimensional array defining the right sight of the equation system. Default: ``b=None``
    block_matrix : <numpy array>
        Matrix describing the block assignment for each feature. If no block-structure given, block_matrix is a diagonal-unity matrix. Default : ``block_matrix=None``.
    block_list : <list>
        List describing the block assignment for each feature, if block structure is present. Default : ``block_list=None``.
    constraint_types : <list> of <strings>
        Constraint types. Possible options are:
            - "max_size" : maximal number of features that shall be selected 
            - "must_link" : the defined feature set must be selected together
            - "cannot_link" : the defined feature set must not be selected together  
    constraint_vars : <list> of <int> or <lists>
        For each constraint_type, define features sets that are concerned. Default : ```constraint_vars=None``.
    num_elements : <int>
        Total number of features. Default :``num_elements=None``. 
    """
    
    def __init__(self, rho, A=None, b=None, block_matrix=None, block_list=None,
                 constraint_types=None, constraint_vars=None, num_elements=None):
        
        direct_var_setting = (A is not None) and (b is not None)
        indirect_var_setting = (constraint_types is not None) and (constraint_vars is not None) and (num_elements is not None)

        if direct_var_setting and indirect_var_setting:
            sys.exit("Constraints must be defined direnctly or indirectly but not both!")
        
        rho_single = (len(rho) == 1)
        
        if direct_var_setting == True:
            if np.any(rho <=0):
                sys.exit("rho values must be >0")
            
            if len(A) != len(b) != len(rho):
                sys.exit("Constraint dimensions do not fit!")
            
            
                
            self.A = A
            self.b = b
            
            if rho_single:
                self.rho = np.repeat(rho, self.A.shape[0])
            else:
                self.rho = rho
            
        else:
            
            self.A = np.empty((0,num_elements), int)
            self.b = np.empty(0)
            self.rho = np.empty(0)
            
            # check if all constraint types in max, must, cannot
            
            # len constraint types must be len constraint vars
            
            def max_size(smax):
                self.A = np.append(self.A, np.ones((1,num_elements)), axis=0)
                self.b = np.append(self.b, smax)
                
                
            def must_link(sel):
                if len(sel) > 1:
                    pairs = [(x, y) for x in sel for y in sel if x != y]
                    for pair in pairs:
                        new_row = 1*np.array((np.arange(0,num_elements) == pair[0])) - \
                    1*np.array((np.arange(0,num_elements) == pair[1]))
                        new_row = new_row.reshape(1, len(new_row))
                        self.A = np.append(self.A, new_row, axis=0)
                        self.b = np.append(self.b, 0)

            def cannot_link(sel):
                if len(sel) > 1:
                    new_row = np.zeros((1,num_elements))
                    new_row[:,sel] = 1
                    self.A = np.append(self.A, new_row, axis=0)
                    self.b = np.append(self.b, 1)
                    
            # iterate over constraints
            
            for i, (cv, ct) in enumerate(zip(constraint_vars, constraint_types)):
                if ct == "max_size":
                    max_size(cv)
                    if rho_single:
                        self.rho = np.append(self.rho, rho[0])
                    else:
                        self.rho = np.append(self.rho, rho[i])
                elif ct == "must_link":
                    must_link(cv)
                    N = len(cv)
                    combinations = math.factorial(N) / math.factorial((N-2))
                    if rho_single:
                        self.rho = np.append(self.rho, np.repeat(rho[0],combinations))
                    else:
                        self.rho = np.append(self.rho, np.repeat(rho[i],combinations))
                elif ct == "cannot_link":
                    cannot_link(cv)
                    if rho_single:
                        self.rho = np.append(self.rho, rho[0])
                    else:
                        self.rho = np.append(self.rho, rho[i])
                else:
                    print("The constraint type '", ct, "' is unknown.")

            
        if (block_matrix is None) and (block_list is None):
            block_matrix = np.identity(np.shape(self.A)[1])
            self.block_matrix = block_matrix  
            
        elif (block_matrix is None) and (block_list is not None):
            
            block_matrix = np.zeros((len(block_list), (max(list(chain.from_iterable(block_list)))+1)))
            for i in range(len(block_list)):
                block_matrix[i,block_list[i]] = 1
            self.block_matrix = block_matrix
        else:
            self.block_matrix = block_matrix
         
        
    def get_dimensions(self):
        """
        Get the dimensions of ...?
           
        Returns
        -----
        ...
        """
        return np.array([np.shape(self.A)[0], np.shape(self.block_matrix)[1]])
        
    def group_admissibility(self, state, log=True):
        """
        Evaluate the value of the admissibility function 'kappa' for a group of constraints (with a common block)-

        PARAMETERS
        -----
        state: <np.array>
            1-dimensional binary array describing a feature set. 1: feature selected, 0: feature not selected.
        log : <boolean>
            Indicates whether the admissibility should be returned on log scale.
        
        Returns
        -----
        An admissibility value <float>.
        """
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
    
    def get_maxsize(self):
        """
        Get the right side (b) of the max size constraint.
        
        Returns
        -----
        An integer.
        """
        ms = None
        
        if np.array_equal(self.block_matrix, np.identity(np.shape(self.A)[1])):
            for j in range(len(self.A)):
                if np.array_equal(self.A[j,:], np.ones(len(self.A[j,:]))):
                    ms = self.b[j]
        return ms
        
    def get_constraints(self):
        """
        Get the constraints.
        
        Returns
        -----
        A dictionary including A, b, and block_matrix.
        """
        return {"A":self.A, "b": self.b, "rho": self.rho, "block_matrix": self.block_matrix}
       