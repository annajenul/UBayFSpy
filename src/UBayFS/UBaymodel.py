# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:55:40 2023

@author: annaj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample
from sklearn.feature_selection import SelectKBest, chi2
from skfeature.function.similarity_based import fisher_score
import mrmr
import sys
from scipy.special import logsumexp
from pygad import GA


# import from own files
from UBayconstraint import UBayconstraint


class UBaymodel():
    """
    The constructor initializes common variables of UBaymodel.
    
    PARAMETERS
    -----
    data: <numpy array> or <pandas dataframe>
        Dataset on which feature selection shall be performed. 
        Variable types must be numeric or integer.
    target: <numpy array> or <pandas dataframe>
        Response variable of data.
    feat_names : <list>
        List holding feature names. Preferably a list of string values. 
        If empty, feature names will be generated automatically. 
        Default: ``feat_names=[]``.
    M : <int>
        Number of unique train-test splits. Default ``K=100``.
    tt_split : <float>
        traint test split. Default ``tt_split=0.75``.
    nr_features : <string or int>
        Set a random state to reproduce your results. Default: ``string="auto"``.
            - ``string="auto"`` : .... 
            - ``int`` : .       
    """
    
    def __init__(self, data, target, feat_names = [], M=100, tt_split=0.75, 
                 nr_features="auto",
                 method=["mrmr"], prior_model="dirichlet", weights=[1], 
                 constraints=None, l=1, optim_method="GA", popsize=100, maxiter=100):
        
        
        self.data = pd.DataFrame(data)
        self.target = target
        self.M = M
        self.tt_split = tt_split
        self.method = method
        self.prior_model = prior_model
        self.weights = weights
        self.l = l
        self.optim_method = optim_method
        self.popsize = popsize
        self.maxiter = maxiter
        
        
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints
        
        # catch errors
        if self.data.isnull().values.any():
            sys.exit("Error: NA values not supported!")
        if len(self.data) != len(self.target):
            sys.exit("Error: number of labels must match number of data rows!") 
        if (self.M % 1 != 0) or (self.M <= 0):
            sys.exit("Error: M must be a positive integer")
        if (self.tt_split < 0) or (self.tt_split >1):
            sys.exit("Error: tt_split should not be outs")
        if (self.tt_split < 0.5) or (self.tt_split > 0.99):
            print("Warning: tt_split should not be outside [0.5,0.99]")
        if not ((isinstance(self.l, int)) or (isinstance(self.l, float))):
            sys.exit("Error: l must be a positive scalar!")
        if self.l <= 0:
            sys.exit("Error: l must be a positive scalar!")
            
            
        # binary classification or regression
        self.binary = np.array_equal(self.target, self.target.astype(bool))
        
        # other useful variables
        self.nrow, self.ncol = np.shape(data)
        
        if len(feat_names) == 0:
            self.feat_names = ['f' + str(ind) for ind in range(self.ncol)]
        else:
            self.feat_names = feat_names
        
        self.data.columns = self.feat_names
        
        if len(self.weights) == 1:
            self.weights = np.repeat(self.weights, self.ncol)
        
        self.ensemble_matrix = pd.DataFrame(columns=self.feat_names)
        #method_names = method[!fs_vs_string]
        
        for i in range(self.M):
            if self.binary == True:
                train_data, test_data,train_labels, test_labels = train_test_split(data, target, 
                                                           train_size=self.tt_split, stratify=target)
            else:
                train_data, test_data,train_labels, test_labels = train_test_split(data, target, 
                                                           train_size=self.tt_split)
            # non constant columns
            nconst_cols = np.where(train_data.nunique() != 1)[0]
            train_data = train_data.iloc[:,nconst_cols]
            
            # number of features
            if nr_features == "auto":
                self.nr_features = sample(list(np.arange(1,self.ncol)),1)[0]
            else:
                self.nr_features = nr_features
                
                
            for m in self.method:
                
                try:
                
                    if m in ["mRMR", "mrmr"]:
                        if self.binary:
                            ranks = mrmr.mrmr_classif(train_data, train_labels, 
                                                      self.nr_features)
                        else:
                            ranks = mrmr.mrmr_regression(train_data, train_labels, 
                                                      self.nr_features)
                        name="mrmr"
                        
                    if m in ["chi"]:
                        chi2_f = SelectKBest(chi2, k=self.nr_features)
                        chi2_f.fit_transform(train_data, train_labels)
                        ranks = chi2_f.get_feature_names_out()
                        name="chi"
                    if m in["fisher", "Fisher"]:
                        if self.binary:
                            ranks = fisher_score.fisher_score(train_data.values, 
                                                              train_labels)[:self.nr_features]
                            ranks = [self.feat_names[i] for i in ranks]
                            name="fisher"
                        else:
                            sys.exit("Fisher score not usable for regression problems!")
                        
                    vec = pd.DataFrame(columns=self.feat_names)
                    vec.loc[0] = np.repeat(0, self.ncol)
                    vec.loc[:,ranks] = 1
                    vec.index = [name + "_" + str(i)]
                except:
                    print("method not working for in this iteration...")
                    vec = pd.DataFrame(columns=self.feat_names)
                    vec.loc[0] = np.repeat(np.nan, self.ncol)
                    print(name + "_" + str(i))
                    vec.index = [name + "_" + str(i)]
                
                self.ensemble_matrix = pd.concat([self.ensemble_matrix,
                                                  vec], ignore_index=False)
        
        self.ensemble_matrix = self.ensemble_matrix.dropna()
        
        if np.ceil(len(self.ensemble_matrix) / len(self.method)) < np.ceil(self.M / 2):
            sys.exit("Too many ensembles could not be performed!")
        
        # structure results
        self.counts = pd.Series(np.sum(self.ensemble_matrix, axis=0))
        
        
    def setWeights(self, weights, block_list=None, block_matrix=None):
        
        if (len(weights) >1) and (len(weights) != self.ncol):
            sys.exit("Error: length of prior weights does not match data matrix")
            
        
        if len(weights) == 1:
            weights = np.repeat(weights, self.ncol)
        
        if any(weights) <= 0:
            sys.exit("Error: weights must be positive")
            
        if (block_matrix is not None) or (block_list is not None):
            if block_matrix is None:
                block_matrix = np.zeros((len(block_list), self.ncol))
            for i in range(len(block_list)):
                block_matrix[i,block_list[i]] = 1
            weights = np.matmul(np.transpose(block_matrix), weights.reshape(-1,1))
            
            if np.shape(block_matrix)[0] != len(weights):
                sys.exit("Error: wrong length of weights vector: must match number of blocks, if block_matrix or block_list are provided")
            self.weights = weights
        else:
            self.weights = weights
            
        self.block_matrix = block_matrix
        
    def getWeights(self):
        return self.weights
    
    
                
    def setOptim(self, method, popsize, maxiter):
        # check if method is empty
        self.method = method
        self.popsize = popsize
        self.maxiter = maxiter
        
    def getOptim(self):
        return {"method":self.method, "popsize":self.popsize, "maxiter":self.maxiter}
        
    def setConstraints(self, constraints, append=False):
        
        if constraints.get_dimensions()[1] != self.ncol:
            sys.exit("Dimensions of constraints do not match")
        
        if append:
            # check if block matrix already present
            bm_appearance = [np.array_equal(constraints.block_matrix, i.block_matrix) for i in self.constraints]
            if sum(bm_appearance) > 0:
                index = int(np.where(bm_appearance)[0])
                self.constraints[index].A = np.append(self.constraints[index].A, constraints.A, axis=0)
                self.constraints[index].b = np.append(self.constraints[index].b, constraints.b)
                self.constraints[index].rho = np.append(self.constraints[index].rho, constraints.rho)
            else:
                self.constraints = self.constraints + [constraints]
        else:
            self.constraints = [constraints]
            
    def getConstraints(self):
        
        constraints = {}
        for i, constraint in enumerate(self.constraints):
            constraints[i] = {"A":constraint.A, "b":constraint.b, "rho": constraint.rho, "block_matrix":constraint.block_matrix}
        return constraints
            
        
    def admissibility(self, state, log=True):
        
        adm = 1-log

        for i in self.constraints:
            if log:
                adm = adm + i.group_admissibility(state)
            else:
                adm = adm * i.group_admissibility(state)
        return adm
        
    def posteriorExpectation(self):
        
        post_scores = self.counts.values.astype(int) + self.weights
        post_scores = np.log(post_scores) - np.log(np.sum(post_scores))
        return post_scores
        
        
    def train(self):
        
        # check if any constraint present:
        if len(self.constraints) == 0:
            sys.exit("At least a max-size constraint must be present for training!")
        
        
        theta = self.posteriorExpectation()
        
        def neg_loss(state):
            return logsumexp(np.array(theta[state==1] + [np.log(self.l) + self.admissibility(state)]))
        
        def fitness_fun(ga_instance, solution, solution_idx):
            return neg_loss(solution)
        
        x_start = self.sampleInitial(post_scores = np.exp(theta), size=self.popsize)
        ga_instance = GA(num_generations = self.maxiter,
                   num_parents_mating = self.popsize,
                   fitness_func = fitness_fun,
                   initial_population = x_start,
                   gene_type=int,
                   init_range_high=1,
                   init_range_low=0
                   )
        
        x_optim, x_optim_fitness, _ = ga_instance.best_solution()
        
        
        return  pd.DataFrame(x_optim, index=self.feat_names), list(np.array(self.feat_names)[np.where(x_optim ==1)[0]])
        
    def sampleInitial(self, post_scores, size):
        
        n = len(post_scores)
        num_constraints_per_block = [const.get_dimensions()[0] for const in self.constraints]
        cum_num_constraints_per_block = np.array([0] + list(np.cumsum(num_constraints_per_block)))
        rho = np.concatenate([const.rho for const in self.constraints])
        rho = 1 / (1 + rho)
        
        def full_admissibility(state, constraint_dropout, log=True):
            
            active_constraints = np.where(constraint_dropout == 1)[0]
            res = 1-log
            
            for i in range(len(self.constraints)):
                
                active_constraints_in_block = \
                    active_constraints[np.where([np.sum(j >= cum_num_constraints_per_block) == (i+1) for j in active_constraints])[0]] - \
                    cum_num_constraints_per_block[i]
                
                if len(active_constraints_in_block) > 0:
                    constraint_new = UBayconstraint(rho=self.constraints[i].rho[active_constraints_in_block],
                                                    A=self.constraints[i].A[active_constraints_in_block,:],
                                                    b=self.constraints[i].b[active_constraints_in_block],
                                                    block_matrix=self.constraints[i].block_matrix)
                        
                    a = constraint_new.group_admissibility(state, log=log)
                    res = res + a if log else res * a
            return res
          
        
        def order_features(order):
            
            rho_mat = np.column_stack((rho, 1-rho))
            constraint_dropout = np.array([])
            for i in range(rho_mat.shape[0]):
                probs = rho_mat[i,:]
                constraint_dropout = np.append(constraint_dropout, \
                                               [np.random.choice([0,1],size=1,replace=True,p=probs)[0]])
            
            x = np.zeros(n)
            for i in range(n):
                x_new = x.copy()
                x_new[order[i]] = 1
                if full_admissibility(state=x_new, constraint_dropout=constraint_dropout,log=False) == 1:
                    x = x_new             
            return x
        
        # apply along feature_orders
        
        feature_orders = [] 
        for i in range((size)):
            feature_orders.append(np.random.choice(range(self.ncol), size=self.ncol, replace=False, p=post_scores))
    
        feature_orders = np.transpose(np.stack(feature_orders)) 
        x_start = np.transpose(np.apply_along_axis(order_features, 0, feature_orders))
        
        
        # always add feature set with best scores
        
        ms = []
        for i in self.constraints:
            ms_c = i.get_maxsize()
            if ms_c is not None:
                ms.append(ms_c)
                
        if (len(ms) == 1) and (ms[0] > 0):
            ms = int(ms[0])
            ms_sel = (-post_scores).argsort()[:ms]
            add_x = np.zeros(x_start.shape[1])
            add_x[ms_sel] = 1
            x_start = np.vstack([x_start, add_x])
            self.x_start = x_start
        else:
            sys.exit("No max-size constraint!")
            
        return x_start
            
        
    def evaluateFS(self, state, method="spearman", log=False):
        
        results = {}
        # correlation
        if np.sum(state) >1:
            c = np.abs(self.data.iloc[:,state==1].corr(method=method)).values
            average_feature_correlation = np.round((np.sum(c) - np.sum(np.diag(c))) / (np.sum(state) * (np.sum(state)-1)),3)
        else:
            c = None
            average_feature_correlation = None
        
        
        # posterior scores
        post_scores = self.posteriorExpectation()
        log_post = logsumexp(post_scores[state == 1])
        
        neg_loss = np.exp(logsumexp(np.array(post_scores[state==1] + [np.log(self.l) + self.admissibility(state)]))) - \
            self.l
        if log:
            neg_loss = np.log(neg_loss)
            
        # calculate number of violated constraints
        num_violated_constraints = 0
        for constraint in self.constraints:
            num_violated_constraints +=  \
            np.sum(np.matmul(constraint.A, np.matmul(constraint.block_matrix, state)>0) > constraint.b)
            
        # calculate output metrics
        results["cardinality"] = np.sum(state)
        results["total utility"] = np.round(neg_loss,3)
        results["posterior feature utility"] = np.round(log_post, 3) if log else np.round(np.exp(log_post),3)
        results["admissibility"] = np.round(self.admissibility(state,log=log),3)
        results["number of violated constraints"] = num_violated_constraints
        results["average feature correlation"] = average_feature_correlation
        
        return results
                   
                
                    
                    
        