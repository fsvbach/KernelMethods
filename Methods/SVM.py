#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:05:19 2021

@author: fsvbach
"""

import numpy as np
from sklearn.svm import SVC

np.random.seed(8364)
    
class SVM:
    
    def __init__(self):
        self.model = None
        
    def fit(self, K_train, y_train, C):
        
        self.model = SVC(kernel='precomputed', C=C)
        self.model.fit(K_train, y_train)
        
    def predict(self, K_test):
        
        assert self.model 
        y_pred = self.model.predict(K_test)
        
        return y_pred
    
    def accuracy(self, y_pred, y_true):
        assert len(y_pred) == len(y_true)
        return 1 - np.linalg.norm(y_pred-y_true,ord=1)/len(y_true)

    def cross_validation(self, K_train, y_train, C_range=10.0**np.arange(-2,2), D=10):
        '''
        Parameters
        ----------
        K_train : NxN matrix
            Kernel train matrix.
        y_train : Nx1 vector
            Train labels.
        C_range : TYPE, optional
            values for C to test. The default is 10.0**np.arange(-2,2).
        D : scalar, optional
            D-fold cross validation. The default is 10.

        Returns
        -------
        accuracy for each C in C_range
        '''
        
        scores = []
        
        N          = len(y_train)
        valid_size = int(N / D)
        indices    = np.arange(N)
        np.random.shuffle(indices)
        indices = np.split(indices[:D*valid_size], D)
                
        for C in C_range:
            
            score = 0
            
            for idx in indices:
                
                yte = y_train[idx]
                ytr = np.delete(y_train, idx)

                K   = np.delete(K_train, idx, 1)
                Kte = K[idx]
                Ktr = np.delete(K, idx, 0)
                
                self.fit(Ktr, ytr, C)
                pred = self.predict(Kte)           
                score += self.accuracy(pred,yte)/D
            
            scores.append(score)
            print(f'{C}: {score}')
        
        return scores
                
                
                
                
                
                
                
                
           