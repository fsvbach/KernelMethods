#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:05:19 2021

@author: fsvbach
"""
from abc import ABC, abstractmethod #abstract classes
import numpy as np

from sklearn.svm import SVC
from cvxopt import solvers, matrix

np.random.seed(8364)

class Model(ABC):

    @abstractmethod
    def name(self):
        '''
        Create model with paramenters.
        '''
        pass

    @abstractmethod
    def fit(self, train_matrix, labels):
        '''
        fit model to training data (as kernel matrix)
        '''
        pass

    @abstractmethod
    def predict(self, test_matrix):
        '''
        predict labels for data (given as train x test matrix)
        '''
        pass

    def fit_and_predict(self, kernel, train, test):
        '''
        shorthand for fit and predict (takes a kernel object)
        '''
        K_train = kernel.kernel_matrix(train, train)
        K_test = kernel.kernel_matrix(test, train)
        self.fit(K_train, train.labels())
        return self.predict(K_test)


class our_SVM(Model):
    
    def __init__(self, C=1.0):
        self._name  = f'SVM_C={C}' 
        self.C      = C
        self.alphas = None
        
    def fit(self, K, y):
        y = 1.0*y
        G = np.vstack((-np.diag(y), np.diag(y)))
        y_r = y.reshape(len(y),1)
        h = np.vstack((np.zeros_like(y_r), np.zeros_like(y_r)+self.C))
        # print(y.shape)
        # print(K.shape)
        # print(G.shape)
        # print(h.shape)
        # input()
        solution    = solvers.qp(matrix(K), matrix(y*1.0), matrix(G), matrix(h*1.0))
        self.alphas = np.array(solution['x'])
        
    def predict(self, K):
        assert self.alphas   
        return np.sign( K @ self.alphas ) 
    
    def name(self):
        return self._name


class SVM(Model):
    
    def __init__(self, C=1):
        self._name = f'SVM_C={C}' 
        self.model = SVC(kernel='precomputed', C=C)
        
    def fit(self, train_matrix, labels):
        self.model.fit(train_matrix, labels)

    def predict(self, test_matrix):
        assert self.model.fit_status_ == 0
        return self.model.predict(test_matrix)
    
    def name(self):
        return self._name
                
                
                
                
                
                
           