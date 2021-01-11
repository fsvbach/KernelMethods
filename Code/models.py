#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:05:19 2021

@author: fsvbach
"""
from abc import ABC, abstractmethod #abstract classes
import numpy as np
from sklearn.svm import SVC

np.random.seed(8364)

class Model(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, train_matrix, labels):
        pass

    @abstractmethod
    def predict(self, test_matrix):
        pass

    def fit_and_predict(self, kernel, train, test):
        K_train = kernel.kernel_matrix(train, train)
        K_test = kernel.kernel_matrix(test, train)
        self.fit(K_train, train.labels())
        return self.predict(K_test)


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
                
                
                
                
                
                
           