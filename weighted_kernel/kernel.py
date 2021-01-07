#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:43:22 2021

@author: fsvbach
"""

import os
os.chdir("..")

import numpy as np
from Data.loader import *


def dumb_kernel(seq1, seq2):
    if seq1 == seq2:
        return 1
    return 0


class WDKernel:
    
    def __init__(self, folder='Storage', kappa=5, delta=0):
        self.folder = folder
        self.kappa = kappa
        self.delta = delta

    def kernel_matrix(self, A, B):
        '''
        Parameters
        ----------
        A : DataFrame 
            contains M strings of len L
        B : DataFrame
            contains N strings of len L
    
        Returns
        -------
        Kernel Matrix of size MxN
        '''
        matrix = np.zeros((len(A),len(B)))
        for i,a in enumerate(A):
            print(i)
            for j,b in enumerate(B):
                matrix[i,j] = self.kernel_function(a,b) 
        return matrix
        
    def compute(self):
        X = load_train_data()
        Z = load_test_data()
        
        for i,(x,z) in enumerate(zip(X,Z)):
            K_train = self.kernel_matrix(x,x)
            K_test  = self.kernel_matrix(z,x)
            np.save(f'weighted_kernel/{self.folder}/Ktr{i}.npy', K_train)
            np.save(f'weighted_kernel/{self.folder}/Kte{i}.npy', K_test)
    
    def load(self):
        
        X=[]
        Z=[]
        for i in range(3):
            K_train = np.load(f'weighted_kernel/{self.folder}/Ktr{i}.npy')
            K_test  = np.load(f'weighted_kernel/{self.folder}/Kte{i}.npy')
            X.append(K_train)
            Z.append(K_test)
        
        return X,Z   
        
    def kernel_function(self, seq1, seq2):
        '''
        Parameters
        ----------
        seq1 : sequence 1
            DESCRIPTION.
        seq2 : sequence 2
            DESCRIPTION.
        k : scalar
            length of substring.

        Returns
        -------
        count : scalar
            kernel value.

        '''
        count = 0
        L = len(seq1)
        for l in range(0,L-self.kappa+1):
            if seq1[l:l+self.kappa] == seq2[l:l+self.kappa]:
                count += 1
        return count   

