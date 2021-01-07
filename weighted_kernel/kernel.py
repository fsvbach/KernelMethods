#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:43:22 2021

@author: fsvbach
"""

import numpy as np
from Data.loader import *

class WDKernel:
    
    def __init__(self):
        X = load_train_data()
        
        Z = load_test_data()
        
        d=5
        beta = np.arange(d)
        beta = 2*(d-beta+1)/(d*(d+1))

    def dumb_kernel(seq1, seq2):
        if seq1 == seq2:
            return 1
        return 0

    def kernel_matrix(kernel, A, B):
        '''
        Parameters
        ----------
        kernel: function
                name of kernel function to use
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
                matrix[i,j] = kernel(a,b) 
        return matrix


        
    def weighted_kernel(seq1, seq2, k):
        assert len(seq1) == len(seq2)
        
        count = 0
        L = len(seq1)
        for l in range(0,L-k+1):
            if seq1[l:l+k] == seq2[l:l+k]:
                count += 1
        return count   


