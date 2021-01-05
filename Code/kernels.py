#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:43:22 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

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
        for j,b in enumerate(B):
            matrix[i,j] = kernel(a,b) 
    return matrix
