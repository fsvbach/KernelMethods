#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:31:36 2021

@author: fsvbach
"""

import numpy as np

def linear(A,B):
    matrix = A@B.T
    return matrix

def gaussian(A,B, sigma=1):
    # % This is equivalent to computing the kernel on every pair of examples
    A1 = np.sum(A*A,1)
    B1 = np.sum(B*B,1)
    A2 = np.ones(len(A))
    B2 = np.ones(len(B))
    
    K1 = np.outer(A1,B2)
    K2 = np.outer(A2,B1)
    
    K0 = K1 + K2 - 2 * np.inner(A,B)
    K = np.exp(- K0 / sigma**2)
    return K