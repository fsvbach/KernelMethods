#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:35:55 2021

@author: fsvbach
"""

from Code.data import *
from Code.kernels import *
from Code.SVM import SVM

X = load_train_data()
Y = load_train_labels()
Z = load_test_data()

P = []
for x,y,z in zip(X,Y,Z):
    K_train = kernel_matrix(dumb_kernel, x, x)
    K_test  = kernel_matrix(dumb_kernel, z, x)
    
    model = SVM()
    pred = model.base_model(K_train, y, K_test)
    P.append( pred )
    
save_predictions(P)