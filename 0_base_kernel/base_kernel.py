#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:35:55 2021

@author: fsvbach
"""

import os
os.chdir("..")

from Code.data import *
from Code.kernels import *
from Code.SVM import SVM

Y = load_train_labels()
X = []
for i in range(3):
    x = pd.read_csv(f'Data/Xtr{i}_mat100.csv', header=None, sep=' ')
    X.append(np.array(x))
Z = []
for i in range(3):
    z = pd.read_csv(f'Data/Xte{i}_mat100.csv', header=None, sep=' ')
    Z.append(np.array(z))
    
def linear(A,B):
    print(A.shape)
    print(B.shape)
    matrix = A@B.T
    print(matrix.shape)
    return matrix

predictions = []
cross_valid = []

for x,y,z in zip(X,Y,Z):
    K_train = linear(x, x)
    #K_test  = kernel_matrix(gaussian, z, x)
    
    model  = SVM()
    scores = model.cross_validation(K_train, y, C_range=10.0**np.arange(4))

    print(scores)
    #print('ready')
    
#save_predictions(P)
