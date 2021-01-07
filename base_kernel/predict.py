#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:23:33 2021

@author: fsvbach
"""

import os
os.chdir("..")

import numpy as np
from Data.loader import *
from base_kernel.kernels import gaussian
from Methods.SVM import SVM

Z = load_test_data_mat()
Y = load_train_labels()
X = load_train_data_mat()

predictions = []

for x,y,z in zip(X,Y,Z):
    K_train = gaussian(x, x)   
    K_test  = gaussian(z, x)
    model  = SVM()
    model.fit(K_train, y, C=1000)
    predictions.append(model.predict(K_test))
    print(f'finished dataset')

save_predictions(predictions, directory='base_kernel/predictions.csv')
    
