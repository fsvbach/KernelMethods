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
from weighted_kernel.kernel import WDKernel
from Methods.SVM import SVM


Y = load_train_labels()

predictions = []

for x,y,z in zip(X,Y,Z):
    K_train = gaussian(x, x)   
    K_test  = gaussian(z, x)
    model  = SVM()
    model.fit(K_train, y, C=1000)
    predictions.append(model.predict(K_test))
    print(f'finished dataset')

save_predictions(predictions, directory='base_kernel/predictions.csv')
    
