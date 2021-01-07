#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:23:33 2021

@author: fsvbach
"""

import os
#os.chdir("..")

import numpy as np
from Data.loader import *
from weighted_kernel.kernel import WDKernel
from Methods.SVM import SVM


Y = load_train_labels()
K = WDKernel()
Ktr, Kte = K.load()


predictions = []
    
for x,y,z in zip(Ktr,Y,Kte):
    norm = np.max(x)
    x /= norm
    z /= norm
    
    model  = SVM()
    model.fit(x, y, C=1)
    predictions.append(model.predict(z))
    print(f'finished dataset')

save_predictions(predictions, directory='weighted_kernel/predictions.csv')
    
