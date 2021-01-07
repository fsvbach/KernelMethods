#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:35:55 2021

@author: fsvbach
"""

import os
os.chdir("..")

import numpy as np
from Data.loader import load_train_labels
from Methods.SVM import SVM
from weighted_kernel.kernel import WDKernel
import matplotlib.pyplot as plt

Y = load_train_labels()
    
def create_crossval_plot():
    predictions = []
    cross_valid = 10.0**np.arange(-1,4)
    
    for x,y,z in zip(X,Y,Z):
        K_train = gaussian(x, x)    
        model  = SVM()
        scores = model.cross_validation(K_train, y, C_range=cross_valid)
        print(f'finished dataset: {scores}')
        predictions.append(scores)
        #K_test  = kernel_matrix(gaussian, z, x)
        
    for i in range(3):
        plt.plot(cross_valid,predictions[i], label=f'Dataset {i}')
    plt.title('Different Accuracies after Cross-Validation')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig('base_kernel/Cross Validation', dpi=300)
    plt.show()

# create_crossval_plot()