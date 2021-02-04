#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:04:39 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt


def plot_cross_val(scores, view=(0,1,2)):
    '''
    Parameters
    ----------
    scores : 3D-array
        containing the scores of cross-validation.
    view : TUPLE, optional
        defining the plot design by giving permutation of (0,1,2):
            - first variable will have own plot for each entry (default: 0 = model)
            - second variable will have own line for each entry (default: 1 = kernel)
            - third variable will define x-axis (default: 2 = dataset)
    '''
    
    labels = np.array(['model','kernel','dataset'])[list(view)]
    scores = scores.transpose(view)
    
    for mat in scores:
        
        for row in mat:
            plt.plot(row,label= labels[1])
            
        plt.title('Different Accuracies after Cross-Validation')
        # plt.xscale('log')
        plt.xlabel('labels[1]')
        plt.ylabel('percent')
        plt.legend()
        plt.savefig(f'Plots/{labels[0]}', dpi=300)
        plt.show()
    

tr, te = data.load_data()

# gauss = kernels.GaussianKernel(1)
# wdk   = kernels.WDKernel([0,0,1,1,1])
# sumk   = kernels.SumKernel([gauss,wdk],[1,1])
# wdk1   = kernels.WDKernel([0,0,0,1])
svm1  = models.SVM(5)
# svm2  = models.our_SVM(5)

spec1 = kernels.MismatchKernel(10,1)
spec2 = kernels.MismatchKernel(8,2)
spec3 = kernels.MismatchKernel(6,2)
spec4 = kernels.MismatchKernel(4,1)

models = [svm1]
kernels = [spec1,spec2,spec3,spec4]
datasets = tr[:1]

#S = util.cross_validation(models, kernels, datasets, D=5)

plot_cross_val(S, view=(0,2,1))

