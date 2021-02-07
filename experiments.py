#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:04:39 2021

@author: fsvbach
"""

from Code import kernels, util, models, data
import numpy as np


### LOADING DATA ###

(training_data, test_data) = data.load_data()


### CREATE KERNEL MATRICES ###

# mkd = kernels.MismatchKernel(10, 2,1)
# K = mkd.plot_matrix(training_data[1], training_data[1])
# print(K)


### PLOT CROSS VAL ###

k3 = kernels.MismatchKernel(10, 2, 1)
k1 = kernels.MismatchKernelDirect(8, 2)
k2 = kernels.MismatchKernelDirect(10, 1)
k4 = kernels.MismatchKernel(11, 2, 1)

grid = {'model': [models.SVM(C) for C in 2*np.arange(4,9)], 
        'kernel': [k1,k2,k3,k4], 
        'dataset': training_data[1:2]
        }

view = {'title' : 'dataset',
        'legend': 'kernel',
        'xaxis' : 'model'
        }

scores = util.cross_validation(grid, D=10)

util.plot_cross_val(scores, view)


### SAVE PREDICTIONS ###

# # best models and kernels so far
# models  = [models.SVM(C) for C in [7,14,8]]
# kernels = [kernels.MismatchKernelDirect(K, M) for (K,M) in zip([10,10,11],[1,1,1])]

# #save predictions
# util.save_predictions(models, kernels, training_data, test_data)

