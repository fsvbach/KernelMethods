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

# k4 = kernels.MismatchKernelDirect(10, 1)
# # k5 = kernels.MismatchKernelDirect(11, 0)

# grid = {'model': [models.SVM(C) for C in np.arange(14.5,20.5,0.5)], 
#         'kernel': [k4], 
#         'dataset': training_data[1:2]
#         }

# view = {'title' : 'dataset',
#         'legend': 'kernel',
#         'xaxis' : 'model'
#         }

# scores = util.cross_validation(grid, D=10)

# util.plot_cross_val(scores, view)


### SAVE PREDICTIONS ###

# # best models and kernels so far
# models  = [models.SVM(C) for C in [7,16.5,2.25]]
# kernels = [kernels.MismatchKernelDirect(8, 2),
#             kernels.MismatchKernelDirect(10, 1),
#             kernels.MismatchKernel(11, 2, 1)]

# #save predictions
# util.save_predictions(models, kernels, training_data, test_data)

