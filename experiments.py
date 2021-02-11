#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:04:39 2021

@author: fsvbach
"""

from Code import kernels, util, models, data
import numpy as np
import pandas as pd

### LOADING DATA ###

# create data loader
(training_data, test_data) = data.load_data()


### SAVE PREDICTIONS ###

# best models and kernels so far
model  = [models.SVM(C) for C in [4.5,        #67.1%
                                   13.5,      #72.6%
                                   6]]      #74.7%

kernel = [kernels.MismatchKernel(10, 2, 2),
            kernels.MismatchKernel(10, 2, 2),
            kernels.MismatchKernel(12, 2, 2)]

#save predictions
util.save_predictions(model, kernel, training_data, test_data)


### PLOT CROSS VAL ###

grid = {'model': [models.SVM(5), models.SVM(15)], 
        'kernel': [kernels.MismatchKernel(k,1,1) for k in np.arange(4,13)], 
        'dataset': training_data[:1]
        }

view = {'title' : 'model',
        'legend': 'dataset',
        'xaxis' : 'kernel'
        }

# # compute scores
# scores = util.cross_validation(grid, D=10)

# # # if already computed
# # scores = pd.read_csv('Plots/scores.csv', index_col=[0,1,2])

# # do the plot
# util.plot_cross_val(scores, view)






