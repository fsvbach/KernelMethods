#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:04:39 2021

@author: fsvbach
"""

from Code import kernels, util, models, data
import numpy as np
import pandas as pd

plot    = 1
predict = 0

### LOADING DATA ###

(training_data, test_data) = data.load_data()


### SAVE PREDICTIONS ###

# best models and kernels so far
model  = [models.SVM(C) for C in [6,        #67.1%
                                   13,      #72.1%
                                   6]]      #75.0%

kernel = [kernels.MismatchKernel(11, 2, 2),
            kernels.MismatchKernel(10, 2, 2),
            kernels.MismatchKernel(12, 2, 2)]

#save predictions
if predict:
    util.save_predictions(model, kernel, training_data, test_data)


### PLOT CROSS VAL ###

if plot:
    grid = {'model': [models.SVM(5), models.SVM(15)], 
            'kernel': [kernels.MismatchKernel(k,1,1) for k in np.arange(4,13)], 
            'dataset': training_data[:1]
            }

    view = {'title' : 'model',
            'legend': 'dataset',
            'xaxis' : 'kernel'
            }

    scores = util.cross_validation(grid, D=10)
    #scores = pd.read_csv('Plots/scores.csv', index_col=[0,1,2])
    
    util.plot_cross_val(scores, view)






