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


def plot_cross_val(scores, view):
    '''
    Parameters
    ----------
    scores : DataFrame
        containing the scores of cross-validation with MultiIndex
    view   : dict
        containing the plot-hierarchy
    '''
    title, legend, xaxis = view['title'], view['legend'], view['xaxis']
    
    for i,mat in scores.groupby(level=title):
        for j, row in mat.groupby(level=legend):
            plt.plot(row.values, marker='o', linestyle='--', label=j)
        plt.title(f"{title}: {i}")
        plt.xlabel(xaxis)
        plt.xticks(range(len(row.values)), row.index.get_level_values(xaxis) ,rotation='vertical')
        plt.legend(title=legend)
        plt.ylabel('accuracy [%]')
        plt.savefig(f"Plots/CV_{title}", dpi=300)
        plt.show()

tr, te = data.load_data()

gauss = kernels.GaussianKernel(1)
wdk   = kernels.WDKernel([0,0,1,1,1])
sumk   = kernels.SumKernel([gauss,wdk],[1,1])
wdk1   = kernels.WDKernel([0,0,0,1])

# svm1  = models.SVM(5)
# svm2  = models.our_SVM(5)

# spec1 = kernels.MismatchKernel(11,1)
# spec2 = kernels.MismatchKernel(9,1)
# spec3 = kernels.MismatchKernel(9,0)
# spec4 = kernels.MismatchKernel(4,1)

# k1 = kernels.MismatchKernelDirect(8, 2)
# k2 = kernels.MismatchKernel(9, 2, 1)
# k3 = kernels.MismatchKernel(10, 2, 1)

grid = {'model': [models.SVM(C) for C in 10.0**np.arange(-1, 2)], 
        'kernel': [gauss,wdk,sumk,wdk1], 
        'dataset': tr[:2]
        }

view = {'title' : 'dataset',
        'legend': 'kernel',
        'xaxis' : 'model'
        }

view = {'title' : 'model',
        'legend': 'dataset',
        'xaxis' : 'kernel'
        }

scores = util.cross_validation(grid, D=5)

plot_cross_val(scores, view)



