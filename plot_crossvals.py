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


def plot_cross_val(scores, grid, view=(0,1,2)):
    '''
    Parameters
    ----------
    scores : 3D-array
        containing the scores of cross-validation.
    view : TUPLE, optional
        defining the plot design by permutation of (model=0, kernel=1, dataset=2):
            - first entry: plot title
            - second entry: legend title
            - third entry: x-axis 
    '''
    
    x,y,z = np.array(['model','kernel','dataset'])[list(view)]
    scores = scores.transpose(view)
    
    for i,mat in enumerate(scores):
        
        for j,row in enumerate(mat):
            plt.plot(row, marker='o', linestyle='--', label= grid[y][j].name())
            
        plt.title(f'Cross-Validation for {x}: {grid[x][i].name()}')
        # plt.xscale('log')
        plt.xlabel(z)
        plt.xticks(range(len(row)), [a.name() for a in grid[z]], rotation='vertical')
        plt.ylabel('accuracy [%]')
        plt.legend(title=y)
        plt.savefig(f'Plots/CV_{x}_{grid[x][i].name()}', dpi=300)
        plt.show()

tr, te = data.load_data()

# gauss = kernels.GaussianKernel(1)
# wdk   = kernels.WDKernel([0,0,1,1,1])
# sumk   = kernels.SumKernel([gauss,wdk],[1,1])
# wdk1   = kernels.WDKernel([0,0,0,1])
svm1  = models.SVM(5)
# svm2  = models.our_SVM(5)

spec1 = kernels.MismatchKernel(11,1)
spec2 = kernels.MismatchKernel(9,1)
spec3 = kernels.MismatchKernel(9,0)
spec4 = kernels.MismatchKernel(4,1)

grid = {'model': [svm1], 'kernel': [spec1,spec2,spec3,spec4], 'dataset': tr[:1]}

S = util.cross_validation(grid, D=5)

plot_cross_val(S, grid, view=(2,0,1))

