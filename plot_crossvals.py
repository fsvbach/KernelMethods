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


def plot_cross_val(scores, fix=0, x=1, y=2):
    
    scores = scores.transpose(fix,x,y)
    
    for view in scores:
        
        for row in view:
            plt.plot(row)
            
        plt.title('Different Accuracies after Cross-Validation')
        # plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('percent')
        plt.legend()
        plt.savefig(f'Plots/CV', dpi=300)
        plt.show()
    
    # return scores

tr, te = data.load_data()

gauss = kernels.GaussianKernel(1)
wdk   = kernels.WDKernel([0,0,1,1,1])
sumk   = kernels.SumKernel([gauss,wdk],[1,1])
wdk1   = kernels.WDKernel([0,0,0,1])
svm1  = models.SVM(5)
svm2  = models.our_SVM(5)

S = util.cross_validation([svm1,svm2], [gauss,wdk,sumk,wdk1], tr[:1], D=3)
plot_cross_val(S)
