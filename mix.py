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


def plot_cross_val(models, kernels, datasets, fix=0, D=5):
    
    scores = util.cross_validation(models, kernels, datasets, D=D)
    
    plt.plot(scores)
    plt.title('Different Accuracies after Cross-Validation')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig(f'Plots/CV', dpi=300)
    plt.show()
    
    return scores



tr, te = data.load_data()

gauss = kernels.GaussianKernel(0.5)
wdk   = kernels.WDKernel([0,0,1,1])
svm1  = models.SVM(1)
svm2  = models.SVM(5)

S = util.cross_validation([svm1,svm2], [gauss,wdk], tr)

