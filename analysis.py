#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:40 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt


(train, test) = data.load_data()

zeros = []
mass = []

for i in range(1,16):
    betas = np.zeros(i)
    betas[-1] = 1
    wd=kernels.WDKernel(betas)
    M = wd.kernel_matrix(train[0],train[0])
    
    plt.imshow(np.log(M+0.00001))
    # plt.imshow(M)
    zeros.append( (np.count_nonzero(M)-len(M))/np.count_nonzero(M))
    mass.append( (M.sum()- M.diagonal().sum() )/M.sum() )
    plt.title(f'log matrix {i}')
    plt.colorbar()
    plt.savefig(f'Plots/log heatmap of k={i}')
    plt.show()
    
plt.plot(zeros, label='off diagonal nonzero-elements')
    
plt.plot(mass, label='off diagonal mass')
plt.title('Analysis of different k mer length without shifts')
plt.xlabel('length of kmers')
plt.ylabel('percent')
plt.legend()
plt.savefig(f'Plots/analysis of WDKernel')
plt.show()
