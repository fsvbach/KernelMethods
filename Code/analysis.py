#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:40 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd 

from kernels import *
from util import *
from models import *
from data import load_data
import matplotlib.pyplot as plt


os.chdir('/home/fsvbach/Unikurse/2-MSIAM Kernel Methods/Data Challenge')

(train, test) = load_data()


betas = np.arange(15)
    
wd = WDKernel(betas)
M = wd.kernel_matrix(train[0], train[0])
print(M)

