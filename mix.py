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

tr, te = data.load_data()

X, y_train = tr[0], tr[0].labels()

# util.save_predictions(svm, kernels.GaussianKernel(0.5), tr, te)
# why doesnt it work?

gauss = kernels.GaussianKernel(0.5)
wdk   = kernels.WDKernel([0,0,1,1])

K1 =  gauss.kernel_matrix(X,X)
K2 = wdk.kernel_matrix(X,X)

plt.imshow(K1)
plt.colorbar()
plt.show()

plt.imshow(K2)
plt.colorbar()
plt.show()

a = 1
b = 1

K_train = a*K1 + b*K2

plt.imshow(K_train)
plt.colorbar()
plt.show()

D=10
parameter_range = 2.0**np.arange(-2,5)

N          = len(y_train)
valid_size = int(N / D)
indices    = np.arange(N)
np.random.shuffle(indices)
indices = np.split(indices[:D*valid_size], D)

scores = []
for params in parameter_range:
    score = 0
    model = models.SVM(params)
    for idx in indices:
        yte = y_train[idx]
        ytr = np.delete(y_train, idx)
        K   = np.delete(K_train, idx, 1)
        Kte = K[idx]
        Ktr = np.delete(K, idx, 0)
        model.fit(Ktr, ytr)
        pred = model.predict(Kte)
        score += util.accuracy(pred, yte) / D
    scores.append(score)

plt.plot(parameter_range, scores)
plt.show()



