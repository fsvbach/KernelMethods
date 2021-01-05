#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:05:19 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC

class SVM:
    def __init__(self):
        pass        
        
    def base_model(self, K_train, y_train, K_test):
        clf = SVC(kernel='precomputed')
    
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        
        return y_pred