#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:50:55 2021

@author: fsvbach
"""

import os
import numpy as np
import pandas as pd

def load_train_labels():
    Y_train = []
    for i in range(3):
        Y = pd.read_csv(f'Data/Ytr{i}.csv', index_col=0)
        Y_train.append(np.array(Y['Bound']))
    return Y_train

def load_train_data():
    X_train = []
    for i in range(3):
        X = pd.read_csv(f'Data/Xtr{i}.csv', index_col=0)
        X_train.append(X['seq'])
    return X_train

def load_test_data():
    X_train = []
    for i in range(3):
        X = pd.read_csv(f'Data/Xte{i}.csv', index_col=0)
        X_train.append(X['seq'])
    return X_train

def save_predictions(Y_list, directory='predictions.csv'):
    '''
    Y_pred: list of three numpy arrays with 1000 labels each 
    '''
    Y_pred = pd.DataFrame(np.concatenate(Y_list), columns=['Bound'])
    index  = Y_pred.index.rename('Id')
    Y_pred = Y_pred.set_index(index)
    Y_pred.to_csv(directory)



if __name__ == '__main__':
    os.chdir("..")
