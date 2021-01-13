import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt


def SVM_cross_validation(kernel, train, C_range):
    
    for tr in train:
        scores = util.model_cross_validation(lambda C: models.SVM(C), kernel, tr, C_range, D=5)
        plt.plot(C_range, scores, label=tr.name())
        print('C Scores:',scores)
    
    plt.title('Different Accuracies after Cross-Validation')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig(f'Plots/SVM_Cross_valid_{kernel.name()}', dpi=300)
    plt.show()

def Hyper_cross_validation(model, train, Kernel, parameter_range, name=""):
    
    for tr in train:
        scores = util.kernel_cross_validation(model, lambda params: Kernel(params), tr, parameter_range)
        plt.plot(scores, label=tr.name())
        print('Hyper Scores:',scores)
    
    plt.title('Different Accuracies after Cross-Validation (C=1)')
    plt.xlabel('~beta')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig(f'Plots/Hyperparameter_{name}_{model.name()}', dpi=300)
    plt.show()
    
    
(train, test) = data.load_data()

# svm = models.SVM(C=100)
# sigma = 2.0**np.arange(-2,5)
# Hyper_cross_validation(svm, train, kernels.GaussianKernel, sigma, "Gaussian")

# wd = WDKernel([0, 0, 0, 0, 1])
# C_range = 10.0**np.arange(-1,4)
# SVM_cross_validation(wd, train[:2], C_range)

# svm = SVM()
# parameter_range = [ [4 + k * i for k in range(16)] for i in [1,2,3]]
# # parameter_range = [ 
# Hyper_cross_validation(svm, train[:1], WDKernel, parameter_range, "WDKernel")