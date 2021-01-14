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

def Hyper_cross_validation(model, train, Kernel, parameter_range, name):
    
    for tr in train:
        scores = util.kernel_cross_validation(model, lambda params: Kernel(params), tr, parameter_range)
        plt.plot(scores, label=tr.name())
        print('Hyper Scores:',scores)
    
    plt.title('Different Accuracies after Cross-Validation (C=1)')
    plt.xlabel('~beta')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig(f'Plots/Hyperparameter_{name}', dpi=300)
    plt.show()
    
    
(train, test) = data.load_data()

# svm = models.SVM(C=100)
# sigma = 2.0**np.arange(-2,5)
# Hyper_cross_validation(svm, train, kernels.GaussianKernel, sigma, "Gaussian")

# # betas = np.arange(16)+2
# betas = np.array([ 0,0,0,1, 2, 3,  3,  2,  2, 1, 1, 1, 1, 1, 1, 1])
# print(len(betas))
#wd = kernels.WDKernel([0,0,0,1])
#wd.kernel_matrix(train[2], train[2])
# C_range = 10.0**np.arange(-2,4)
# SVM_cross_validation(wd, train[:1], C_range, "linear increase")

svm = models.SVM()
sk = kernels.SpectrumKernel(3)
#sk.plotMatrix(train[0], train[0])
Hyper_cross_validation(svm, train, kernels.SpectrumKernel, np.arange(2,8), "Spectrum")
#SVM_cross_validation(sk, train[:1], 10.0**np.arange(-2,4))

#w = kernels.WDShiftedKernel([0,0,0,1], 100)
# SVM_cross_validation(w, train[:1], 10.0**np.arange(-2,4))

# svm = models.SVM()
# # parameter_range = [ [4 + k * i for k in range(16)] for i in [1,2,3]]
# parameter_range = []
# for i in range(1,17):
#     betas = np.zeros(i)
#     betas[-1] = 1
#     parameter_range.append(betas)

# Hyper_cross_validation(svm, train[:1], kernels.WDKernel, parameter_range, "WDKernel")