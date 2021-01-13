import numpy as np
import pandas as pd 

from .kernels import *
from .util import *
from .models import *
from .data import load_data
import matplotlib.pyplot as plt


def SVM_C_cross_validation(kernel, train):
    C_range = 10.0**np.arange(-1,4)

    for tr in train:
        scores = model_cross_validation(lambda C: SVM(C), kernel, tr, C_range, D=5)
        plt.plot(C_range, scores, label=tr.name())

    plt.title('Different Accuracies after Cross-Validation')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('percent')
    plt.legend()
    plt.savefig(f'Plots/SVM_Cross_valid_{kernel.name()}', dpi=300)
    plt.show()

(train, test) = load_data()

svm = SVM()

gaussian = GaussianKernel(1.1)
linear = LinearKernel()
wd = WDKernel([0, 0, 0, 0, 1])
#save_predictions(svm, wd, train, test)

SVM_C_cross_validation(wd, train[2:3])

parameter_range = [ [4 + k * i for k in range(5)] for i in [-1, 0, 1]]
print(parameter_range)
scores = kernel_cross_validation(svm, lambda betas: WDKernel(betas), train[0], parameter_range)
plt.plot([-1, 0, 1], scores, label=train[0].name())

plt.title('Different Accuracies after Cross-Validation')
plt.xlabel('C')
plt.ylabel('percent')
plt.legend()
plt.savefig(f'Plots/SVM_Cross_valid_{wd.name()}', dpi=300)
plt.show()