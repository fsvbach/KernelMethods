import numpy as np
import pandas as pd 

from kernels import *
from util import *
from models import *
from data import load_data
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
wd = WDKernel(list(1 for i in range(5)))

#save_predictions(svm, wd, train, test)


print(wd.kernel_matrix(train[0], train[0]))

#SVM_C_cross_validation(gaussian, train)
SVM_C_cross_validation(wd, [train[0]])