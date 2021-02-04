import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt

    
(train, test) = data.load_data()
toy = data.load_toy_data()[0]



for i in range(5):
    print(f"m = {i}")
    mismatchKernel = kernels.MismatchKernel(9, i)
    K = mismatchKernel.kernel_matrix(train[0], train[0])

# plt.imshow(np.log(K))
# plt.show()
#print(K)

# mismatchKernelOld = kernels.MismatchKernelOld(10, 1)
# K2 = mismatchKernelOld.plot_matrix(train[0], train[0])
# print(K2)