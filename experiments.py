import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt

    
(train, test) = data.load_data()


# mismatchKernelOld = kernels.MismatchKernelOld(10, 1)
# K2 = mismatchKernelOld.plot_matrix(train[0], train[0])
# print(K2)