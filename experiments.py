import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt

    
(train, test) = data.load_data()

Linear = kernels.LinearKernel()
K = Linear.kernel_matrix(train[0], train[0])
y = train[0].labels()

svm = models.our_SVM(C=1)
svm.fit(K, y)
