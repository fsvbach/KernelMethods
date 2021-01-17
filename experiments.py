import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt

    
(train, test) = data.load_data()

Linear = kernels.LinearKernel()
Ktr = Linear.kernel_matrix(train[0], train[0])
ytr = train[0].labels()

svm = models.our_SVM(C=1)
svm.fit(Ktr, ytr)

Kte = Linear.kernel_matrix(test[0], train[0])

pred = svm.predict(Kte)

