import numpy as np
import pandas as pd 

from Code import kernels
from Code import util
from Code import models
from Code import data
import matplotlib.pyplot as plt

    
(training_data, test_data) = data.load_data()


mkd = kernels.MismatchKernel(11, 2,1)
K2 = mkd.plot_matrix(training_data[2], training_data[2])
print(K2)


# # best models fo far
# models  = [models.SVM(C) for C in [7,14,8]]
# kernels = [kernels.MismatchKernelDirect(K, M) for (K,M) in zip([10,10,11],[1,1,1])]

# #save predictions
# util.save_predictions(models, kernels, training_data, test_data)

