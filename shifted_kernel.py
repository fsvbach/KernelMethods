from Code import kernels, models, data
import numpy as np
from experiments import SVM_cross_validation

(train, test) = data.load_data()

w = kernels.WDShiftedKernel([0,0,0,1], 20)
SVM_cross_validation(w, train[:1], 10.0**np.arange(-2,4))
