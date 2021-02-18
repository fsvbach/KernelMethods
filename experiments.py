from Code import kernels, util, models, data
import numpy as np
import pandas as pd

### LOADING DATA ###
(training_data, test_data) = data.load_data()


### SAVE PREDICTIONS ###

# best models and kernels so far
model  = [models.our_SVM(C) for C in [4.5, 13.5, 3.5]]      

kernel = [kernels.MismatchKernel(10, 2, 2),
          kernels.MismatchKernel(10, 2, 2),
          kernels.MismatchKernel(10, 2, 2)]

#save predictions
util.save_predictions(model, kernel, training_data, test_data)


### PLOT CROSS VALIDATION ###

grid = {'model': [models.our_SVM(C) for C in range(1,20)], 
        'kernel': [kernels.MismatchKernel(12,2,2)], 
        'dataset': training_data
        }

view = {'title' : 'kernel',
        'legend': 'dataset',
        'xaxis' : 'model'
        }

## compute scores
# scores = util.cross_validation(grid, D=10)

## load, if already computed
# scores = pd.read_csv('Plots/scores.csv', index_col=[0,1,2])

## do the plot
# util.plot_cross_val(scores, view)






