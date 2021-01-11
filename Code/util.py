import numpy as np
import pandas as pd
import os

storage_folder_name = "cache"
predictions_folder_name = "Predictions"
plots_foler_name = "Plots"

def compute_kernel_matrix_elementwise(A, B, kernel_function):
        matrix = np.zeros((len(A),len(B)))
        for i,a in enumerate(A):
            print(i)
            for j,b in enumerate(B):
                matrix[i,j] = kernel_function(a,b) 
        return matrix


def cached(unique_name, function):
    if (not os.path.exists(storage_folder_name)):
        os.mkdir(storage_folder_name)
    filename = f'{storage_folder_name}/{unique_name}.npy'
    if (os.path.exists(filename)):
        return np.load(filename)
    else:
        obj = function()
        np.save(filename, obj)
        return obj

def save_predictions(model, kernel, training_data, test_data):
    '''
    model -- a model object
    kernel -- a kernel object
    training_data: list of TrainingData objects
    test_data: list of TestDataObjects
    '''

    Y_list = []
    for (train, test) in zip(training_data, test_data):
        pred = model.fit_and_predict(kernel, train, test)
        Y_list.append(pred)

    Y_pred = pd.DataFrame(np.concatenate(Y_list), columns=['Bound'])
    index  = Y_pred.index.rename('Id')
    Y_pred = Y_pred.set_index(index)
    filename = f'predictions_{model.name()}_{kernel.name()}.csv'
    if (not os.path.exists(predictions_folder_name)):
        os.mkdir(predictions_folder_name)
    Y_pred.to_csv(f'{predictions_folder_name}/{filename}')

def accuracy(y_pred, y_true):
        assert len(y_pred) == len(y_true)
        return 1 - np.linalg.norm(y_pred - y_true,ord=1) / len(y_true)

def model_cross_validation(build_model, kernel, training_data, parameter_range, D=10):
    '''
    Use cross validation to score different model parameters.
    Parameters
    ----------
    build_model: function that creates a model object from an element in the parameter_range iterable
    kernel: kernel object
        Kernel to use
    training_data: label data
    parameter_range: iterable containing all parameters to try
    Returns
    -------
    scores: list of doubles
        the accuracy of the model for each value in parameter_range
    '''

    K_train = kernel.kernel_matrix(training_data, training_data)
    y_train = training_data.labels()

    N          = len(y_train)
    valid_size = int(N / D)
    indices    = np.arange(N)
    np.random.shuffle(indices)
    indices = np.split(indices[:D*valid_size], D)
    
    scores = []
    for params in parameter_range:
        score = 0
        for idx in indices:
            yte = y_train[idx]
            ytr = np.delete(y_train, idx)
            K   = np.delete(K_train, idx, 1)
            Kte = K[idx]
            Ktr = np.delete(K, idx, 0)

            model = build_model(params)
            model.fit(Ktr, ytr)
            pred = model.predict(Kte)
            score += accuracy(pred, yte) / D
        scores.append(score)
    return scores


def kernel_cross_validation(model, build_kernel, training_data, parameter_range):
    #todo: evaluate different kernel params
    pass

