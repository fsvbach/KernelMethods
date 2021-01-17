import numpy as np
import scipy.sparse as sp
import pandas as pd
import os

storage_folder_name = "cache"
predictions_folder_name = "Predictions"
plots_foler_name = "Plots"

def compute_spectrum(sequences, k):
    '''
    seq: sequences as array of base4 encoded ints
    k: size of the kmers (begins at 1)
    '''
    #seq = [0, 1, 1, 2, 3 ...]
    n = len(sequences)
    spectrum = sp.dok_matrix((n, 4**k))
    mask = (1 << (2 * k)) - 1
    for i,seq in enumerate(sequences):
        kmer = 0
        for j, c in enumerate(seq):
            kmer = ((kmer << 2) & mask) | c
            if j + 1 >= k:
                spectrum[i, kmer] += 1 
    return spectrum.tocsr()



def compute_kernel_matrix_elementwise(A, B, kernel_function, symmetric = False):
        matrix = np.zeros((len(A),len(B)))
        for i,a in enumerate(A):
            for j,b in enumerate(B):
                if not symmetric or j>=i: 
                    matrix[i,j] = kernel_function(a,b) 
        if symmetric:
            matrix = matrix + matrix.T - np.diag(matrix.diagonal())
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

def cross_validation(models, kernels, datasets, D=5):
    
    scores = np.zeros( shape=(len(datasets), len(kernels), len(models)) ) 
    
    for i,dataset in enumerate(datasets):
        y_train    = dataset.labels()
        indices    = np.arange(len(y_train))
        np.random.shuffle(indices)
        indices    = np.array_split(indices, D)
        
        for j,kernel in enumerate(kernels):
            K_train = kernel.kernel_matrix(dataset, dataset)
            
            for k, model in enumerate(models):
                
                for idx in indices:
                    yte = y_train[idx]
                    ytr = np.delete(y_train, idx)
                    K   = np.delete(K_train, idx, 1)
                    Kte = K[idx]
                    Ktr = np.delete(K, idx, 0)
        
                    model.fit(Ktr, ytr)
                    scores[i,j,k] += accuracy(model.predict(Kte), yte)/D
                
                print(f'{np.round(scores[i,j,k],3)}% validation score for {dataset.name(), model.name(),kernel.name()}')
                       
    return scores
