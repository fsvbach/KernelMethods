import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from itertools import combinations, product

storage_folder_name = "cache"
predictions_folder_name = "Predictions"
plots_foler_name = "Plots"

def compute_spectrum(sequences, k, m = 0):
    '''
    seq: sequences as array of base4 encoded ints
    k: size of the kmers (begins at 1)
    '''
    n = len(sequences)
    spectrum = sp.dok_matrix((n, 4**k))
    mask = (1 << (2 * k)) - 1
    for i,seq in enumerate(sequences):
        print(f'...iteration {i+1} of {n}', end='\r')
        kmer = 0
        for j, c in enumerate(seq):
            kmer = ((kmer << 2) & mask) | c
            if j + 1 >= k:
                for variant in neighbourhood(kmer, k, m):
                    spectrum[i, variant] += 1
               # spectrum[i, kmer] += 1 
    return spectrum.tocsr()

def kmer2int(kmer):
    to_int = { 'A' : 0, 'C': 1, 'G' : 2, 'T' : 3}
    res = 0
    for l in kmer:
        res |= to_int[l]
        res <<= 2
    res >>= 2
    return res

def int2kmer(kmer, k):
    to_letter = ['A', 'C', 'G', 'T']
    res = ""
    for i in range(k):
        res = to_letter[kmer & 3] + res
        kmer >>= 2
    return res


def neighbourhood(kmer, k, m):
    def get_letter(s, n):
        return (s >> (2 * n)) & 3
    def set_letter(s, n, l):
        return (s & ~(3 << (2*n))) | (l << (2*n))

    for i in range(m + 1):
        for positions in combinations(range(k), i):
            for letters in product(range(4), repeat=i):
                skip = False
                copy = kmer
                for j,l in enumerate(letters):
                    if get_letter(kmer, positions[j]) == l:
                        skip = True
                        break
                    copy = set_letter(copy, positions[j], l)
                if (skip): 
                    continue
            #    print(f'yield {int2kmer(copy, k)}')
                yield copy


def compute_kernel_matrix_elementwise(A, B, kernel_function, symmetric = False):
        matrix = np.zeros((len(A),len(B)))
        for i,a in enumerate(A):
            print(f'... iteration {i+1} of {len(A)}', end='\r')
            for j,b in enumerate(B):
                if not symmetric or j>=i: 
                    matrix[i,j] = kernel_function(a,b) 
        if symmetric:
            matrix = matrix + matrix.T - np.diag(matrix.diagonal())
        return matrix


def cached(unique_name, function, sp_sparse=False):
    (load, save, filetype) = (sp.load_npz, sp.save_npz, 'npz') if sp_sparse else (np.load, np.save, 'npy')
    if (not os.path.exists(storage_folder_name)):
        os.mkdir(storage_folder_name)
    filename = f'{storage_folder_name}/{unique_name}.{filetype}'
    if (os.path.exists(filename)):
        print(f'Load {filename}')
        return load(filename)
    else:
        print(f'Compute {filename}...')
        obj = function()
        save(filename, obj)
        return obj

def save_predictions(models, kernels, training_data, test_data):
    '''
    models -- list of model objects
    kernels -- list of kernel objects
    training_data: list of TrainingData objects
    test_data: list of TestDataObjects
    '''

    name = ''
    for m,k in zip(models,kernels): name+='_'+m.name()+','+k.name()
    filename = f'predictions{name}.csv'
        
    Y_list = []
    for (model, kernel, train, test) in zip(models, kernels, training_data, test_data):
        print(f'Calculate Predictions for {model.name(), kernel.name(), train.name()}')
        pred = model.fit_and_predict(kernel, train, test)
        Y_list.append((pred/2+0.5).astype(int))

    Y_pred = pd.DataFrame(np.concatenate(Y_list), columns=['Bound'])
    index  = Y_pred.index.rename('Id')
    Y_pred = Y_pred.set_index(index)

    if (not os.path.exists(predictions_folder_name)):
        os.mkdir(predictions_folder_name)
    Y_pred.to_csv(f'{predictions_folder_name}/{filename}')
    print(f'Saved predictions to {predictions_folder_name}/{filename}')

def accuracy(y_pred, y_true):
        # assert y_pred.shape == y_true.shape
        # return 1 - np.linalg.norm(y_pred - y_true,ord=1) / len(y_true)
        return 1 - np.linalg.norm(y_pred.flatten() - y_true.flatten(), ord=1) / 2 / len(y_true)
    
def cross_validation(grid, D=10):

    scores = []
    for dataset in grid['dataset']:
        y_train    = dataset.labels()
        indices    = np.arange(len(y_train))
        np.random.shuffle(indices)
        indices    = np.array_split(indices, D)
        
        for kernel in grid['kernel']:
            print(f'Begin with Kernel {kernel.name()}')
            K_train = kernel.kernel_matrix(dataset, dataset)
            
            for model in grid['model']:
                
                score = 0
                for d, idx in enumerate(indices):
                    print(f'... iteration {d+1} of {D}', end='\r')
                    yte = y_train[idx]
                    ytr = np.delete(y_train, idx)
                    K   = np.delete(K_train, idx, 1)
                    Kte = K[idx]
                    Ktr = np.delete(K, idx, 0)
        
                    model.fit(Ktr, ytr)
                    score += accuracy(model.predict(Kte), yte)
                
                scores.append(score/D)
                print(f'{np.round(scores[-1],3)}% validation score for {dataset.name(), model.name(),kernel.name()}')
             
    names = ['dataset','kernel','model']
    labels = [[i.name() for i in grid[key]] for key in names]
    index  = pd.MultiIndex.from_product(labels, names=names)
    return pd.Series(scores, name = 'Accuracy', index=index, dtype='float32')
  