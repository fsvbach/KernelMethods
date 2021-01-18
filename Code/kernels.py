from abc import ABC, abstractmethod #abstract classes
import numpy as np
import matplotlib.pyplot as plt

from .util import cached, compute_kernel_matrix_elementwise
from .proxy import cpp_functions

class Kernel(ABC): 

    def __init__ (self):
        self.cache = {}

    @abstractmethod
    def kernel_matrix(self, A, B):
        pass

    @abstractmethod
    def name(self):
        pass

    def plot_matrix(self, A, B):
        M = self.kernel_matrix(A, B)
        plt.imshow(M)
        plt.colorbar()
        plt.show()
        return M

class StringKernel(Kernel):

    @abstractmethod
    def compute_kernel_matrix(self, A, B):
        pass

    def kernel_matrix(self, A, B):
        name = f'kernel_matrix_{self.name()}_{A.name()}x{B.name()}'
        def fun():
            self.compute_kernel_matrix(A.as_strings(), B.as_strings())
        return cached(name, fun)

class LinearKernel(Kernel):

    def __init__(self, data_format = lambda d: d.as_bag_of_words()):
        self.data_format = data_format

    def name(self):
        return "LinearKernel"

    def kernel_matrix(self, A, B):
        a = self.data_format(A)
        b = self.data_format(B)
        matrix = a@b.T
        return matrix

class SpectrumKernel(LinearKernel):
    def __init__(self, k):
        super().__init__(lambda data: data.as_spectrum(k))
        self.k = k

    def name(self):
        return f'SpectrumKernel (k={self.k})'

    def kernel_matrix(self, A, B):
        M = super().kernel_matrix(A, B).toarray()
        M /= M.max()
        return M

class MismatchKernel(LinearKernel):
    def __init__(self, k, m):
        super().__init__(lambda data: data.as_spectrum(k, m))
        self.k = k
        self.m = m

    def name(self):
        return f'MismatchKernel (k={self.k} m={self.m})'

    def kernel_matrix(self, A, B):
        M = super().kernel_matrix(A, B).toarray()
        M /= M.max()
        return M
    

class GaussianKernel(Kernel):

    def __init__(self, sigma,  data_format = lambda d: d.as_bag_of_words() ):
        self.data_format = data_format
        self.sigma = sigma

    def name(self):
        return f'GaussianKernel_(sigma={self.sigma})'
    
    def kernel_matrix(self, A, B):
        A = self.data_format(A)
        B = self.data_format(B)
        A1 = np.sum(A*A,1)
        B1 = np.sum(B*B,1)
        A2 = np.ones(len(A))
        B2 = np.ones(len(B))
        
        K1 = np.outer(A1,B2)
        K2 = np.outer(A2,B1)
        
        K0 = K1 + K2 - 2 * np.inner(A,B)
        K = np.exp(- K0 / self.sigma**2)/np.sqrt(2 * np.pi)/self.sigma
        return K

class WDKernel(Kernel):
    def __init__(self, beta):
        self.k = len(beta)
        self.beta = beta
    
    def name(self):
        return f"WD_kernel_{self.k}"
    
    def params(self):
        return "beta"

    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k in range(self.k):
            if (self.beta[k] == 0): continue
            identifier = f'WD_kernel_{A.name()}x{B.name()}_k={k}'
            matrix = cached(identifier, lambda: self.compute_kernel_matrix_for_k(A, B, k))
            result = result + self.beta[k] * matrix
        result = result / result.max()
        return result

    def compute_kernel_matrix_for_k(self, a, b, k):
        kernel_function = lambda seq1, seq2: cpp_functions.wd_kernel(seq1, seq2, k, len(seq1))
        return compute_kernel_matrix_elementwise(a.as_ctype_int_array(), b.as_ctype_int_array(), kernel_function, a == b)

class WDShiftedKernel(Kernel):
    def __init__(self, beta, S):
        self.k = len(beta)
        self.beta = beta
        self.S = S
    
    def name(self):
        return f"WD_shifted_kernel_k={self.k}_S={self.S}"
    
    def params(self):
        return "beta"

    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k in range(self.k):
            if (self.beta[k] == 0): continue
            identifier = f'WD_shifted_kernel_{A.name()}x{B.name()}_k={k}_S={self.S}'
            matrix = cached(identifier, lambda: self.compute_kernel_matrix_for_k(A, B, k))
            result = result + self.beta[k] * matrix
        result = result / result.max()
        return result

    def compute_kernel_matrix_for_k(self, a, b, k):
        kernel_function = lambda seq1, seq2: cpp_functions.wd_shifted(seq1, seq2, k, len(seq1), self.S)
        return compute_kernel_matrix_elementwise(a.as_ctype_int_array(), b.as_ctype_int_array(), kernel_function, a == b)

class SumKernel(Kernel):
    def __init__(self, kernels, weights):
        assert len(kernels) == len(weights)
        self.kernels = kernels
        self.weights = np.array(weights)
    
    def name(self):
        return "Sum Kernel"
    
    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k,w in zip(self.kernels, self.weights):
            result += w * k.kernel_matrix(A,B)
        return result / self.weights.sum()
        
            
            