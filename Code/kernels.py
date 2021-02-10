from abc import ABC, abstractmethod #abstract classes
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

from .util import cached, compute_kernel_matrix_elementwise, neighbourhood, int2kmer
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
        plt.imshow(np.log(M+0.00001))
        plt.colorbar()
        plt.show()
        return M

class LinearKernel(Kernel):

    def __init__(self, data_format = lambda d: d.as_bag_of_words()):
        self.data_format = data_format

    def name(self):
        return "LinearKernel"

    def kernel_matrix(self, A, B):
        a = self.data_format(A)
        b = self.data_format(B)
        print(f"Compute Linear Kernel for {A.name(), B.name()}")
        matrix = a@b.T
        return matrix

class SpectrumKernel(LinearKernel):
    def __init__(self, k):
        super().__init__(lambda data: data.as_spectrum(k))
        self.k = k

    def name(self):
        return f"SK_{self.k}"

    def kernel_matrix(self, A, B):
        M = super().kernel_matrix(A, B).toarray()
        M /= M.max()
        return M

class MismatchKernel(Kernel):

    def __init__(self, k, m1, m2, timer = False):
        self.k = k
        self.m1 = m1
        self.m2 = m2 
        self.m = m1 + m2

    def name(self):
        return f"MK_{self.k}_{self.m}"

    def kernel_matrix(self, A, B):
        def compute_kernel_matrix():
            t1 = time.perf_counter()
            A_spectrum = A.as_spectrum(self.k, self.m1)
            t2 = time.perf_counter()
            print(f'... finished in {t2 - t1:2.4f}s)')
            
            t1 = time.perf_counter()
            B_spectrum = B.as_spectrum(self.k, self.m2)
            t2 = time.perf_counter()
            print(f'... finished in {t2 - t1:2.4f}s)')

            print(f"Compute dot-product for {A.name(), B.name()}")
            t1 = time.perf_counter()
            matrix = A_spectrum@B_spectrum.T
            matrix /= matrix.max()
            t2 = time.perf_counter()
            print(f'... finished in {t2 - t1:2.4f}s)')
            return matrix.toarray()

        identifier = f"MismatchKernel_{A.name()}x{B.name()}_k={self.k}_m={self.m}"
        matrix = compute_kernel_matrix()
        return matrix 

class GaussianKernel(Kernel):

    def __init__(self, sigma,  data_format = lambda d: d.as_bag_of_words() ):
        self.data_format = data_format
        self.sigma = sigma

    def name(self):
        return f"G_{self.sigma}"
    
    def kernel_matrix(self, A, B):
        a = self.data_format(A)
        b = self.data_format(B)
        print(f"Compute Gaussian Kernel for {A.name(), B.name()}")
        A1 = np.sum(a*a,1)
        B1 = np.sum(b*b,1)
        A2 = np.ones(len(a))
        B2 = np.ones(len(b))
        
        K1 = np.outer(A1,B2)
        K2 = np.outer(A2,B1)
        
        K0 = K1 + K2 - 2 * np.inner(a,b)
        K = np.exp(- K0 / self.sigma**2)/np.sqrt(2 * np.pi)/self.sigma
        return K

class WDKernel(Kernel):
    def __init__(self, beta):
        self.k = len(beta)
        self.beta = beta

    def name(self):
        return f"WDK_{self.beta}"

    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k in range(self.k):
            if (self.beta[k] == 0): continue
            identifier = f"WD_kernel_{A.name()}x{B.name()}_k={k}"
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
        return f"WDK_{self.k}_{self.S}"

    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k in range(self.k):
            if (self.beta[k] == 0): continue
            identifier = f"WD_shifted_kernel_{A.name()}x{B.name()}_k={k}_S={self.S}"
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
        return "Sum Kernel {self.weights}"

    def kernel_matrix(self, A, B):
        result = np.zeros((len(A), len(B)))
        for k,w in zip(self.kernels, self.weights):
            result += w * k.kernel_matrix(A,B)
        return result / self.weights.sum()
        
            
            
