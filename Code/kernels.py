from abc import ABC, abstractmethod #abstract classes
import numpy as np
from util import cached, compute_kernel_matrix_elementwise

class Kernel(ABC): 

    def __init__ (self):
        self.cache = {}

    @abstractmethod
    def kernel_matrix(self, A, B):
        pass

    @abstractmethod
    def name(self):
        pass

class VectorKernel(Kernel):

    @abstractmethod
    def compute_kernel_matrix(self, A, B):
        pass

    def kernel_matrix(self, A, B):
        a = A.as_bag_of_words()
        b = B.as_bag_of_words()
        return self.compute_kernel_matrix(a,b)


class StringKernel(Kernel):

    @abstractmethod
    def compute_kernel_matrix(self, A, B):
        pass

    def kernel_matrix(self, A, B):
        name = f'kernel_matrix_{self.name()}_{A.name()}x{B.name()}'
        def fun():
            self.compute_kernel_matrix(A.as_strings(), B.as_strings())
        return cached(name, fun)

class LinearKernel(VectorKernel):

    def name(self):
        return "LinearKernel"

    def compute_kernel_matrix(self, A, B):
        matrix = A@B.T
        return matrix

class GaussianKernel(VectorKernel):

    def __init__(self, sigma = 1):
        self.sigma = 1

    def name(self):
        return f'GaussianKernel_(sigma={self.sigma})'
    
    def compute_kernel_matrix(self, A, B):
        A1 = np.sum(A*A,1)
        B1 = np.sum(B*B,1)
        A2 = np.ones(len(A))
        B2 = np.ones(len(B))
        
        K1 = np.outer(A1,B2)
        K2 = np.outer(A2,B1)
        
        K0 = K1 + K2 - 2 * np.inner(A,B)
        K = np.exp(- K0 / self.sigma**2)
        return K

class WDKernel(StringKernel):
    def __init__(self, beta):
        self.k = len(beta)
        self.beta = beta
    
    def name(self):
        return f"WD_kernel_{self.k}"

    def compute_kernel_matrix(self, A, B):
        def kernel_function(seq1, seq2):
            count = 0
            L = len(seq1)
            for l in range(0, L - self.k + 1):
                if seq1[l:l+self.k] == seq2[l:l+self.k]:
                    count += 1
            return count  
        return compute_kernel_matrix_elementwise(A, B, kernel_function)