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

class WDKernel(Kernel):
    def __init__(self, beta):
        self.k = len(beta)
        self.beta = beta
    
    def name(self):
        return f"WD_kernel_{self.k}"

    def kernel_matrix(self, A, B):
        a = A.as_int_encoded_strings()
        b = B.as_int_encoded_strings()
        result = np.zeros((len(a), len(b)))
        for k in range(self.k):
            identifier = f'WD_kernel_{A.name()}x{B.name()}_k={k}'
            matrix = cached(identifier, lambda: self.compute_kernel_matrix_for_k(a, b, k))
            result = result + self.beta[k] * matrix
        return result

    def compute_kernel_matrix_for_k(self, a, b, k):
        def kernel_function(seq1, seq2):
            L = len(seq1)
            sum = 0
            u1, u2 = 0, 0
            mask = (1 << 2 * (k + 1)) - 1
            for l in range(L):
                u1 = (u1 << 2) & mask | seq1[l]
                u2 = (u2 << 2) & mask | seq2[l]
                if l >= k and u1 == u2:
                    sum += 1
            return sum 

        return compute_kernel_matrix_elementwise(a, b, kernel_function)