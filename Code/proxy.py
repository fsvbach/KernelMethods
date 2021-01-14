from ctypes import *
from os.path import abspath
so_file = abspath("Code/c-functions/kernel-functions.so")
functions = cdll.LoadLibrary(so_file)
functions.wd_kernel.argtypes = [POINTER(c_int), POINTER(c_int), c_int]
functions.wd_kernel.restype = c_double

def to_int_array(seq):
    return (c_int * len(seq))(*seq)


def wd_kernel_function(seq1, seq2, k):
    arr1 = to_int_array(seq1)
    arr2 = to_int_array(seq2)
    return (functions.wd_kernel(arr1, arr2, k, len(seq1)))

#print(wd_kernel_function([1,3,2],[1,3,2], 2))