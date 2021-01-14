from ctypes import *
from os.path import abspath
so_file = abspath("Code/c-functions/kernel-functions.so")
cpp_functions = cdll.LoadLibrary(so_file)
#functions.wd_kernel.argtypes = [POINTER(c_int), POINTER(c_int), c_int, c_int]
cpp_functions.wd_kernel.restype = c_double

#functions.wd_shifted.argtypes = [POINTER(c_int), POINTER(c_int), c_int, c_int]
cpp_functions.wd_shifted.restype = c_double


