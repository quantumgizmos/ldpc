#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np
cimport cython

cdef char* numpy2char(np.ndarray[np.int_t, ndim=1] np_array,char* char_array)
cdef double* numpy2double(np.ndarray[np.float_t, ndim=1] np_array,double* double_array)
cdef np.ndarray[np.int_t, ndim=1] char2numpy(char* char_array, int n)
cdef np.ndarray[np.float_t, ndim=1] double2numpy(double* char_array, int n)
 

