#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np
cimport cython

from mod2sparse cimport mod2sparse

cdef inline char* numpy2char(np.ndarray[np.int_t, ndim=1] np_array,char* char_array):
    
    cdef int n = np_array.shape[0]
    for i in range(n): char_array[i]=np_array[i]
    return char_array


cdef inline double* numpy2double(np.ndarray[np.float_t, ndim=1] np_array,double* double_array):
    cdef int n = np_array.shape[0]
    for i in range(n): double_array[i]=np_array[i]
    return double_array

cdef inline np.ndarray[np.int_t, ndim=1] char2numpy(char* char_array, int n):
    
    cdef np.ndarray[np.int_t, ndim=1] np_array=np.zeros(n).astype(int)
    for i in range(n):np_array[i]=char_array[i]
    return np_array

cdef inline np.ndarray[np.float_t, ndim=1] double2numpy(double* char_array, int n):
    
    cdef np.ndarray[np.float_t, ndim=1] np_array=np.zeros(n)
    for i in range(n):np_array[i]=char_array[i]
    return np_array

