#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
from scipy.sparse import spmatrix

# cdef char* numpy2char(np.ndarray[np.int_t, ndim=1] np_array,char* char_array):

#     cdef int n = np_array.shape[0]
#     for i in range(n): char_array[i]=np_array[i]
#     return char_array

cdef char* numpy2char(np_array,char* char_array):

    cdef int n = np_array.shape[0]
    for i in range(n): char_array[i]=np_array[i]
    return char_array

cdef char* spmatrix2char(matrix,char* char_array):
    cdef int n = matrix.shape[1]

    for i in range(n): char_array[i]=0

    for i, j in zip(*matrix.nonzero()):
        char_array[j]=1

    return char_array

cdef double* numpy2double(np.ndarray[np.float_t, ndim=1] np_array,double* double_array):
    cdef int n = np_array.shape[0]
    for i in range(n): double_array[i]=np_array[i]
    return double_array

cdef np.ndarray[np.int_t, ndim=1] char2numpy(char* char_array, int n):

    cdef np.ndarray[np.int_t, ndim=1] np_array=np.zeros(n).astype(int)
    for i in range(n):np_array[i]=char_array[i]
    return np_array

cdef np.ndarray[np.float_t, ndim=1] double2numpy(double* char_array, int n):

    cdef np.ndarray[np.float_t, ndim=1] np_array=np.zeros(n)
    for i in range(n):np_array[i]=char_array[i]
    return np_array

