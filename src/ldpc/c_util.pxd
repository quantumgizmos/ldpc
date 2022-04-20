#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np
cimport cython

# cdef char* numpy2char(np.ndarray[np.int_t, ndim=1] np_array,char* char_array)
cdef char* numpy2char(np_array,char* char_array)
cdef char* spmatrix2char(matrix,char* char_array)
cdef double* numpy2double(np.ndarray[np.float_t, ndim=1] np_array,double* double_array)
cdef np.ndarray[np.int_t, ndim=1] char2numpy(char* char_array, int n)
cdef np.ndarray[np.float_t, ndim=1] double2numpy(double* char_array, int n)
 

cdef extern from "binary_char.h":
    cdef void print_char_nonzero(char *val,int len)
    cdef int bin_char_equal(char *vec1, char *vec2, int len)
    cdef int bin_char_is_zero(char *vec1, int len)
    cdef void print_char(char *val, int len)
    cdef int bin_char_add(char *vec1, char *vec2, char *out_vec, int len)
    cdef char *decimal_to_binary_reverse(int n,int K)
    cdef int bin_char_weight(char *val,int len)
    cdef int hamming_difference(char *v1,char *v2,int len)

cdef extern from "sort.h":
    cdef void soft_decision_col_sort(double *soft_decisions,int *cols, int N)

