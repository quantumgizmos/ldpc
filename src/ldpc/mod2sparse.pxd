#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np

from ldpc.c_util cimport numpy2double,numpy2char,char2numpy,double2numpy

cdef extern from "mod2sparse.h":
    
    ctypedef struct mod2entry:
        int row, col, sgn
        double check_to_bit, bit_to_check

    ctypedef struct mod2sparse:
        int n_rows
        int n_cols

        mod2entry *rows
        mod2entry *cols

cdef extern from "mod2sparse.h":

    cdef mod2sparse *mod2sparse_allocate(int n_rows, int n_cols)
    cdef mod2entry *mod2sparse_insert(mod2sparse *m,int row,int col)
    cdef void mod2sparse_free(mod2sparse *m)
    cdef void mod2sparse_mulvec(mod2sparse *m, char *a, char *b)
    cdef void mod2sparse_copycols(mod2sparse *A, mod2sparse *B, int *cols)

    cdef mod2entry* mod2sparse_first_in_row(mod2sparse *m, int i)
    cdef mod2entry* mod2sparse_first_in_col(mod2sparse *m, int j)
    cdef mod2entry* mod2sparse_last_in_row(mod2sparse *m, int i)
    cdef mod2entry* mod2sparse_last_in_col(mod2sparse *m, int j)
    cdef mod2entry* mod2sparse_next_in_row(mod2entry *e)
    cdef mod2entry* mod2sparse_next_in_col(mod2entry *e)
    cdef mod2entry* mod2sparse_prev_in_row(mod2entry *e)
    cdef mod2entry* mod2sparse_prev_in_col(mod2entry *e)
    cdef int mod2sparse_at_end(mod2entry *e)
    cdef int mod2sparse_row(mod2entry *e)
    cdef int mod2sparse_col(mod2entry *e)
    cdef int mod2sparse_rows(mod2sparse *m)
    cdef int mod2sparse_cols(mod2sparse *m)
    cdef int MEM_ALLOCATED


cdef mod2sparse* numpy2mod2sparse(mat)
cdef mod2sparse* alist2mod2sparse(fname)


cdef class pymod2sparse():

    cdef mod2sparse *matrix
    cdef mod2entry *e
    cdef int m,n,iter_axis,reverse_iterate, row_index, col_index,start
    cdef char *vec_n
    cdef char *vec_m
    cdef int MEM_ALLOCATED

    cpdef iter_row(self,int row_index,int reverse_iterate)
    cpdef iter_col(self,int col_index,int reverse_iterate)
    cpdef np.ndarray[np.int_t, ndim=1] mul(self, np.ndarray[np.int_t, ndim=1] vector)


