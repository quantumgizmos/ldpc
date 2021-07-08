#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "mod2sparse.h":
    ctypedef struct mod2entry:
        int row, col, sgn
        double check_to_bit, bit_to_check

    ctypedef struct mod2sparse:
        int n_rows
        int n_cols

        mod2entry *rows
        mod2entry *cols

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

cdef inline mod2sparse* numpy2mod2sparse(mat):
    
    cdef mod2sparse* sparse_mat
    cdef int i,j,m,n
    m=mat.shape[0]
    n=mat.shape[1]
    sparse_mat=mod2sparse_allocate(m,n)

    for i in range(m):
        for j in range(n):
            if mat[i,j]:
                mod2sparse_insert(sparse_mat,i,j)

    return sparse_mat


cdef inline mod2sparse* alist2mod2sparse(fname):

    cdef mod2sparse* sparse_mat

    alist_file = np.loadtxt(fname, delimiter='\n',dtype=str)
    matrix_dimensions=alist_file[0].split()
    m=int(matrix_dimensions[0])
    n=int(matrix_dimensions[1])

    sparse_mat=mod2sparse_allocate(m,n)

    for i in range(m):
        for item in alist_file[i+4].split():
            if item.isdigit():
                column_index = int(item)
                mod2sparse_insert(sparse_mat,i,column_index)

    return sparse_mat


