#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "gf2sparse.hpp":
    cdef cppclass cygf2_entry:
        cygf2_entry() except +
        uint8_t value
        int row_index
        bool at_end()

    cdef cppclass cygf2_sparse:
        cygf2_sparse() except +
        cygf2_sparse(int m, int n) except +
        cygf2_entry* insert_entry(int i, int j, unsigned char value)
        cygf2_entry* get_entry(int i, int j)
        cygf2_sparse* L
        cygf2_sparse* U
        cygf2_sparse* transpose()
        cygf2_sparse* kernel()
        int lu_decomposition(bool reset_cols, bool full_reduce)
        vector[uint8_t]& lu_solve(vector[uint8_t]& y, vector[uint8_t]& x)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[int] rows
        vector[int] cols
        vector[int] inv_rows
        vector[int] inv_cols
        vector[vector[int]] nonzero_coordinates()
        int rank
        int m
        int n
        int node_count
        

cdef class gf2sparse:
    cdef bool PCM_ALLOCATED
    cdef bool PCMT_ALLOCATED
    cdef bool KERN_ALLOCATED
    cdef cygf2_sparse *pcm
    cdef int m
    cdef int n
    # cdef cygf2_sparse* _transpose()
    cdef void c_object_init(self,cygf2_sparse* mat)
