#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "gf2sparse.hpp" namespace "gf2sparse" nogil:
    cdef cppclass GF2Entry "cygf2_entry":
        GF2Entry() except +
        int row_index
        int col_index
        bool at_end()

    cdef cppclass GF2Sparse "cygf2_sparse":
        int m
        int n
        GF2Sparse() except +
        GF2Sparse(int m, int n) except +
        GF2Entry* insert_entry(int i, int j)
        GF2Entry* get_entry(int i, int j)
        shared_ptr[GF2Sparse] transpose()
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[vector[int]] nonzero_coordinates()
        int entry_count()

cdef extern from "gf2sparse_linalg.hpp" nogil:

    cdef cppclass RowReduce "cy_row_reduce":
        RowReduce(shared_ptr[GF2Sparse] A) except +
        vector[int] rows
        vector[int] cols
        shared_ptr[GF2Sparse] L
        shared_ptr[GF2Sparse] U
        int rank
        int rref(bool full_reduce, bool lower_triangular)
        vector[uint8_t]& lu_solve(vector[uint8_t]& y)

    vector[vector[np.uint8_t]] cy_kernel(shared_ptr[GF2Sparse] mat)

cdef class LuDecomposition():
    cdef int m
    cdef int n
    cdef shared_ptr[GF2Sparse] cpcm
    cdef shared_ptr[GF2Sparse] L
    cdef shared_ptr[GF2Sparse] U
    cdef shared_ptr[RowReduce] rr


    

