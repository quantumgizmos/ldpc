#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "bp2.hpp":
    cdef cppclass gf2csr:
        gf2csr(int rows, int cols, int max_row_weight, int max_col_weight) except +
        vector[int] row_indices
        vector[int] col_indices
        vector[int] col_index_map
        vector[int] row_widths
        vector[int] col_heights
        int max_row_weight
        int max_col_weight
        int row_count
        int col_count
        int get_row_index(int)
        int get_col_index(int)
        void insert_entry(int, int)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        bool entry_exists(int, int)
        void print()

    cdef cppclass bp_decoder_cpp "bpcsr_decoder":
        bp_decoder_cpp(gf2csr*, double, int, double) except +
        int converge
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding
        vector[double] log_prob_ratios
        int iterations


cdef class bp_decoder:
    cdef gf2csr* pcm
    cdef int m, n, max_iter
    cdef vector[uint8_t] syndrome
    cdef double error_rate
    cdef double ms_scaling_factor
    cdef bp_decoder_cpp *bpd
