#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc,free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
cimport cython
from ldpc.union_find_decoder cimport union_find_decoder_cpp
from ldpc.bp_decoder cimport bp_sparse

ctypedef np.uint8_t uint8_t

cdef extern from "mbp.hpp":
    cdef cppclass mbp_entry "cymbp_entry":
        mbp_entry() except +
        uint8_t value
        bool at_end()

    cdef cppclass mbp_sparse:
        mbp_sparse() except +
        mbp_sparse(int m, int n) except +
        mbp_entry* insert_entry(int i, int j, uint8_t value)
        mbp_entry* get_entry(int i, int j)
        int m
        int n
        int print()

    cdef cppclass mbp_decoder_cpp "mbp_decoder":
        mbp_decoder_cpp(mbp_sparse* pcm, vector[vector[double]] error_channel, int max_iter, vector[vector[double]] alpha_parameter, double beta_paramter, int bp_method, double gamma_parameter) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding
        vector[vector[double]] log_prob_ratios
        vector[vector[double]] channel_probs
        vector[vector[double]] alpha
        double beta
        int max_iter
        int iterations
        bool converge

cdef class mbp_decoder:
    cdef mbp_sparse *pcm
    cdef bp_sparse *pcmX
    cdef bp_sparse *pcmZ
    cdef int stab_count, qubit_count, max_iter
    cdef vector[uint8_t] bp_decoding
    cdef vector[uint8_t] syndrome
    cdef vector[vector[double]] error_channel
    cdef mbp_decoder_cpp *bpd
    cdef uf_decoder_cpp *uf_bpdX
    cdef uf_decoder_cpp *uf_bpdZ
    cdef double error_rate
    cdef double[3] xyz_bias
    cdef bool MEMORY_ALLOCATED
    cdef vector[vector[double]] alpha
    cdef double beta_parameter
    cdef double gamma_parameter
    cdef int bp_method
    cdef int OUTPUT_TYPE

