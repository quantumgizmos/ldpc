#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
import numpy as np

cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log, tanh, isnan, abs

from ldpc.mod2sparse cimport *
from ldpc.c_util cimport numpy2char, char2numpy, numpy2double, double2numpy, spmatrix2char

cdef class bp_decoder:
    cdef mod2sparse* H
    cdef int m, n
    cdef char* error
    cdef char* synd
    cdef char* bp_decoding_synd
    cdef char* bp_decoding
    cdef char* received_codeword
    cdef char* decoding
    cdef int iter
    cdef int converge
    cdef double* channel_probs
    cdef double* log_prob_ratios
    cdef double error_rate
    cdef int max_iter
    cdef int bp_method
    cdef double ms_scaling_factor
    cdef int MEM_ALLOCATED
    cdef int input_vector_type

    cpdef np.ndarray[np.uint8_t, ndim=1] decode(self, input_vector)

    cdef char* bp_decode_cy(self)

    # Belief propagation with probability ratios
    cdef int bp_decode_prob_ratios(self)

    # Belief propagation with log probability ratios
    cdef int bp_decode_log_prob_ratios(self)


# cdef class bposd_decoder(bp_decoder):
#     cdef int test
#     pass


