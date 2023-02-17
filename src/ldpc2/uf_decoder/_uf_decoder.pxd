#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc2.bp_decoder cimport bp_sparse, bp_entry, bp_decoder_cpp, bp_decoder_base
ctypedef np.uint8_t uint8_t

cdef extern from "union_find.hpp":
    cdef cppclass uf_decoder_cpp "uf_decoder":
        uf_decoder_cpp(bp_sparse* pcm) except +
        vector[uint8_t]& peel_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step)
        vector[uint8_t]& matrix_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step)
        # vector[uint8_t]& bposd_decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding

    cdef const vector[double] NULL_DOUBLE_VECTOR

cdef class uf_decoder:
    cdef bp_sparse* pcm
    cdef uf_decoder_cpp* ufd
    cdef vector[double] error_channel
    cdef vector[uint8_t] uf_decoding
    cdef vector[uint8_t] syndrome
    cdef vector[double] bit_weights
    cdef int m, n
    cdef int bits_per_step
    cdef bool MEMORY_ALLOCATED