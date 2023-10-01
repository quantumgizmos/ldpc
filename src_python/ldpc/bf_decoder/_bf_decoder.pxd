#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport bp_sparse, bp_entry, bp_decoder_cpp, bp_decoder_base
# from ldpc.uf_decoder cimport uf_decoder, uf_decoder_cpp
from ldpc.uf_decoder cimport uf_decoder_cpp
ctypedef np.uint8_t uint8_t

cdef class bf_decoder(bp_decoder_base):
    cdef uf_decoder_cpp* ufd
    cdef vector[uint8_t] bf_decoding
    cdef vector[uint8_t] residual_syndrome
    cdef bool matrix_solve
    cdef int bits_per_step
