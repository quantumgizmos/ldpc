#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpEntry, BpDecoderBase
ctypedef np.uint8_t uint8_t


cdef extern from "union_find.hpp" namespace "ldpc::uf":
    cdef cppclass uf_decoder_cpp "ldpc::uf::UfDecoder":
        uf_decoder_cpp(BpSparse& pcm) except +
        vector[uint8_t]& peel_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step)
        vector[uint8_t]& matrix_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step)
        vector[uint8_t] decoding

    cdef const vector[double] EMPTY_DOUBLE_VECTOR

cdef class BeliefFindDecoder(BpDecoderBase):
    cdef uf_decoder_cpp* ufd
    cdef vector[uint8_t] bf_decoding
    cdef vector[uint8_t] residual_syndrome
    cdef str uf_method
    cdef int bits_per_step



