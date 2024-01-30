#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpEntry, BpDecoderBase
ctypedef np.uint8_t uint8_t

cdef extern from "lsd.hpp" namespace "ldpc::lsd":

    cdef const vector[double] NULL_DOUBLE_VECTOR "ldpc::lsd::NULL_DOUBLE_VECTOR"

    cdef cppclass lsd_decoder_cpp "ldpc::lsd::LsdDecoder":
        lsd_decoder_cpp(BpSparse& pcm) except +
        # vector[uint8_t]& on_the_fly_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights = NULL_DOUBLE_VECTOR)
        vector[uint8_t]& lsd_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step, bool on_the_fly_decode, int lsd_order)
        vector[uint8_t] decoding
        vector[int] cluster_size_stats

cdef class BpLsdDecoder(BpDecoderBase):
    cdef lsd_decoder_cpp* lsd
    cdef int bits_per_step
    cdef vector[uint8_t] bplsd_decoding
    cdef int lsd_order