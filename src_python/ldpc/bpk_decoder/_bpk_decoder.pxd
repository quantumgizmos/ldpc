#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpDecoderBase, BpDecoderCpp
ctypedef np.uint8_t uint8_t

cdef extern from "bp_k.hpp" namespace "ldpc::bpk" nogil:

    cdef vector[uint8_t]& bp_k_decode(BpDecoderCpp& bpd, vector[uint8_t]& syndrome)
    cdef vector[uint8_t]& bp_k_decode_ps(BpDecoderCpp& bpd, vector[uint8_t]& syndrome)

cdef class BpKruskalDecoder(BpDecoderBase):
    pass
