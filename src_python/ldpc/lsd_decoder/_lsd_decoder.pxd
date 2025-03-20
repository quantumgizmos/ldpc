#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpEntry, BpDecoderBase
ctypedef np.uint8_t uint8_t
from ldpc.bposd_decoder cimport OsdMethod

cdef extern from "lsd.hpp" namespace "ldpc::lsd":

     cdef const vector[double] EMPTY_DOUBLE_VECTOR "ldpc::lsd::EMPTY_DOUBLE_VECTOR"

     cdef cppclass LsdDecoderCpp "ldpc::lsd::LsdDecoder":
          LsdDecoderCpp(BpSparse& pcm, OsdMethod lsd_method, int lsd_order) except +
          vector[uint8_t]& lsd_decode(vector[uint8_t]& syndrome, const vector[double]& bit_weights, int bits_per_step, bool on_the_fly_decode)
          vector[uint8_t] decoding
          OsdMethod lsd_method
          int lsd_order

cdef class LsdDecoder():
     cdef int m
     cdef int n
     cdef bool MEMORY_ALLOCATED
     cdef LsdDecoderCpp* lsd
     cdef BpSparse* pcm
     cdef vector[uint8_t] _syndrome
     cdef vector[double] _bit_weights
     cdef int bits_per_step
     cdef vector[uint8_t] decoding
