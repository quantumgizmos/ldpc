#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpEntry, BpDecoderBase
ctypedef np.uint8_t uint8_t

cdef extern from "osd.hpp" namespace "ldpc::osd" nogil:

    cdef enum OsdMethod:
        OSD_OFF = 0
        OSD_0 = 1
        EXHAUSTIVE = 2
        COMBINATION_SWEEP = 3

    cdef cppclass OsdDecoderCpp "ldpc::osd::OsdDecoder":
        OsdDecoderCpp(BpSparse& parity_check_matrix, OsdMethod osd_method, int osd_order, vector[double] channel_probabilities) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome, vector[double]& log_prob_ratios)
        vector[uint8_t] osd0_decoding
        vector[uint8_t] osdw_decoding
        vector[double] channel_probabilities
        OsdMethod osd_method
        int osd_setup()
        int osd_method
        int osd_order

cdef class OsdDecoder():
     cdef int m
     cdef int n
     cdef bool MEMORY_ALLOCATED
     cdef OsdDecoderCpp* osdD
     cdef BpSparse* pcm
     cdef vector[uint8_t] _syndrome
     cdef vector[double] _bit_weights
     cdef int bits_per_step
     cdef vector[uint8_t] decoding
     cdef vector[double] _channel_probabilities
