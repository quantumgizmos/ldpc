#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc2.bp_decoder cimport BpSparse, BpDecoderBase
ctypedef np.uint8_t uint8_t

cdef extern from "osd.hpp" namespace "osd" nogil:
    cdef cppclass OsdDecoderCpp "osd::OsdDecoder":
        OsdDecoderCpp(shared_ptr[BpSparse] parity_check_matrix, int osd_method, int osd_order, vector[double] channel_probabilities) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome, vector[double]& log_prob_ratios)
        vector[uint8_t] osd0_decoding
        vector[uint8_t] osdw_decoding
        int osd_method
        int osd_order

cdef class BpOsdDecoder(BpDecoderBase):
    cdef OsdDecoderCpp* osdD