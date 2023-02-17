#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc2.bp_decoder cimport bp_sparse, bp_entry, bp_decoder_cpp, bp_decoder_base
ctypedef np.uint8_t uint8_t

cdef extern from "osd.hpp":
    cdef cppclass osd_decoder_cpp "osd_decoder":
        osd_decoder_cpp(bp_sparse* parity_check_matrix, int osd_method, int osd_order, vector[double] channel_probabilities) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome, vector[double]& log_prob_ratios)
        vector[uint8_t] bp_decoding
        vector[uint8_t] osd0_decoding
        vector[uint8_t] osdw_decoding
        vector[double] channel_probs
        int osd_method
        int osd_order

cdef class bposd_decoder(bp_decoder_base):
    cdef int osd_order
    cdef int osd_method
    cdef osd_decoder_cpp* osd