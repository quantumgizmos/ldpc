#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
from ldpc.bp_decoder cimport BpSparse, BpDecoderBase
ctypedef np.uint8_t uint8_t

cdef extern from "flip.hpp" namespace "ldpc::flip" nogil:

    cdef cppclass FlipDecoderCpp "ldpc::flip::FlipDecoder":
        FlipDecoderCpp(BpSparse& pcm, int max_iter, int pfreq, int seed) except +
        vector[uint8_t]& decode(vector[uint8_t]& syndrome)
        vector[uint8_t] decoding
        

cdef class BpFlipDecoder(BpDecoderBase):
    cdef FlipDecoderCpp* flipD
    cdef int flip_iterations
