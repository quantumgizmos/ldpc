#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
import numpy as np

cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log, tanh, isnan, abs
from ldpc.mod2sparse cimport *
from ldpc.c_util cimport numpy2char, char2numpy, numpy2double, double2numpy, spmatrix2char
from ldpc.bp_decoder cimport bp_decoder

cdef class bposd_decoder(bp_decoder):
    cdef int test
    pass


