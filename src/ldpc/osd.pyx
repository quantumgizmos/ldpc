#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
from ldpc.bp_decoder import bp_decoder
from scipy.sparse import spmatrix

cdef class bposd_decoder(bp_decoder):

    def __cinit__(self,parity_check_matrix,**kwargs):
        self.test=149
        pass

    def print_test(self):
        return self.test















