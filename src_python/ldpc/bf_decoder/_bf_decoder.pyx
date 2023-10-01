#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class bf_decoder(bp_decoder_base):
  
    def __cinit__(self,pcm, error_rate=None, error_channel=None, max_iter=0, bp_method=1, ms_scaling_factor=1.0, schedule=0, osd_method=0,osd_order=0, matrix_solve = False, bits_per_step= 0):
        self.MEMORY_ALLOCATED=False
        self.ufd = new uf_decoder_cpp(self.pcm)
        self.bf_decoding.resize(self.n) #C vector for the bf decoding
        self.residual_syndrome.resize(self.m) #C vector for the bf decoding
        self.matrix_solve = matrix_solve
        self.bits_per_step = bits_per_step

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.ufd

    def decode(self,syndrome):
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        DTYPE = syndrome.dtype
        for i in range(self.m): self.syndrome[i] = syndrome[i]
        self.bpd.decoding = self.bpd.decode(self.syndrome)
        out = np.zeros(self.n,dtype=DTYPE)
        if self.bpd.converge:
            for i in range(self.n): out[i] = self.bpd.decoding[i]

        if not self.bpd.converge:
            if self.matrix_solve:
                self.ufd.decoding = self.ufd.matrix_decode(self.syndrome, self.bpd.log_prob_ratios,self.bits_per_step)
            else:
                self.ufd.decoding = self.ufd.peel_decode(self.syndrome, self.bpd.log_prob_ratios,self.bits_per_step)
            for i in range(self.n):
                # self.bf_decoding[i] = self.ufd.decoding[i]^self.bpd.decoding[i]
                out[i] = self.ufd.decoding[i]
        
        return out

    # def maximum_cluster_size(self):
    #     return self.ufd.maximum_cluster_size[0], self.ufd.maximum_cluster_size[1]