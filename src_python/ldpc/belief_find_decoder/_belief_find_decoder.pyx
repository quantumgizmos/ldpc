#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class BeliefFindDecoder(BpDecoderBase):
  
    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, matrix_solve: bool = False, bits_per_step:int = 0):
        self.MEMORY_ALLOCATED=False
        self.ufd = new uf_decoder_cpp(self.pcm[0])
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
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            self.bpd.converge = True
            return np.zeros(self.n,dtype=DTYPE)

        self.bpd.decoding = self.bpd.decode(self._syndrome)
        out = np.zeros(self.n,dtype=DTYPE)
        if self.bpd.converge:
            for i in range(self.n): out[i] = self.bpd.decoding[i]

        if not self.bpd.converge:
            if self.matrix_solve:
                self.ufd.decoding = self.ufd.matrix_decode(self._syndrome, self.bpd.log_prob_ratios,self.bits_per_step)
            else:
                self.ufd.decoding = self.ufd.peel_decode(self._syndrome, self.bpd.log_prob_ratios,self.bits_per_step)
            for i in range(self.n):
                # self.bf_decoding[i] = self.ufd.decoding[i]^self.bpd.decoding[i]
                out[i] = self.ufd.decoding[i]
        
        return out

    # def maximum_cluster_size(self):
    #     return self.ufd.maximum_cluster_size[0], self.ufd.maximum_cluster_size[1]