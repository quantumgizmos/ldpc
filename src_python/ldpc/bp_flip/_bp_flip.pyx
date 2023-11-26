#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

cdef class BpFlipDecoder(BpDecoderBase):

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None, osd_method: int = 0,
                 osd_order: int = 0, flip_iterations: int = 0, pflip_frequency: int = 0, pflip_seed: int = 0):
        
        self.MEMORY_ALLOCATED=False

        ## set up OSD with default values and channel probs from BP
        self.flip_iterations = flip_iterations
        self.flipD = new FlipDecoderCpp(self.pcm[0], self.flip_iterations, pflip_frequency, pflip_seed)
        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.flipD

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
   
        cdef i

        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        
        zero_syndrome = True
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            self.bpd.converge = True
            return np.zeros(self.n, dtype=syndrome.dtype)
        
        self.flipD.decode(self._syndrome)

        cdef vector[uint8_t] flip_syndrome = self.pcm.mulvec(self.flipD.decoding)
        
        # a = []
        for i in range(self.pcm.m):
            self._syndrome[i] ^= flip_syndrome[i]
            # a.append(flip_syndrome[i])

        # print(a)

        self.bpd.decode(self._syndrome)
        out = np.zeros(self.n, dtype=syndrome.dtype)

        for i in range(self.n):
            out[i] = self.bpd.decoding[i]^self.flipD.decoding[i]

        return out



   

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.osD.osdw_decoding[i]
        return out

    