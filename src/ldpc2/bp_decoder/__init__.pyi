#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix
from typing import Optional, List, Union

class BpDecoder(bp_decoder_base):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes.
    The decoder uses a sparse parity check matrix to decode received codewords. The decoding algorithm
    can be configured using various parameters, such as the belief propagation method used, the scheduling
    method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm: np.ndarray or scipy.sparse.spmatrix
        The parity check matrix for the code.
    error_rate: Optional[float]
        The probability of a bit being flipped in the received codeword.
    error_channel: Optional[List[float]]
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code.
    max_iter: Optional[int]
        The maximum number of iterations for the decoding algorithm.
    bp_method: Optional[str]
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}.
    ms_scaling_factor: Optional[float]
        The scaling factor used in the minimum sum method.
    schedule: Optional[str]
        The scheduling method used. Must be one of {'parallel', 'serial'}.
    omp_thread_count: Optional[int]
        The number of OpenMP threads used for parallel decoding.
    random_serial_schedule: Optional[int]
        Whether to use a random serial schedule order.
    serial_schedule_order: Optional[List[int]]
        A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code.
    """

    def __init__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_serial_schedule: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None):
        
        pass

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the input syndrome using belief propagation decoding algorithm.

        Parameters
        ----------
        syndrome : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input syndrome does not match the number of rows in the parity check matrix.
        """
        
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        cdef bool zero_syndrome = True
        DTYPE = syndrome.dtype
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]: zero_syndrome = False
        if zero_syndrome: return np.zeros(self.n,dtype=DTYPE)
        
        self.bpd.decode(self._syndrome)
        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out

    @property
    def decoding(self):
        out = np.zeros(self.n).astype(int)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out
    
bp_decoder = BpDecoder






