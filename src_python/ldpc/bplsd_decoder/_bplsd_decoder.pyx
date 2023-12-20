#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class BpLsdDecoder(BpDecoderBase):
  
    """
    A class representing a decoder that combines Belief Propagation (BP) with the Localised Statistics Decoder (LSD) algorithm.

    The BpLsdDecoder is designed to decode binary linear codes by initially attempting BP decoding, and if that fails,
    it falls back to the Localised Statistics Decoder algorithm.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float], optional
        The probability of a bit being flipped in the received codeword, by default None.
    error_channel : Optional[List[float]], optional
        A list of probabilities specifying the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations for the decoding algorithm, by default 0.
    bp_method : Optional[str], optional
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor used in the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method used. Must be one of {'parallel', 'serial'}, by default 'parallel'.
    omp_thread_count : Optional[int], optional, NotImplemented
        The number of OpenMP threads used for parallel decoding, by default 1.
    random_schedule_seed : Optional[int], optional
        Whether to use a random serial schedule order, by default 0.
    serial_schedule_order : Optional[List[int]], optional
        A list of integers specifying the serial schedule order. Must be of length equal to the block length of the code,
        by default None.
    bits_per_step : int, optional, NotImplemented
        Specifies the number of bits added to the cluster in each step of the LSD algorithm. If no value is provided, this is set the block length of the code.

    Notes
    -----
    The `BpLsdDecoder` class leverages soft information outputted by the BP decoder to guide the cluster growth
    in the LSD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, bits_per_step:int = 1, input_vector_type: str = "syndrome"):
        self.MEMORY_ALLOCATED=False
        self.lsd = new lsd_decoder_cpp(self.pcm[0])
        self.bplsd_decoding.resize(self.n) #C vector for the bf decoding
        # self.residual_syndrome.resize(self.m) #C vector for the bf decoding
        if bits_per_step == 0:
            self.bits_per_step = pcm.shape[1]
        else:
            self.bits_per_step = bits_per_step
        self.input_vector_type = "syndrome"

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.lsd

    def decode(self,syndrome):

        """
        Decodes the input syndrome using the belief propagation and LSD decoding methods.

        Initially, the method attempts to decode the syndrome using belief propagation. If this fails to converge,
        it falls back to the LSD algorithm.

        Parameters
        ----------
        syndrome : np.ndarray
            The input syndrome to decode.

        Returns
        -------
        np.ndarray
            The decoded output.

        Raises
        ------
        ValueError
            If the length of the input syndrome is not equal to the length of the code.
        """

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
            self.lsd.decoding = self.lsd.lsd_decode(self._syndrome, self.bpd.log_prob_ratios,self.bits_per_step)
            for i in range(self.n):
                out[i] = self.lsd.decoding[i]
        
        return out

    @property
    def cluster_size_stats(self):
        """
        Returns the cluster size statistics for the LSD algorithm.

        Returns
        -------
        np.ndarray
            The cluster size statistics.
        """

        cdef vector[int] clss = self.lsd.cluster_size_stats
        cdef int i
        out = np.zeros(clss.size(),dtype=int)
        for i in range(clss.size()):
            out[i] = clss[i]
        return out

        return self.lsd.cluster_size_stats