#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix
import json
from ldpc.bposd_decoder cimport OsdMethod
import warnings

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
    lsd_order: int, optional
        The order of the OSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
    lsd_method: OsdMethod
        The OSD method of the OSD algorithm applied to each cluster. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}.
        By default 'OSD_0'.
    Notes
    -----
    The `BpLsdDecoder` class leverages soft information outputted by the BP decoder to guide the cluster growth
    in the LSD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel',
                 omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0,
                 serial_schedule_order: Optional[List[int]] = None,
                  bits_per_step:int = 1,
                  input_vector_type: str = "syndrome",
                  lsd_order: int = 0,
                  lsd_method: Union[str, int] = 0, **kwargs):

        # compatability with osd_method/osd_order
        if "osd_method" in kwargs:
            lsd_method = kwargs["osd_method"]
        else:
            lsd_method = lsd_method

        if "osd_order" in kwargs:
            lsd_order = kwargs["osd_order"]
        else:
            lsd_order = lsd_order
        if(lsd_order < 0):
            raise ValueError(f"lsd_order must be greater than or equal to 0. Not {lsd_order}.")

        if isinstance(lsd_method, str):
            if lsd_method.lower() not in ["osd_0", "osd_e", "osd_cs"]:
                raise ValueError(f"lsd_method must be one of 'OSD_0', 'OSD_E', 'OSD_CS'. Not {lsd_method}.")
        elif isinstance(lsd_method, int):
            if lsd_method not in [0, 1, 2]:
                raise ValueError(f"lsd_method must be one of 0, 1, 2. Not {lsd_method}.")
        else:
            raise ValueError(f"lsd_method must be one of 'OSD_0' (0), 'OSD_E' (1), 'OSD_CS' (2). Not {lsd_method}.")

        self.MEMORY_ALLOCATED = False
        self.lsd = new LsdDecoderCpp(pcm=self.pcm[0], lsd_method=OsdMethod.OSD_0, lsd_order=lsd_order)
        self.bplsd_decoding.resize(self.n) #C vector for the bf decoding
        self.lsd_method = lsd_method

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
            self.lsd.decoding = self.lsd.lsd_decode(self._syndrome, self.bpd.log_prob_ratios,self.bits_per_step, True)
            for i in range(self.n):
                out[i] = self.lsd.decoding[i]
        
        return out

    @property
    def statistics(self) -> Statistics:
        """
        Returns the statistics for the LSD algorithm.
        May be None if the statistics are not being collected.
        -------
        Statistics
            The statistics object.
        """
        return self.lsd.statistics

    @property
    def do_stats(self) -> bool:
        """
        Returns whether the statistics are being collected.

        Returns
        -------
        bool
            Whether the statistics are being collected.
        """

        return self.lsd.get_do_stats()

    def set_do_stats(self, value: bool) -> None:
        """
        Sets whether the statistics are being collected.

        Parameters
        ----------
        value : bool
            Whether the statistics are being collected.
        """

        self.lsd.set_do_stats(value)

    @property
    def lsd_method(self) -> Optional[str]:
        """
        The Ordered Statistic Decoding (OSD) method used.

        Returns
        -------
        Optional[str]
            A string representing the OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}. If no OSD method
            has been set, returns `None`.
        """
        if self.lsd.lsd_method == OsdMethod.OSD_0:
            return 'OSD_0'
        elif self.lsd.lsd_method == OsdMethod.EXHAUSTIVE:
            return 'OSD_E'
        elif self.lsd.lsd_method == OsdMethod.COMBINATION_SWEEP:
            return 'OSD_CS'
        elif self.lsd.lsd_method == OsdMethod.OSD_OFF:
            return 'OSD_OFF'
        else:
            return None

    @lsd_method.setter
    def lsd_method(self, method: Union[str, int, float]) -> None:
        """
        Sets the LSD method used. That is, the OSD method per cluster.

        Parameters
        ----------
        method : Union[str, int, float]
            A string, integer or float representing the OSD method to use. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}, corresponding to
            OSD order-0, OSD Exhaustive or OSD-Cominbation-Sweep.
        """
        # OSD method
        if str(method).lower() in ['osd_0', '0', 'osd0']:
            self.lsd.lsd_method = OsdMethod.OSD_0
            self.lsd.lsd_order = 0
        elif str(method).lower() in ['osd_e', 'e', 'exhaustive']:
            self.lsd.lsd_method = OsdMethod.EXHAUSTIVE
        elif str(method).lower() in ['osd_cs', '1', 'cs', 'combination_sweep']:
            self.lsd.lsd_method = OsdMethod.COMBINATION_SWEEP
        elif str(method).lower() in ['off', 'osd_off', 'deactivated', -1]:
            self.lsd.lsd_method = OsdMethod.OSD_OFF
        else:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:\
                'OSD_0', 'OSD_E' or 'OSD_CS'.")


    @property
    def lsd_order(self) -> int:
        """
        The LSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """
        return self.lsd.lsd_order


    @lsd_order.setter
    def lsd_order(self, order: int) -> None:
        """
        Set the order for the LSD method.

        Parameters
        ----------
        order : int
            The order for the OSD method. Must be a positive integer.

        Raises
        ------
        ValueError
            If order is less than 0.

        Warns
        -----
        UserWarning
            If the LSD method is 'OSD_E' and the order is greater than 15.

        """
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")

        if self.lsd.lsd_method == OsdMethod.OSD_0 and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")

        if self.lsd.lsd_method == OsdMethod.EXHAUSTIVE and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not "
                        "recommended. Use the 'osd_cs' method instead.")

        self.lsd.lsd_order = order
