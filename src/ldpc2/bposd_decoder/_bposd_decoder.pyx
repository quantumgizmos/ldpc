#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

cdef class BpOsdDecoder(BpDecoderBase):
    """
    Belief propagation and Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    This class provides an implementation of an OSD decoder for binary linear codes that uses belief propagation decoding
    as a fallback method if the OSD does not converge. The class inherits from the `BpDecoderBase` class.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float], optional
        The probability of a bit being flipped in the received codeword, by default None.
    error_channel : Optional[List[float]], optional
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations for the decoding algorithm, by default 0.
    bp_method : Optional[str], optional
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor used in the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method used. Must be one of {'parallel', 'serial'}, by default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads used for parallel decoding, by default 1.
    random_serial_schedule : Optional[int], optional
        Whether to use a random serial schedule order, by default False.
    serial_schedule_order : Optional[List[int]], optional
        A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code,
        by default None.
    osd_method : int, optional
        The OSD method used. Must be one of {0, 1, 2}, where 0 represents 'OSD_0', 1 represents 'OSD_E', and 2 represents 'OSD_CS',
        by default 0.
    osd_order : int, optional
        The OSD order, by default 0.

    Notes
    -----
    This class makes use of the C++ module `osd::OsdDecoderCpp` for implementing the OSD decoder. The `__cinit__` method
    initializes this module with the parity check matrix and channel probabilities from the belief propagation decoder. The `__del__`
    method deallocates memory if it has been allocated.
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_serial_schedule: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None, osd_method: int = 0,
                 osd_order: int = 0):
        
        self.MEMORY_ALLOCATED=False

        ## set up OSD with default values and channel probs from BP
        self.osdD = new OsdDecoderCpp(self.pcm, -1, 0, self.bpd.channel_probs)
        self.osd_order=int(osd_order)
        self.osd_method=int(osd_method)

        self.osdD.osd_setup()

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.osd

    def decode(self,syndrome):
        
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

        if(self.bpd.converge):
            for i in range(self.n): out[i] = self.bpd.decoding[i]
        else:
            self.osdD.decode(self._syndrome, self.bpd.log_prob_ratios)
            for i in range(self.n): out[i] = self.osdD.osdw_decoding[i]

        return out


    @property
    def osd_method(self):
        if self.osdD.osd_method==0:
            return 'OSD_0'
        elif self.osdD.osd_method==1:
            return 'OSD_E'
        elif self.osdD.osd_method==2:
            return 'OSD_CS'
        else:
            return None

    @osd_method.setter
    def osd_method(self, method: Union[str, int, float]):
        # OSD method
        if str(method).lower() in ['OSD_0','osd_0','0','osd0']:
            self.osdD.osd_method=0
            self.osdD.osd_order=0
        elif str(method).lower() in ['osd_e','osde','exhaustive','e']:
            self.osdD.osd_method=1
        elif str(method).lower() in ['osd_cs','1','osdcs','combination_sweep','combination_sweep','cs']:
            self.osdD.osd_method=2
        else:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following\
            methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")

    @property
    def osd_order(self):
        return self.osdD.osd_order

    @osd_order.setter
    def osd_order(self, order: int):
        # OSD order
        if order<0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")
                
        if self.osdD.osd_method == 0 and order>15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth\
            greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        
        self.osdD.osd_order=order

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

    @property
    def osd0_decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)

        if self.bpd.converge:
            for i in range(self.n):
                out[i] = self.bpd.decoding[i]
            return out

        for i in range(self.n):
            out[i] = self.osdD.osd0_decoding[i]
        return out

    @property
    def osdw_decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)

        if self.bpd.converge:
            for i in range(self.n):
                out[i] = self.bpd.decoding[i]
            return out

        for i in range(self.n):
            out[i] = self.osdD.osdw_decoding[i]
        return out

bposd_decoder = BpOsdDecoder