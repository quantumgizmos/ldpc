#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

cdef class BpOsdDecoder(BpDecoderBase):
    """
    Belief propagation and Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    This class provides an implementation of the BP decoding that uses Ordered Statistic Decoding (OSD)
    as a fallback method if the BP does not converge. The class inherits from the `BpDecoderBase` class.

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
        The OSD method used.  Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}.
    osd_order : int, optional
        The OSD order, by default 0.

    Notes
    -----
    This class makes use of the C++ module `ldpc::osd::OsdDecoderCpp` for implementing the OSD decoder. The `__cinit__` method
    initializes this module with the parity check matrix and channel probabilities from the belief propagation decoder. The `__del__`
    method deallocates memory if it has been allocated.
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, osd_method: Union[str, int, float] = 0,
                 osd_order: int = 0, input_vector_type: str = "syndrome", **kwargs):
        
        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")
        
        self.MEMORY_ALLOCATED=False

        ## set up OSD with default values and channel probs from BP
        self.osdD = new OsdDecoderCpp(self.pcm[0], OSD_OFF, 0, self.bpd.channel_probabilities)
        self.osd_method=osd_method
        self.osd_order=osd_order

        self.input_vector_type = "syndrome"

        self.osdD.osd_setup()

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.osdD

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decodes the input syndrome using the belief propagation and OSD decoding methods.

        This method takes an input syndrome and decodes it using the belief propagation (BP) decoding method. If the BP
        decoding method converges, it returns the decoding output. Otherwise, the method falls back to using the Ordered
        Statistic Decoding (OSD) decoding method.

        Parameters
        ----------
        syndrome : np.ndarray
            The input syndrome to decode.

        Returns
        -------
        np.ndarray
            A numpy array containing the decoded output.

        Raises
        ------
        ValueError
            If the length of the input syndrome is not equal to the length of the code.

        Notes
        -----
        This method first checks if the input syndrome is all zeros. If it is, it returns an array of zeros of the same
        length as the codeword. If the BP decoding method converges, it returns the decoding output. Otherwise, it falls back
        to using the OSD decoding method. The OSD method used is specified by the `osd_method` parameter passed to the class
        constructor. The OSD order used is specified by the `osd_order` parameter passed to the class constructor.

        """

        cdef int i

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
        
        self.bpd.decode(self._syndrome)
        out = np.zeros(self.n, dtype=syndrome.dtype)

        if self.bpd.converge:
            for i in range(self.n):
                out[i] = self.bpd.decoding[i]
        else:
            self.osdD.decode(self._syndrome, self.bpd.log_prob_ratios)
            for i in range(self.n):
                out[i] = self.osdD.osdw_decoding[i]

        return out



    @property
    def osd_method(self) -> Optional[str]:
        """
        The Ordered Statistic Decoding (OSD) method used.

        Returns
        -------
        Optional[str]
            A string representing the OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}. If no OSD method
            has been set, returns `None`.
        """
        if self.osdD.osd_method == OSD_0:
            return 'OSD_0'
        elif self.osdD.osd_method == EXHAUSTIVE:
            return 'OSD_E'
        elif self.osdD.osd_method == COMBINATION_SWEEP:
            return 'OSD_CS'
        elif self.osdD.osd_method == OSD_OFF:
            return 'OSD_OFF'
        else:
            return None

    @osd_method.setter
    def osd_method(self, method: Union[str, int, float]) -> None:
        """
        Sets the OSD method used.

        Parameters
        ----------
        method : Union[str, int, float]
            A string, integer or float representing the OSD method to use. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}, corresponding to
            OSD order-0, OSD Exhaustive or OSD-Cominbation-Sweep.
        """
        # OSD method
        if str(method).lower() in ['osd_0', '0', 'osd0']:
            self.osdD.osd_method = OSD_0
            self.osdD.osd_order = 0
        elif str(method).lower() in ['osd_e', 'e', 'exhaustive']:
            self.osdD.osd_method = EXHAUSTIVE
        elif str(method).lower() in ['osd_cs', '1', 'cs', 'combination_sweep']:
            self.osdD.osd_method = COMBINATION_SWEEP
        elif str(method).lower() in ['off', 'osd_off', 'deactivated', -1]:
            self.osdD.osd_method = OSD_OFF
        else:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:\
                'OSD_0', 'OSD_E' or 'OSD_CS'.")


    @property
    def osd_order(self) -> int:
        """
        The OSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """
        return self.osdD.osd_order


    @osd_order.setter
    def osd_order(self, order: int) -> None:
        """
        Set the order for the OSD method.

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
            If the OSD method is 'OSD_E' and the order is greater than 15.

        """
        # OSD order
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")

        if self.osdD.osd_method == OSD_0 and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")

        if self.osdD.osd_method == EXHAUSTIVE and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not "
                        "recommended. Use the 'osd_cs' method instead.")

        self.osdD.osd_order = order

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
    def bp_decoding(self) -> np.ndarray:
        """
        Returns the current BP decoding output.

        Returns:
            np.ndarray: A numpy array containing the BP decoding output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out

    

    @property
    def osd0_decoding(self) -> np.ndarray:
        """
        Returns the current OSD-0 decoding output.

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
        Returns the current OSD-W decoding output.

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


# cdef class SoftInfoBpOsdDecoder(BpDecoderBase):
#     """
#     Belief propagation and Ordered Statistic Decoding (OSD) decoder for binary linear codes.

#     This class provides an implementation of the BP decoding that uses Ordered Statistic Decoding (OSD)
#     as a fallback method if the BP does not converge. The class inherits from the `BpDecoderBase` class.

#     Parameters
#     ----------
#     pcm : Union[np.ndarray, spmatrix]
#         The parity check matrix for the code.
#     error_rate : Optional[float], optional
#         The probability of a bit being flipped in the received codeword, by default None.
#     error_channel : Optional[List[float]], optional
#         A list of probabilities that specify the probability of each bit being flipped in the received codeword.
#         Must be of length equal to the block length of the code, by default None.
#     max_iter : Optional[int], optional
#         The maximum number of iterations for the decoding algorithm, by default 0.
#     ms_scaling_factor : Optional[float], optional
#         The scaling factor used in the minimum sum method, by default 1.0.
#     omp_thread_count : Optional[int], optional
#         The number of OpenMP threads used for parallel decoding, by default 1.
#     random_serial_schedule : Optional[int], optional
#         Whether to use a random serial schedule order, by default False.
#     serial_schedule_order : Optional[List[int]], optional
#         A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code,
#         by default None.
#     osd_method : int, optional
#         The OSD method used. Must be one of {0, 1, 2}, where 0 represents 'OSD_0', 1 represents 'OSD_E', and 2 represents 'OSD_CS',
#         by default 0.
#     osd_order : int, optional
#         The OSD order, by default 0.
    

#     Notes
#     -----
#     This class makes use of the C++ module `ldpc::osd::OsdDecoderCpp` for implementing the OSD decoder. The `__cinit__` method
#     initializes this module with the parity check matrix and channel probabilities from the belief propagation decoder. The `__del__`
#     method deallocates memory if it has been allocated.
#     """

#     def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
#                  error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
#                  ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
#                  random_serial_schedule: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None, osd_method: int = 0,
#                  osd_order: int = 0, cutoff: Optional[float] = np.inf, sigma: float = 2.0):
        
#         self.MEMORY_ALLOCATED=False

#         ## set up OSD with default values and channel probs from BP
#         self.osdD = new OsdDecoderCpp(self.pcm, -1, 0, self.bpd.channel_probs)
#         self.osd_order=osd_order
#         self.osd_method=osd_method
#         self.osdD.osd_setup()


#         self.cutoff = cutoff
#         if not isinstance(sigma,float) or sigma <= 0:
#             raise ValueError("The sigma value must be a float greater than 0.")
#         self.sigma = sigma
#         self.schedule = "serial"
#         self.bp_method = "minimum_sum"

#         self.MEMORY_ALLOCATED=True

#     def __del__(self):
#         if self.MEMORY_ALLOCATED:
#             del self.osd

#     def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
#         """
#         Decodes the input syndrome using the belief propagation and OSD decoding methods.

#         This method takes an input syndrome and decodes it using the belief propagation (BP) decoding method. If the BP
#         decoding method converges, it returns the decoding output. Otherwise, the method falls back to using the Ordered
#         Statistic Decoding (OSD) decoding method.

#         Parameters
#         ----------
#         syndrome : np.ndarray
#             The input syndrome to decode.

#         Returns
#         -------
#         np.ndarray
#             A numpy array containing the decoded output.

#         Raises
#         ------
#         ValueError
#             If the length of the input syndrome is not equal to the length of the code.

#         Notes
#         -----
#         This method first checks if the input syndrome is all zeros. If it is, it returns an array of zeros of the same
#         length as the codeword. If the BP decoding method converges, it returns the decoding output. Otherwise, it falls back
#         to using the OSD decoding method. The OSD method used is specified by the `osd_method` parameter passed to the class
#         constructor. The OSD order used is specified by the `osd_order` parameter passed to the class constructor.

#         """
#         if not len(soft_info_syndrome) == self.m:
#             raise ValueError(f"The syndrome must have length {self.m}. Not {len(soft_info_syndrome)}.")
        
#         cdef vector[np.float64_t] soft_syndrome
#         soft_syndrome.resize(self.m)
#         for i in range(self.m):
#             soft_syndrome[i] = soft_info_syndrome[i]
        
        
#         # zero_syndrome = True
        
#         # for i in range(self.m):
#         #     self._syndrome[i] = syndrome[i]
#         #     if self._syndrome[i]:
#         #         zero_syndrome = False
#         # if zero_syndrome:
#         #     return np.zeros(self.n, dtype=syndrome.dtype)
        
#         self.bpd.soft_info_decode_serial(soft_syndrome, self.cutoff, self.sigma)
#         out = np.zeros(self.n, dtype=np.uint8)

#         if self.bpd.converge:
#             for i in range(self.n):
#                 out[i] = self.bpd.decoding[i]
#         else:
#             for i in range(self.m): ##this copies out the value of the soft syndrome and translates it to a hard syndrome
#                 if self.bpd.soft_syndrome[i]<=0:
#                     self._syndrome[i] = 1
#                 else:
#                     self._syndrome[i] = 0

#             self.osdD.decode(self._syndrome, self.bpd.log_prob_ratios)
#             for i in range(self.n):
#                 out[i] = self.osdD.osdw_decoding[i]

#         return out



#     @property
#     def osd_method(self) -> Optional[str]:
#         """
#         The Ordered Statistic Decoding (OSD) method used.

#         Returns
#         -------
#         Optional[str]
#             A string representing the OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}. If no OSD method
#             has been set, returns `None`.
#         """
#         if self.osdD.osd_method == 0:
#             return 'OSD_0'
#         elif self.osdD.osd_method == 1:
#             return 'OSD_E'
#         elif self.osdD.osd_method == 2:
#             return 'OSD_CS'
#         else:
#             return None


#     @osd_method.setter
#     def osd_method(self, method: Union[str, int, float]) -> None:
#         """
#         Sets the OSD method used.

#         Parameters
#         ----------
#         method : Union[str, int, float]
#             A string, integer or float representing the OSD method to use. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}.
#             Alternatively, 'osd_0', '0' or 'osd0' can be used to set the OSD method to 'OSD_0', 'osd_e', 'e' or
#             'exhaustive' can be used to set the OSD method to 'OSD_E', and 'osd_cs', '1', 'cs', 'combination_sweep' or
#             'combination_sweep' can be used to set the OSD method to 'OSD_CS'.
#         """
#         # OSD method
#         if str(method).lower() in ['osd_0', '0', 'osd0']:
#             self.osdD.osd_method = 0
#             self.osdD.osd_order = 0
#         elif str(method).lower() in ['osd_e', 'e', 'exhaustive']:
#             self.osdD.osd_method = 1
#         elif str(method).lower() in ['osd_cs', '1', 'cs', 'combination_sweep']:
#             self.osdD.osd_method = 2
#         else:
#             raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:\
#                 'OSD_0', 'OSD_E' or 'OSD_CS'.")


#     @property
#     def osd_order(self) -> int:
#         """
#         The OSD order used.

#         Returns
#         -------
#         int
#             An integer representing the OSD order used.
#         """
#         return self.osdD.osd_order


#     @osd_order.setter
#     def osd_order(self, order: int) -> None:
#         """
#         Set the order for the OSD method.

#         Parameters
#         ----------
#         order : int
#             The order for the OSD method. Must be a positive integer.

#         Raises
#         ------
#         ValueError
#             If order is less than 0.

#         Warns
#         -----
#         UserWarning
#             If the OSD method is 'OSD_E' and the order is greater than 15.

#         """
#         # OSD order
#         if order < 0:
#             raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")

#         if self.osdD.osd_method == 0 and order > 15:
#             warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not "
#                         "recommended. Use the 'osd_cs' method instead.")

#         self.osdD.osd_order = order

#     @property
#     def decoding(self) -> np.ndarray:
#         """
#         Returns the current decoded output.

#         Returns:
#             np.ndarray: A numpy array containing the current decoded output.
#         """
#         out = np.zeros(self.n).astype(int)
#         for i in range(self.n):
#             out[i] = self.osD.osdw_decoding[i]
#         return out

#     @property
#     def osd0_decoding(self) -> np.ndarray:
#         """
#         Returns the current decoded output.

#         Returns:
#             np.ndarray: A numpy array containing the current decoded output.
#         """
#         out = np.zeros(self.n).astype(int)

#         if self.bpd.converge:
#             for i in range(self.n):
#                 out[i] = self.bpd.decoding[i]
#             return out

#         for i in range(self.n):
#             out[i] = self.osdD.osd0_decoding[i]
#         return out

#     @property
#     def osdw_decoding(self) -> np.ndarray:
#         """
#         Returns the current decoded output.

#         Returns:
#             np.ndarray: A numpy array containing the current decoded output.
#         """
#         out = np.zeros(self.n).astype(int)

#         if self.bpd.converge:
#             for i in range(self.n):
#                 out[i] = self.bpd.decoding[i]
#             return out

#         for i in range(self.n):
#             out[i] = self.osdD.osdw_decoding[i]
#         return out

# bposd_decoder = BpOsdDecoder