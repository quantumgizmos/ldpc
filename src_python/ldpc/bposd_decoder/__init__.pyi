import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

class BpOsdDecoder(BpDecoderBase):
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
                 osd_order: int = 0, input_vector_type: str = "syndrome", **kwargs): ...

    def __del__(self): ...

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


    @property
    def osd_order(self) -> int:
        """
        The OSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """


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

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """

    @property
    def bp_decoding(self) -> np.ndarray:
        """
        Returns the current BP decoding output.

        Returns:
            np.ndarray: A numpy array containing the BP decoding output.
        """

    

    @property
    def osd0_decoding(self) -> np.ndarray:
        """
        Returns the current OSD-0 decoding output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """

    @property
    def osdw_decoding(self) -> np.ndarray:
        """
        Returns the current OSD-W decoding output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
