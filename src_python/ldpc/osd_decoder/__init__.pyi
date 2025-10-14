import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional
import scipy.sparse
import ldpc.helpers.scipy_helpers


class OsdDecoder():
    """
    Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    This class provides an implementation of Ordered Statistic Decoding (OSD) for binary linear codes.
    OSD is a post-processing technique that can be used after belief propagation (BP) decoding to 
    improve decoding performance by performing an exhaustive search over the most reliable bit positions.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code. Must be a binary matrix.
    osd_method : Union[str, int, float], optional
        The OSD method to use. Must be one of:
        - 'OSD_0' or 0: OSD order-0 (fastest, least powerful)
        - 'OSD_E' or 'exhaustive': Exhaustive search (most powerful, slowest)
        - 'OSD_CS' or 'combination_sweep': Combination sweep method (balanced performance)
        Default is 0 (OSD_0).
    osd_order : int, optional
        The order of the OSD algorithm, which determines the number of least reliable
        bits to consider for higher-order osd. Must be a non-negative integer.
        For OSD_0, this must be 0. For OSD_E, values > 15 are not recommended.
        Default is 0.
    channel_probabilities : np.ndarray, optional
        Channel error probabilities for the intial error distribution. Used in the weighted sum
        during the higher-order OSD search. Must have length equal to the number of columns.

    Notes
    -----
    This class uses the C++ implementation `ldpc::osd::OsdDecoderCpp` for efficient
    OSD decoding. The decoder requires log probability ratios from a previous BP
    decoding attempt to determine the reliability ordering of bits.

    The OSD algorithm works by:
    1. Ordering bits by their reliability (from log probability ratios)
    2. Selecting the most reliable independent set of bits
    3. Performing Gaussian elimination to solve for the remaining bits
    4. Optionally searching over error patterns in the least reliable bits
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix],  osd_method: Union[str, int, float] = 0,
                 osd_order: int = 0, channel_probabilities: np.ndarray = None): ...

    def __del__(self): ...

    def decode(self, syndrome: np.ndarray, log_prob_ratios: np.ndarray) -> np.ndarray:
        """
        Decodes the input syndrome using the belief propagation and OSD decoding methods.

        This method takes an input syndrome and decodes ins using OSD.

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
        length as the codeword. The OSD method used is specified by the `osd_method` parameter passed to the class
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
