import numpy as np
import scipy.sparse
from ldpc.bposd_decoder import OsdMethod
import warnings
import ldpc.helpers.scipy_helpers

class LsdDecoder():
    """
    A class representing a decoder that combines Belief Propagation (BP) with the Localised Statistics Decoder (LSD) algorithm.

    The BpLsdDecoder is designed to decode binary linear codes by initially attempting BP decoding, and if that fails,
    it falls back to the Localised Statistics Decoder algorithm.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code.
    lsd_order: int, optional
        The order of the LSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
    lsd_method: str, optional
        The LSD method of the LSD algorithm applied to each cluster. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}.
        By default 'LSD_0'.
    
    Notes
    -----
    The BpLsdDecoder is a wrapper around the C++ class `LsdDecoder
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], bits_per_step: int = 1,
                  lsd_order: int = 0,
                  lsd_method: Union[str, int] = 0, **kwargs): ...

    def __del__(self): ...

    def decode(self,syndrome,bit_weights):
        """
        Decodes the input syndrome the LSD decoding method.

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

    @property
    def lsd_method(self) -> Optional[str]:
        """
        The Localized Statistic Decoding (LSD) method used.

        Returns
        -------
        Optional[str]
            A string representing the LSD method used. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}. If no LSD method
            has been set, returns `None`.
        """

    @lsd_method.setter
    def lsd_method(self, method: Union[str, int, float]) -> None:
        """
        Sets the LSD method used. That is, the OSD method per cluster.

        Parameters
        ----------
        method : Union[str, int, float]
            A string, integer or float representing the OSD method to use. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}, corresponding to
            LSD order-0, LSD Exhaustive or LSD-Cominbation-Sweep.
        """


    @property
    def lsd_order(self) -> int:
        """
        The LSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """


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

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
