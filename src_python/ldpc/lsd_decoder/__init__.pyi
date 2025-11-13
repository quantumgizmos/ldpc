import numpy as np
import scipy.sparse
from ldpc.bposd_decoder import OsdMethod
import warnings
import ldpc.helpers.scipy_helpers

class LsdDecoder():
    """
    A class implementing the Localised Statistics Decoder (LSD) in isolation.

    The LsdDecoder class provides an interface to directly decode a syndrome using the LSD algorithm, without a preceding Belief Propagation (BP) stage. The user provides the syndrome and bit weights as inputs to produce the decoded output.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code.
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the LSD algorithm. The default value is `1`.
    lsd_order: int, optional
        The order of the LSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
    lsd_method: str or int, optional
        The LSD method of the LSD algorithm applied to each cluster. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'} or {0, 1, 2}. By default 'LSD_0'.

    Notes
    -----
    The LsdDecoder class leverages soft information (bit weights) provided by the user to guide the cluster growth in the LSD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], bits_per_step: int = 1,
                  lsd_order: int = 0,
                  lsd_method: Union[str, int] = 0, **kwargs): ...

    def __dealloc__(self): ...

    def decode(self,syndrome,bit_weights):
        """
        Decodes the input syndrome using the LSD algorithm in isolation.

        This method directly invokes the LSD decoding routine without attempting any BP decoding first.
        The provided bit weights are used as input for the LSD decoder.

        Parameters:
            syndrome : np.ndarray
                A 1D numpy array (dtype=np.uint8) representing the syndrome. Its length must equal the number of rows in the parity check matrix.
            bit_weights : list or np.ndarray
                A list or 1D numpy array of doubles, with length equal to the number of columns in the parity check matrix.

        Returns:
            np.ndarray
                A 1D numpy array (dtype=np.uint8) containing the decoded output.

        Raises:
            ValueError: If the length of syndrome or bit_weights does not match the expected dimensions.
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
