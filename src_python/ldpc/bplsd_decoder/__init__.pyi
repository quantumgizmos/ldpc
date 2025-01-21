import numpy as np
from scipy.sparse import spmatrix
import json
from ldpc.bposd_decoder import OsdMethod
import warnings

class BpLsdDecoder(BpDecoderBase):
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
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the LSD algorithm. If no value is provided, this is set the block length of the code.
    lsd_order: int, optional
        The order of the LSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
    lsd_method: str, optional
        The LSD method of the LSD algorithm applied to each cluster. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}.
        By default 'LSD_0'.
    
    Notes
    -----
    The `BpLsdDecoder` class leverages soft information outputted by the BP decoder to guide the cluster growth
    in the LSD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel',
                 omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0,
                 serial_schedule_order: Optional[List[int]] = None,
                  bits_per_step:int = 1,
                  input_vector_type: str = "syndrome",
                  lsd_order: int = 0,
                  lsd_method: Union[str, int] = 0, **kwargs): ...

    def __del__(self): ...

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

    @property
    def statistics(self) -> Statistics:
        """
        Returns the statistics for the LSD algorithm.
        May be None if the statistics are not being collected.
        -------
        Statistics
            The statistics object.
        """

    @property
    def do_stats(self) -> bool:
        """
        Returns whether the statistics are being collected.

        Returns
        -------
        bool
            Whether the statistics are being collected.
        """

    def set_do_stats(self, value: bool) -> None:
        """
        Sets whether the statistics are being collected.

        Parameters
        ----------
        value : bool
            Whether the statistics are being collected.
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

    def set_additional_stat_fields(self, error, syndrome, compare_recover) -> None:
        """
        Sets additional fields to be collected in the statistics.

        Parameters
        ----------
        fields : List[str]
            A list of strings representing the additional fields to be collected in the statistics.
        """

    def reset_cluster_stats(self) -> None:
        """
        Resets cluster statistics of the decoder.
        Note that this also resets the additional stat fields, such as the error, and compare_recovery vectors
        """
