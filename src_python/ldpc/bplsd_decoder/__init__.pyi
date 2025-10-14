import numpy as np
from scipy.sparse import spmatrix
import json
from ldpc.bposd_decoder import OsdMethod
import warnings

class BpLsdDecoder(BpDecoderBase):
    """
    A decoder that combines Belief Propagation (BP) with the Localised Statistics Decoder (LSD) algorithm.

    The BpLsdDecoder is a hybrid decoder that first attempts standard BP decoding. If BP fails to 
    converge to a valid codeword, it falls back to the Localised Statistics Decoder (LSD) algorithm,
    which applies OSD-style decoding to local clusters of the code graph.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code. Must be a binary matrix.
    error_rate : Optional[float], optional
        The probability of a bit being flipped in the received codeword. 
        Mutually exclusive with `error_channel`. Default is None.
    error_channel : Optional[List[float]], optional
        A list of probabilities specifying the probability of each bit being flipped.
        Must have length equal to the block length of the code. 
        Mutually exclusive with `error_rate`. Default is None.
    max_iter : Optional[int], optional
        The maximum number of iterations for the BP decoding algorithm. 
        If 0, defaults to the block length. Default is 0.
    bp_method : Optional[str], optional
        The belief propagation method used. Must be one of:
        - 'product_sum': Product-sum algorithm (exact)
        - 'minimum_sum': Minimum-sum algorithm (approximation, faster)
        Default is 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor used in the minimum-sum method. Values < 1.0 typically
        improve performance. Default is 1.0.
    schedule : Optional[str], optional
        The scheduling method for BP updates. Must be one of:
        - 'parallel': All messages updated simultaneously
        - 'serial': Messages updated sequentially
        Default is 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads for parallel processing. Currently not implemented.
        Default is 1.
    random_schedule_seed : Optional[int], optional
        Seed for random serial scheduling. Only used if random scheduling is enabled.
        Default is 0.
    serial_schedule_order : Optional[List[int]], optional
        Custom order for serial scheduling. Must have length equal to the block length.
        Default is None (uses default ordering).
    bits_per_step : int, optional
        Number of bits added to each cluster in each step of the LSD algorithm.
        Controls the trade-off between performance and complexity. Default is 1.
    lsd_order : int, optional
        The order of the OSD algorithm applied to each cluster. Must be >= 0.
        Higher orders provide better performance but increased complexity. Default is 0.
    lsd_method : Union[str, int], optional
        The OSD method applied to each cluster. Must be one of:
        - 'LSD_0' or 0: Order-0 decoding (fastest)
        - 'LSD_E' or 'exhaustive': Exhaustive search (most powerful)
        - 'LSD_CS' or 'combination_sweep': Combination sweep method
        Default is 'LSD_0'.

    Attributes
    ----------
    bits_per_step : int
        Number of bits added to clusters in each LSD step.
    statistics : Statistics
        Statistics object containing decoding performance metrics (if enabled).

    Notes
    -----
    The LSD algorithm works by:
    1. Using BP log probability ratios to identify unreliable bits
    2. Growing clusters around these unreliable bits
    3. Applying OSD decoding to each cluster
    4. Combining cluster solutions to form the final decoding

    The `bits_per_step` parameter controls the cluster growth rate. Smaller values
    lead to more clusters but potentially better performance. Larger values reduce
    computational complexity but may hurt performance.

    Examples
    --------
    >>> import numpy as np
    >>> from ldpc.bplsd_decoder import BpLsdDecoder
    >>> 
    >>> # Create a parity check matrix
    >>> pcm = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.uint8)
    >>> 
    >>> # Initialize decoder with LSD fallback
    >>> decoder = BpLsdDecoder(pcm, error_rate=0.1, lsd_order=1, bits_per_step=2)
    >>> 
    >>> # Decode a syndrome
    >>> syndrome = np.array([1, 0], dtype=np.uint8)
    >>> decoding = decoder.decode(syndrome)
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
