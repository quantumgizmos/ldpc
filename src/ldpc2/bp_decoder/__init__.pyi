cdef class BpDecoder(BpDecoderBase):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes. The decoder uses a sparse
    parity check matrix to decode received codewords. The decoding algorithm can be configured using various parameters,
    such as the belief propagation method used, the scheduling method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix of the binary linear code, represented as a NumPy array or a SciPy sparse matrix.
    error_rate : Optional[float], optional
        The initial error rate for the decoder, by default None.
    error_channel : Optional[List[float]], optional
        The initial error channel probabilities for the decoder, by default None.
    max_iter : Optional[int], optional
        The maximum number of iterations allowed for decoding, by default 0 (adaptive).
    bp_method : Optional[str], optional
        The belief propagation method to use: 'product_sum' or 'minimum_sum', by default 'minimum_sum'.
    ms_scaling_factor : Optional[float], optional
        The scaling factor for the minimum sum method, by default 1.0.
    schedule : Optional[str], optional
        The scheduling method for belief propagation: 'parallel' or 'serial', by default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads to use, by default 1.
    random_schedule_seed : Optional[int], optional
        The seed for the random serial schedule, by default 0. If set to 0, the seed is set according the system clock.
    serial_schedule_order : Optional[List[int]], optional
        The custom order for serial scheduling, by default None.

    Attributes
    ----------
    pcm : BpSparse
        The internal representation of the parity check matrix in the decoder.
    bp_method : str
        The currently set belief propagation method: 'product_sum' or 'minimum_sum'.
    max_iter : int
        The maximum number of iterations allowed for decoding.
    ms_scaling_factor : float
        The scaling factor for the minimum sum method.
    schedule : str
        The currently set scheduling method for belief propagation: 'parallel' or 'serial'.
    omp_thread_count : int
        The number of OpenMP threads used by the decoder.
    random_schedule_seed : int
        The seed value for random serial scheduling.
    serial_schedule_order : Union[None, np.ndarray]
        The order for serial scheduling, or None if no schedule has been set.

    Methods
    -------
    decode(syndrome: np.ndarray) -> np.ndarray:
        Decode the input syndrome using the belief propagation decoding algorithm.

    Properties
    ----------
    error_rate : np.ndarray
        The current error rate vector.
    error_channel : np.ndarray
        The current error channel vector.
    log_prob_ratios : np.ndarray
        The current log probability ratio vector.
    converge : bool
        Whether the decoder has converged.
    iter : int
        The number of iterations performed by the decoder.
    check_count : int
        The number of rows in the parity check matrix.
    bit_count : int
        The number of columns in the parity check matrix.
    decoding : np.ndarray
        The current decoded output.
    """