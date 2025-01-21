import numpy as np
from scipy.sparse import spmatrix

class BeliefFindDecoder(BpDecoderBase):
  
    """
    A class representing a decoder that combines Belief Propagation (BP) with the Union Find Decoder (UFD) algorithm.

    The BeliefFindDecoder is designed to decode binary linear codes by initially attempting BP decoding, and if that fails,
    it falls back to the Union Find Decoder algorithm. The UFD algorithm is based on the principles outlined in
    https://arxiv.org/abs/1709.06218, with an option to utilise a more general version as described in
    https://arxiv.org/abs/2103.08049 for LDPC codes by setting `uf_method=True`.

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
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads used for parallel decoding, by default 1.
    random_schedule_seed : Optional[int], optional
        Whether to use a random serial schedule order, by default 0.
    serial_schedule_order : Optional[List[int]], optional
        A list of integers specifying the serial schedule order. Must be of length equal to the block length of the code,
        by default None.
    uf_method : str, optional
        The method used to solve the local decoding problem in each cluster. Choose from: 1) 'inversion' or 2) 'peeling'.
        By default set to 'inversion'. The 'peeling' method is only suitable for LDPC codes with point like syndromes.
        The inversion method can be applied to any parity check matrix.
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the UFD algorithm. If no value is provided, this is set the block length of the code.

    Notes
    -----
    The `BeliefFindDecoder` class leverages soft information outputted by the BP decoder to guide the cluster growth
    in the UFD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    The `uf_method` parameter activates a more general version of the UFD algorithm suitable for LDPC codes when set to True.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, uf_method: str = "peeling", bits_per_step:int = 0, input_vector_type: str = "syndrome"): ...

    def __del__(self): ...

    def decode(self,syndrome):

        """
        Decodes the input syndrome using the belief propagation and UFD decoding methods.

        Initially, the method attempts to decode the syndrome using belief propagation. If this fails to converge,
        it falls back to the UFD algorithm.

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
    def uf_method(self): ...
