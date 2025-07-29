import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional

class BpFlipDecoder(BpDecoderBase):
    """
    A class representing a decoder that combines Belief Propagation (BP) with a flipping algorithm.

    This decoder performs iterative decoding on a given parity-check matrix using the belief propagation
    algorithm combined with a flipping strategy to correct errors. The class is initialized with the
    parity-check matrix and various decoding parameters.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity-check matrix, can be a dense (numpy.ndarray) or sparse (scipy.sparse) matrix.
    error_rate : Optional[float], optional
        The expected error rate of the channel, by default None
    error_channel : Optional[List[float]], optional
        A list representing the error channel, by default None
    max_iter : Optional[int], optional
        The maximum number of iterations for the decoding process, by default 0
    bp_method : Optional[str], optional
        The method used for belief propagation, by default 'minimum_sum'
    ms_scaling_factor : Optional[float], optional
        The scaling factor for the min-sum algorithm, by default 1.0
    schedule : Optional[str], optional
        The schedule for updating nodes, by default 'parallel'
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads to use, by default 1
    random_schedule_seed : Optional[int], optional
        The seed for random schedule, by default False
    serial_schedule_order : Optional[List[int]], optional
        The order of nodes for serial schedule, by default None
    osd_method : int, optional
        The method used for ordered statistic decoder, by default 0
    osd_order : int, optional
        The order for the ordered statistic decoder, by default 0
    flip_iterations : int, optional
        The number of iterations for the flipping decoder, by default 0
    pflip_frequency : int, optional
        The frequency of probabilistic flipping, by default 0
    pflip_seed : int, optional
        The seed for probabilistic flipping, by default 0
    dynamic_scaling_factor_damping : Optional[float], optional
        The damping factor for dynamic scaling in the minimum sum method, by default -1.0.
    """

    def __del__(self): ...

    def decode(self, syndrome: np.ndarray) -> np.ndarray: ...



   

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
