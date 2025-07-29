import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]): ...



class BpDecoderBase:
    """
    Base class for Belief Propagation (BP) decoders.

    This class provides the foundational structure for BP decoders, including initialization, 
    memory management, and common properties such as error rates, channel probabilities, 
    and scheduling methods.

    Attributes:
        pcm (BpSparse): The parity check matrix in sparse format.
        m (int): Number of rows in the parity check matrix.
        n (int): Number of columns in the parity check matrix.
        MEMORY_ALLOCATED (bool): Indicates whether memory has been allocated for the decoder.
        bpd (BpDecoderCpp): The underlying C++ BP decoder object.
    """

    def __cinit__(self, pcm, **kwargs):
        """
        Initialize the BP decoder base class.

        Args:
            pcm (Union[np.ndarray, scipy.sparse.spmatrix]): The parity check matrix.
            **kwargs: Additional parameters for configuring the decoder.

        Keyword Args:
            error_rate (Optional[float]): Initial error rate for the decoder.
            error_channel (Optional[List[float]]): Initial error channel probabilities.
            max_iter (int): Maximum number of iterations for decoding.
            bp_method (int): Belief propagation method (0 for product-sum, 1 for minimum-sum).
            ms_scaling_factor (float): Scaling factor for the minimum-sum method.
            schedule (int): Scheduling method (0 for serial, 1 for parallel, 2 for serial-relative).
            omp_thread_count (int): Number of OpenMP threads to use.
            random_serial_schedule (bool): Whether to enable random serial scheduling.
            random_schedule_seed (int): Seed for random serial scheduling.
            serial_schedule_order (Optional[List[int]]): Custom order for serial scheduling.
            channel_probs (Optional[List[float]]): Channel probabilities for the decoder.
            dynamic_scaling_factor_damping (float): Damping factor for dynamic scaling in the minimum-sum method.

        Raises:
            TypeError: If the input matrix is not a valid type.
            ValueError: If required parameters are missing or invalid.
        """

    def __del__(self): ...

    @property
    def error_rate(self) -> np.ndarray:
        """
        Get the current error rate vector.

        Returns:
            np.ndarray: A numpy array containing the current error rate vector.
        """

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        """
        Set the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set.

        Raises:
            ValueError: If the input value is not a valid float.
        """

    @property
    def error_channel(self) -> np.ndarray:
        """
        Get the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]], np.ndarray]) -> None:
        """
        Set the error channel for the decoder.

        Args:
            value (Union[Optional[List[float]], np.ndarray]): The error channel vector to be set.

        Raises:
            ValueError: If the input vector length does not match the block length of the code.
        """

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None: ...

    @property
    def channel_probs(self) -> np.ndarray: ...


    @property
    def input_vector_type(self)-> str:
        """
        Returns the current input vector type.

        Returns:
            str: The current input vector type.
        """


    @input_vector_type.setter
    def input_vector_type(self, input_type: str):
        """
        Sets the input vector type.

        Args:
            input_type (str): The input vector type to be set. Must be either 'syndrome' or 'received_vector'.
        """


    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """

    @property
    def max_iter(self) -> int:
        """
        Returns the maximum number of iterations allowed by the decoder.

        Returns:
            int: The maximum number of iterations allowed by the decoder.
        """

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        """
        Sets the maximum number of iterations allowed by the decoder.

        Args:
            value (int): The maximum number of iterations allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def bp_method(self) -> str:
        """
        Returns the belief propagation method used.

        Returns:
            str: The belief propagation method used. Possible values are 'product_sum' or 'minimum_sum'.
        """

    @bp_method.setter
    def bp_method(self, value: Union[str,int]) -> None:
        """
        Sets the belief propagation method used.

        Args:
            value (str): The belief propagation method to use. Possible values are 'product_sum' or 'minimum_sum'.

        Raises:
            ValueError: If value is not a valid option.
        """

    @property
    def schedule(self) -> str:
        """
        Returns the scheduling method used.

        Returns:
            str: The scheduling method used. Possible values are 'parallel' or 'serial'.
        """

    @schedule.setter
    def schedule(self, value: Union[str,int]) -> None:
        """
        Sets the scheduling method used.

        Args:
            value (str): The scheduling method to use. Possible values are 'parallel' or 'serial'.

        Raises:
            ValueError: If value is not a valid option.
        """

    @property
    def serial_schedule_order(self) -> Union[None, np.ndarray]:
        """
        Returns the serial schedule order.

        Returns:
            Union[None, np.ndarray]: The serial schedule order as a numpy array, or None if no schedule has been set.
        """

    @serial_schedule_order.setter
    def serial_schedule_order(self, value: Union[None, List[int], np.ndarray]) -> None:
        """
        Sets the serial schedule order.

        Args:
            value (Union[None, List[int]]): The serial schedule order to set. Must have length equal to the block
            length of the code `self.n`.

        Raises:
            Exception: If value does not have the correct length.
            ValueError: If value contains an invalid integer value.
        """

    @property
    def ms_scaling_factor(self) -> float:
        """Get the scaling factor for minimum sum method.

        Returns:
            float: The current scaling factor.
        """

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value: float) -> None:
        """Set the scaling factor for minimum sum method.

        Args:
            value (float): The new scaling factor.

        Raises:
            TypeError: If the input value is not a float.
        """

    @property
    def ms_scaling_factor_vector(self) -> np.ndarray:
        """
        Get the vector of scaling factors for the minimum-sum method.

        Returns:
            np.ndarray: The current vector of scaling factors.
        """

    @ms_scaling_factor_vector.setter
    def ms_scaling_factor_vector(self, value: Union[List[float], np.ndarray]) -> None:
        """
        Set the vector of scaling factors for the minimum-sum method.

        Args:
            value (Union[List[float], np.ndarray]): The new vector of scaling factors.

        Raises:
            ValueError: If the input vector length does not match the maximum iterations.
        """

    @property
    def omp_thread_count(self) -> int:
        """Get the number of OpenMP threads.

        Returns:
            int: The number of threads used.
        """

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        """Set the number of OpenMP threads.

        Args:
            value (int): The number of threads to use.

        Raises:
            TypeError: If the input value is not an integer or is less than 1.
        """

    @property
    def random_schedule_seed(self) -> int:
        """Get the value of random_schedule_seed.

        Returns:
            int: The current value of random_schedule_seed.
        """

    @random_schedule_seed.setter
    def random_schedule_seed(self, value: int) -> None:
        """Set the value of random_schedule_seed.

        Args:
            value (int): The new value of random_schedule_seed.

        Raises:
            ValueError: If the input value is not a postive integer.
        """

    @property
    def random_serial_schedule(self) -> bool:
        """
        Returns whether the random serial schedule is enabled.

        Returns:
            bool: True if random serial schedule is enabled, False otherwise.
        """

    @random_serial_schedule.setter
    def random_serial_schedule(self, value: bool) -> None:
        """
        Sets whether the random serial schedule is enabled.

        Args:
            value (int): True to enable random serial schedule, False to disable it.

        Raises:
            ValueError: If random serial schedule is enabled while a fixed serial schedule is set.
        """

    @property
    def dynamic_scaling_factor_damping(self) -> float:
        """
        Get the dynamic scaling factor damping value.

        Returns:
            float: The current dynamic scaling factor damping value.
        """

    @dynamic_scaling_factor_damping.setter
    def dynamic_scaling_factor_damping(self, value: float) -> None:
        """
        Set the dynamic scaling factor damping value.

        Args:
            value (float): The new dynamic scaling factor damping value.

        Raises:
            ValueError: If the input value is not a non-negative float.
        """

class BpDecoder(BpDecoderBase):
    """
    Belief Propagation (BP) decoder for binary linear codes.

    This class provides an implementation of BP decoding for binary linear codes. It supports 
    various configurations, including different BP methods, scheduling strategies, and scaling factors.

    Parameters:
        pcm (Union[np.ndarray, scipy.sparse.spmatrix]): The parity check matrix.
        error_rate (Optional[float]): Initial error rate for the decoder.
        error_channel (Optional[List[float]]): Initial error channel probabilities.
        max_iter (Optional[int]): Maximum number of iterations for decoding.
        bp_method (Optional[str]): Belief propagation method ('product_sum' or 'minimum_sum').
        ms_scaling_factor (Optional[float]): Scaling factor for the minimum-sum method.
        schedule (Optional[str]): Scheduling method ('parallel', 'serial', or 'serial_relative').
        omp_thread_count (Optional[int]): Number of OpenMP threads to use.
        random_schedule_seed (Optional[int]): Seed for random serial scheduling.
        serial_schedule_order (Optional[List[int]]): Custom order for serial scheduling.
        input_vector_type (str): Input vector type ('syndrome', 'received_vector', or 'auto').
        random_serial_schedule (bool): Whether to enable random serial scheduling.
        dynamic_scaling_factor_damping (Optional[float]): Damping factor for dynamic scaling in the minimum-sum method.
    """

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None,
                 input_vector_type: str = "auto", random_serial_schedule: bool = False, dynamic_scaling_factor_damping: Optional[float] = -1, **kwargs): ...

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input vector using the BP decoding algorithm.

        Parameters:
            input_vector (np.ndarray): A 1D numpy array representing the input vector.

        Returns:
            np.ndarray: A 1D numpy array representing the decoded output.

        Raises:
            ValueError: If the input vector length does not match the expected length.
        """
        

    @property
    def decoding(self) -> np.ndarray:
        """
        Get the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """


class SoftInfoBpDecoder(BpDecoderBase):
    """
    Soft Information Belief Propagation (BP) decoder for binary linear codes.

    This class implements a modified BP decoding algorithm that accounts for uncertainty in 
    the syndrome readout using a serial belief propagation schedule.

    Parameters:
        pcm (Union[np.ndarray, spmatrix]): The parity check matrix.
        error_rate (Optional[float]): Initial error rate for the decoder.
        error_channel (Optional[List[float]]): Initial error channel probabilities.
        max_iter (Optional[int]): Maximum number of iterations for decoding.
        bp_method (Optional[str]): Belief propagation method ('minimum_sum').
        ms_scaling_factor (Optional[float]): Scaling factor for the minimum-sum method.
        cutoff (Optional[float]): Threshold value below which syndrome soft information is used.
        sigma (float): Standard deviation of the noise.
    """

    def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the input syndrome using the soft information BP decoding algorithm.

        Parameters
        ----------
        soft_info_syndrome: np.ndarray
            A 1-dimensional numpy array containing the soft information of the syndrome.

        Returns
        -------
        np.ndarray
            A 1-dimensional numpy array containing the decoded output.
        """

    @property
    def soft_syndrome(self) -> np.ndarray:
        """
        Get the current soft syndrome.

        Returns:
            np.ndarray: A numpy array containing the current soft syndrome.
        """


    @property
    def decoding(self) -> np.ndarray:
        """
        Get the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
