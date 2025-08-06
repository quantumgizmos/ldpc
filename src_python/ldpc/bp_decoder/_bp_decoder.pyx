#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers

cdef BpSparse* Py2BpSparse(pcm):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input\
        a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse.spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef BpSparse* cpcm = new BpSparse(m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse.spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.insert_entry(rows[i], cols[i])
    
    return cpcm

cdef coords_to_scipy_sparse(vector[vector[int]]& entries, int m, int n, int entry_count):

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef BpSparse2Py(BpSparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    cdef BpSparse* cpcm = Py2BpSparse(pcm)
    output = BpSparse2Py(cpcm)
    del cpcm
    return output



cdef class BpDecoderBase:
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
        error_rate=kwargs.get("error_rate",None)
        error_channel=kwargs.get("error_channel", None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        schedule=kwargs.get("schedule", 0)
        omp_thread_count = kwargs.get("omp_thread_count", 1)
        random_serial_schedule = kwargs.get("random_serial_schedule", False)
        random_schedule_seed = kwargs.get("random_schedule_seed", 0)
        serial_schedule_order = kwargs.get("serial_schedule_order", None)
        channel_probs = kwargs.get("channel_probs", [None])
        dynamic_scaling_factor_damping = kwargs.get("dynamic_scaling_factor_damping", -1.0)
        ms_converge = kwargs.get("ms_converge_value", 1.0)
        
        # input_vector_type = kwargs.get("input_vector_type", "auto")
        # print(kwargs.get("input_vector_type"))
        # print("input vector type:", input_vector_type)
        
        """
        Docstring test
        """

        cdef int i, j, nonzero_count
        self.MEMORY_ALLOCATED=False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)
 
        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        # allocate vectors for decoder input
        self._error_channel.resize(self.n) #C++ vector for the error channel
        self._syndrome.resize(self.m) #C++ vector for the syndrome
        self._serial_schedule_order = NULL_INT_VECTOR



        ## initialise the decoder with default values
        self.bpd = new BpDecoderCpp(
            self.pcm[0],
            self._error_channel,
            0,
            PRODUCT_SUM,
            PARALLEL,
            1.0,
            1,
            self._serial_schedule_order,
            0,
            False,
            SYNDROME,
            dynamic_scaling_factor_damping
        )

        ## set the decoder parameters
        self.bp_method = bp_method
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.serial_schedule_order = serial_schedule_order
        self.random_schedule_seed = random_schedule_seed
        self.omp_thread_count = omp_thread_count
        self.random_serial_schedule = random_serial_schedule

        if dynamic_scaling_factor_damping >= 0:
            self.dynamic_scaling_factor_damping = dynamic_scaling_factor_damping
            self.ms_converge_value = ms_converge

        ## the ldpc_v1 backwards compatibility
        if isinstance(channel_probs, list) or isinstance(channel_probs, np.ndarray):
            if(len(channel_probs)>0) and (channel_probs[0] is not None):
                error_channel = channel_probs

        if error_channel is not None:
            self.error_channel = error_channel
        elif error_rate is not None:
            self.error_rate = error_rate
        else:
            raise ValueError("Please specify the error channel. Either: 1) error_rate: float or 2) error_channel:\
            list of floats of length equal to the block length of the code {self.n}.")        

        self.bpd.set_up_ms_scaling_factors()

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd
            del self.pcm

    @property
    def error_rate(self) -> np.ndarray:
        """
        Get the current error rate vector.

        Returns:
            np.ndarray: A numpy array containing the current error rate vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        """
        Set the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set.

        Raises:
            ValueError: If the input value is not a valid float.
        """
        if value is not None:
            if not isinstance(value, float):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
            for i in range(self.n):
                self.bpd.channel_probabilities[i] = value

    @property
    def error_channel(self) -> np.ndarray:
        """
        Get the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]], np.ndarray]) -> None:
        """
        Set the error channel for the decoder.

        Args:
            value (Union[Optional[List[float]], np.ndarray]): The error channel vector to be set.

        Raises:
            ValueError: If the input vector length does not match the block length of the code.
        """
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n):
                self.bpd.channel_probabilities[i] = value[i]

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None:
        self.error_channel = value

    @property
    def channel_probs(self) -> np.ndarray:
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out


    @property
    def input_vector_type(self)-> str:
        """
        Returns the current input vector type.

        Returns:
            str: The current input vector type.
        """
        if self.bpd.bp_input_type == SYNDROME:
            return 'syndrome'
        elif self.bpd.bp_input_type == RECEIVED_VECTOR:
            return 'received_vector'
        elif self.bpd.bp_input_type == AUTO:
            return 'auto'
        else:
            raise ValueError(f"The input vector type is invalid. \
                    Please choose from the following methods: \
                    'input_vector_type=syndrome', 'input_vector_type=received_vector'")


    @input_vector_type.setter
    def input_vector_type(self, input_type: str):
        """
        Sets the input vector type.

        Args:
            input_type (str): The input vector type to be set. Must be either 'syndrome' or 'received_vector'.
        """
        if input_type.lower() in ['auto', 'a', '2']:
            if self.m == self.n:
                raise ValueError("Please specify the input vector type. Either: 1) input_vector_type: 'syndrome' or 2) input_vector_type:\
                'received_vector'.")
            else:
                self.bpd.bp_input_type = AUTO

        elif input_type.lower() in ['syndrome', 's', '0']:
            self.bpd.bp_input_type = SYNDROME
        elif input_type.lower() in ['received_vector', 'r', '1']:
            self.bpd.bp_input_type = RECEIVED_VECTOR
        else:
            raise ValueError(f"The input vector type '{input_type}' is invalid. \
                    Please choose from the following methods: \
                    'input_vector_type=syndrome', 'input_vector_type=received_vector'")


    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """
        return self.bpd.converge

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """
        return self.bpd.iterations


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """
        return self.bpd.pcm.m

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """
        return self.bpd.pcm.n

    @property
    def max_iter(self) -> int:
        """
        Returns the maximum number of iterations allowed by the decoder.

        Returns:
            int: The maximum number of iterations allowed by the decoder.
        """
        return self.bpd.maximum_iterations

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        """
        Sets the maximum number of iterations allowed by the decoder.

        Args:
            value (int): The maximum number of iterations allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("max_iter input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"max_iter input parameter must be a positive int. Not {value}.")
        self.bpd.maximum_iterations = value if value != 0 else self.n

    @property
    def bp_method(self) -> str:
        """
        Returns the belief propagation method used.

        Returns:
            str: The belief propagation method used. Possible values are 'product_sum' or 'minimum_sum'.
        """
        if self.bpd.bp_method == PRODUCT_SUM:
            return 'product_sum'
        elif self.bpd.bp_method == MINIMUM_SUM:
            return 'minimum_sum'
        else:
            raise ValueError(f"BP method is invalid. \
                    Please choose from the following methods: \
                    'product_sum', 'minimum_sum'")

    @bp_method.setter
    def bp_method(self, value: Union[str,int]) -> None:
        """
        Sets the belief propagation method used.

        Args:
            value (str): The belief propagation method to use. Possible values are 'product_sum' or 'minimum_sum'.

        Raises:
            ValueError: If value is not a valid option.
        """
        if str(value).lower() in ['prod_sum', 'product_sum', 'ps', '0', 'prod sum']:
            self.bpd.bp_method = PRODUCT_SUM
        elif str(value).lower() in ['min_sum', 'minimum_sum', 'ms', '1', 'minimum sum', 'min sum']:
            self.bpd.bp_method = MINIMUM_SUM
        else:
            raise ValueError(f"BP method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'product_sum', 'minimum_sum'")

    @property
    def schedule(self) -> str:
        """
        Returns the scheduling method used.

        Returns:
            str: The scheduling method used. Possible values are 'parallel' or 'serial'.
        """
        if self.bpd.schedule == PARALLEL:
            return 'parallel'
        elif self.bpd.schedule == SERIAL:
            return 'serial'
        elif self.bpd.schedule == SERIAL_RELATIVE:
            return 'serial_relative'
        else:
            raise ValueError(f"The BP schedule method is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial', 'schedule=serial_relative'")

    @schedule.setter
    def schedule(self, value: Union[str,int]) -> None:
        """
        Sets the scheduling method used.

        Args:
            value (str): The scheduling method to use. Possible values are 'parallel' or 'serial'.

        Raises:
            ValueError: If value is not a valid option.
        """
        if str(value).lower() in ['parallel','p','0']:
            self.bpd.schedule = PARALLEL
        elif str(value).lower() in ['serial','s','1']:
            self.bpd.schedule = SERIAL
        elif str(value).lower() in ['serial_relative', 'sr', '2']:
            self.bpd.schedule = SERIAL_RELATIVE
        else:
            raise ValueError(f"The BP schedule method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial', 'schedule=serial_relative'")

    @property
    def serial_schedule_order(self) -> Union[None, np.ndarray]:
        """
        Returns the serial schedule order.

        Returns:
            Union[None, np.ndarray]: The serial schedule order as a numpy array, or None if no schedule has been set.
        """
        if self.bpd.serial_schedule_order.size() == 0:
            return None

        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.serial_schedule_order[i]
        return out

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
        if value is None:
            self._serial_schedule_order = NULL_INT_VECTOR
            return
        if not len(value) == self.n:
            raise Exception("Input error. The `serial_schedule_order` input parameter must have length equal to the length of the code.")
        for i in range(self.n):
            if not isinstance(value[i], (int, np.int64, np.int32)) or value[i] < 0 or value[i] >= self.n:
                print(type(value[i]),"Value:", value[i], "i:", i, "n:", self.n)
                raise ValueError(f"serial_schedule_order[{i}] is invalid. It must be a non-negative integer less than {self.n}.")
            self.bpd.serial_schedule_order[i] = value[i]
        self.random_serial_schedule = False

    @property
    def ms_scaling_factor(self) -> float:
        """Get the scaling factor for minimum sum method.

        Returns:
            float: The current scaling factor.
        """
        return self.bpd.ms_scaling_factor

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value: float) -> None:
        """Set the scaling factor for minimum sum method.

        Args:
            value (float): The new scaling factor.

        Raises:
            TypeError: If the input value is not a float.
        """
        if not isinstance(value, (float,int)):
            raise TypeError("The ms_scaling factor must be specified as a float")
        self.bpd.ms_scaling_factor = value

    @property
    def ms_scaling_factor_vector(self) -> np.ndarray:
        """
        Get the vector of scaling factors for the minimum-sum method.

        Returns:
            np.ndarray: The current vector of scaling factors.
        """
        out = np.zeros(len(self.bpd.ms_scaling_factor_vector), dtype=np.float64)
        for i in range(len(self.bpd.ms_scaling_factor_vector)):
            out[i] = self.bpd.ms_scaling_factor_vector[i]
        return out

    @ms_scaling_factor_vector.setter
    def ms_scaling_factor_vector(self, value: Union[List[float], np.ndarray]) -> None:
        """
        Set the vector of scaling factors for the minimum-sum method.

        Args:
            value (Union[List[float], np.ndarray]): The new vector of scaling factors.

        Raises:
            ValueError: If the input vector length does not match the maximum iterations.
        """
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError("The ms_scaling_factor_vector must be specified as a list or numpy array of floats.")
        if len(value) != self.bpd.maximum_iterations:
            raise ValueError(f"The ms_scaling_factor_vector must have length {self.bpd.maximum_iterations}.")
        self.bpd.ms_scaling_factor_vector.clear()
        for v in value:
            self.bpd.ms_scaling_factor_vector.push_back(v)

    @property
    def omp_thread_count(self) -> int:
        """Get the number of OpenMP threads.

        Returns:
            int: The number of threads used.
        """
        if self.bpd.omp_thread_count != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")
        return self.bpd.omp_thread_count

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        """Set the number of OpenMP threads.

        Args:
            value (int): The number of threads to use.

        Raises:
            TypeError: If the input value is not an integer or is less than 1.
        """
        if not isinstance(value, int) or value < 1:
            raise TypeError("The omp_thread_count must be specified as a\
            positive integer.")
        self.bpd.set_omp_thread_count(value)
        if self.bpd.omp_thread_count != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")

    @property
    def random_schedule_seed(self) -> int:
        """Get the value of random_schedule_seed.

        Returns:
            int: The current value of random_schedule_seed.
        """
        return self.bpd.random_schedule_seed

    @random_schedule_seed.setter
    def random_schedule_seed(self, value: int) -> None:
        """Set the value of random_schedule_seed.

        Args:
            value (int): The new value of random_schedule_seed.

        Raises:
            ValueError: If the input value is not a postive integer.
        """
        if not isinstance(value, int) or value < -2:
            raise ValueError("The value of random_schedule_seed must\
            be a positive integer. Set as -1 to disable to the random\
            schedule. Set as 0 to use the system clock.")

        self.bpd.random_serial_schedule = True
        self.bpd.set_random_schedule_seed(value)

    @property
    def random_serial_schedule(self) -> bool:
        """
        Returns whether the random serial schedule is enabled.

        Returns:
            bool: True if random serial schedule is enabled, False otherwise.
        """
        return self.bpd.random_serial_schedule

    @random_serial_schedule.setter
    def random_serial_schedule(self, value: bool) -> None:
        """
        Sets whether the random serial schedule is enabled.

        Args:
            value (int): True to enable random serial schedule, False to disable it.

        Raises:
            ValueError: If random serial schedule is enabled while a fixed serial schedule is set.
        """
        # if not isinstance(value, bool):
        #     raise ValueError("The random_serial_schedule must be a boolean value.")
        self.bpd.random_serial_schedule = value

    @property
    def dynamic_scaling_factor_damping(self) -> float:
        """
        Get the dynamic scaling factor damping value.

        Returns:
            float: The current dynamic scaling factor damping value.
        """
        return self.bpd.dynamic_scaling_factor_damping

    @dynamic_scaling_factor_damping.setter
    def dynamic_scaling_factor_damping(self, value: float) -> None:
        """
        Set the dynamic scaling factor damping value.

        Args:
            value (float): The new dynamic scaling factor damping value.

        Raises:
            ValueError: If the input value is not a non-negative float.
        """
        if self.bpd.bp_method != MINIMUM_SUM:
            raise ValueError(f"The dynamic_scaling_factor_damping can only be set for the minimum-sum method. The current method is {self.bp_method}.")
        if not isinstance(value, (float, int)) or value < 0:
            raise ValueError("The dynamic_scaling_factor_damping must be a non-negative float.")
        self.bpd.dynamic_scaling_factor_damping = value
        self.bpd.set_up_ms_scaling_factors()

    @property
    def ms_converge_value(self) -> float:
        """
        Get the ms_converge_value for the minimum-sum method.

        Returns:
            float: The current ms_converge_value.
        """
        return self.bpd.ms_converge_value

    @ms_converge_value.setter
    def ms_converge_value(self, value: float) -> None:
        """
        Set the ms_converge_value for the minimum-sum method.

        Args:
            value (float): The new ms_converge_value.

        Raises:
            ValueError: If the input value is not a float.
        """
        if self.bpd.bp_method != MINIMUM_SUM:
            raise ValueError(f"The ms_converge_value can only be set for the minimum-sum method. The current method is {self.bp_method}.")
        if not isinstance(value, (float, int)):
            raise ValueError("The ms_converge_value must be a float.")
        self.bpd.ms_converge_value = value
        self.bpd.set_up_ms_scaling_factors()

cdef class BpDecoder(BpDecoderBase):
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
        ms_converge_value (Optional[float]): Convergence value for the minimum-sum method.
    """
    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, input_vector_type: str = "auto", random_serial_schedule: bool = False, dynamic_scaling_factor_damping: Optional[float] = -1, ms_converge_value=1.0, **kwargs):

        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")

        self.input_vector_type = input_vector_type
        self._received_vector.resize(self.n) #C++ vector for the received vector

        pass

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None,
                 input_vector_type: str = "auto", random_serial_schedule: bool = False, dynamic_scaling_factor_damping: Optional[float] = -1, ms_converge_value=1.0, **kwargs):
        
        pass

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
        
        if(self.bpd.bp_input_type == SYNDROME and not len(input_vector)==self.m):
            raise ValueError(f"The input_vector must have length {self.m} (for syndrome decoding). Not length {len(input_vector)}.")
        elif(self.bpd.bp_input_type == RECEIVED_VECTOR and not len(input_vector)==self.n):
            raise ValueError(f"The input_vector must have length {self.n} (for received vector decoding). Not length {len(input_vector)}.")
        elif(self.bpd.bp_input_type == AUTO and not (len(input_vector)==self.m or len(input_vector)==self.n)):
            raise ValueError(f"The input_vector must have length {self.m} (for syndrome decoding) or length {self.n} (for received vector decoding). Not length {len(input_vector)}.")

        cdef int i
        cdef bool zero_input_vector = True
        DTYPE = input_vector.dtype

        cdef int len_input_vector = len(input_vector)
        
        if(self.bpd.bp_input_type == SYNDROME or (self.bpd.bp_input_type == AUTO and len(input_vector)==self.m)):
            for i in range(len_input_vector):
                self._syndrome[i] = input_vector[i]
                if self._syndrome[i]: zero_input_vector = False
            if zero_input_vector:
                self.bpd.converge = True
                return np.zeros(self.bit_count,dtype=DTYPE)
            self.bpd.decode(self._syndrome)

        elif(self.bpd.bp_input_type == RECEIVED_VECTOR or (self.bpd.bp_input_type == AUTO and len(input_vector)==self.n)):
            for i in range(len_input_vector):
                self._received_vector[i] = input_vector[i]
                if self._received_vector[i]: zero_input_vector = False
            if zero_input_vector:
                self.bpd.converge = True
                return np.zeros(self.bit_count,dtype=DTYPE)
            self.bpd.decode(self._received_vector)
        
        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out
        

    @property
    def decoding(self) -> np.ndarray:
        """
        Get the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out


cdef class SoftInfoBpDecoder(BpDecoderBase):
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
    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf, sigma: float = 2.0, **kwargs):

        self.cutoff = cutoff
        if not isinstance(sigma,float) or sigma <= 0:
            raise ValueError("The sigma value must be a float greater than 0.")
        self.sigma = sigma
        self.schedule = "serial"
        self.bp_method = "minimum_sum"
        self.input_vector_type = "syndrome"

    # def __init__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
    #              error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
    #              ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf, sigma: float = 2.0, input_vector_type: str = "syndrome"):

    #     pass

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

        cdef vector[np.float64_t] soft_syndrome
        soft_syndrome.resize(self.m)
        for i in range(self.m):
            soft_syndrome[i] = soft_info_syndrome[i]

        self.bpd.soft_info_decode_serial(soft_syndrome,self.cutoff, self.sigma)

        out = np.zeros(self.n,dtype=np.uint8)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out

    @property
    def soft_syndrome(self) -> np.ndarray:
        """
        Get the current soft syndrome.

        Returns:
            np.ndarray: A numpy array containing the current soft syndrome.
        """
        out = np.zeros(self.m)
        for i in range(self.m):
            out[i] = self.bpd.soft_syndrome[i]
        return out


    @property
    def decoding(self) -> np.ndarray:
        """
        Get the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out






