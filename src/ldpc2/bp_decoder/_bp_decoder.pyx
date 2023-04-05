#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix
from typing import Optional, List, Union

cdef class BpDecoderBase:

    def __init__(self,pcm, **kwargs):
        pass

    def __cinit__(self,pcm, **kwargs):

        error_rate=kwargs.get("error_rate",None)
        error_channel=kwargs.get("error_channel", None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        schedule=kwargs.get("schedule", 0)
        omp_thread_count = kwargs.get("omp_thread_count", 1)
        random_serial_schedule = kwargs.get("random_serial_schedule", 0)
        serial_schedule_order = kwargs.get("serial_schedule_order", None)
        
        
        """
        Docstring test
        """

        cdef int i, j, nonzero_count
        self.MEMORY_ALLOCATED=False

        #check the parity check matrix is the right type
        if isinstance(pcm, np.ndarray) or isinstance(pcm, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        # get the number of nonzero entries in the parity check matrix
        if isinstance(pcm,np.ndarray):
            nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
        elif isinstance(pcm,spmatrix):
            nonzero_count = int(pcm.nnz)

        # Matrix memory allocation
        self.pcm = make_shared[BpSparse](self.m,self.n,nonzero_count) #creates the C++ sparse matrix object

        #fill sparse matrix
        if isinstance(pcm,np.ndarray):
            for i in range(self.m):
                for j in range(self.n):
                    if pcm[i,j]==1:
                        self.pcm.get().insert_entry(i,j)
        elif isinstance(pcm,spmatrix):
            rows, cols = pcm.nonzero()
            for i in range(len(rows)):
                self.pcm.get().insert_entry(rows[i], cols[i])
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

        # allocate vectors for decoder input
        self._error_channel.resize(self.n) #C++ vector for the error channel
        self._syndrome.resize(self.m) #C++ vector for the syndrome
        self._serial_schedule_order = NULL_INT_VECTOR

        ## initialise the decoder with default values
        self.bpd = new BpDecoderCpp(self.pcm,self._error_channel,0,0,0.0,0,0,self._serial_schedule_order,0)

        ## set the decoder parameters
        self.bp_method = bp_method
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.serial_schedule_order = serial_schedule_order
        self.random_serial_schedule = random_serial_schedule
        self.omp_thread_count = omp_thread_count

        if error_channel is not None:
            self.error_channel = error_channel
        elif error_rate is not None:
            self.error_rate = error_rate
        else:
            raise ValueError("Please specify the error channel. Either: 1) error_rate: float or 2) error_channel:\
            list of floats of length equal to the block length of the code {self.n}.")

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd

    @property
    def error_rate(self) -> np.ndarray:
        """
        Returns the current error rate vector.

        Returns:
            np.ndarray: A numpy array containing the current error rate vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probs[i]
        return out

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        """
        Sets the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set. Must be a single float value.
        """
        if value is not None:
            if not isinstance(value, float):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
            for i in range(self.n):
                self.bpd.channel_probs[i] = value

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probs[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Optional[List[float]]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n):
                self.bpd.channel_probs[i] = value[i]


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
        return self.pcm.get().m

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """
        return self.pcm.get().n

    @property
    def max_iter(self) -> int:
        """
        Returns the maximum number of iterations allowed by the decoder.

        Returns:
            int: The maximum number of iterations allowed by the decoder.
        """
        return self.bpd.max_iter

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
        self.bpd.max_iter = value if value != 0 else self.n

    @property
    def bp_method(self) -> str:
        """
        Returns the belief propagation method used.

        Returns:
            str: The belief propagation method used. Possible values are 'product_sum' or 'minimum_sum'.
        """
        if self.bpd.bp_method == 0:
            return 'product_sum'
        elif self.bpd.bp_method == 1:
            return 'minimum_sum'
        else:
            return self.bpd.bp_method

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
            self.bpd.bp_method = 0
        elif str(value).lower() in ['min_sum', 'minimum_sum', 'ms', '1', 'minimum sum', 'min sum']:
            self.bpd.bp_method = 1
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
        if self.bpd.schedule == 0:
            return 'parallel'
        elif self.bpd.schedule == 1:
            return 'serial'
        else:
            return self.bpd.schedule

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
            self.bpd.schedule = 0
        elif str(value).lower() in ['serial','s','1']:
            self.bpd.schedule = 1
        else:
            raise ValueError(f"The BP schedule method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial'")

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
    def serial_schedule_order(self, value: Union[None, List[int]]) -> None:
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
        if not isinstance(value, float):
            raise TypeError("The ms_scaling factor must be specified as a float")
        self.bpd.ms_scaling_factor = value

    @property
    def omp_thread_count(self) -> int:
        """Get the number of OpenMP threads.

        Returns:
            int: The number of threads used.
        """
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
            raise TypeError("The omp_thread_count must be specified as a positive integer.")
        self.bpd.set_omp_thread_count(value)

    @property
    def random_serial_schedule(self) -> int:
        """Get the value of random_serial_schedule.

        Returns:
            int: The current value of random_serial_schedule.
        """
        return self.bpd.random_serial_schedule

    @random_serial_schedule.setter
    def random_serial_schedule(self, value: int) -> None:
        """Set the value of random_serial_schedule.

        Args:
            value (int): The new value of random_serial_schedule.

        Raises:
            ValueError: If the input value is not 0 or 1.
        """
        if not isinstance(value, int) or value < 0 or value > 1:
            raise ValueError("The value of random_serial_schedule must be either 0 or 1.")


        

# define the fused types
ctypedef fused SupportedTypes:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


cdef class BpDecoder(BpDecoderBase):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes.
    The decoder uses a sparse parity check matrix to decode received codewords. The decoding algorithm
    can be configured using various parameters, such as the belief propagation method used, the scheduling
    method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm: np.ndarray or scipy.sparse.spmatrix
        The parity check matrix for the code.
    error_rate: Optional[float]
        The probability of a bit being flipped in the received codeword.
    error_channel: Optional[List[float]]
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code.
    max_iter: Optional[int]
        The maximum number of iterations for the decoding algorithm.
    bp_method: Optional[str]
        The belief propagation method used. Must be one of {'product_sum', 'minimum_sum'}.
    ms_scaling_factor: Optional[float]
        The scaling factor used in the minimum sum method.
    schedule: Optional[str]
        The scheduling method used. Must be one of {'parallel', 'serial'}.
    omp_thread_count: Optional[int]
        The number of OpenMP threads used for parallel decoding.
    random_serial_schedule: Optional[int]
        Whether to use a random serial schedule order.
    serial_schedule_order: Optional[List[int]]
        A list of integers that specify the serial schedule order. Must be of length equal to the block length of the code.
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_serial_schedule: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None):
        pass

    def __init__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_serial_schedule: Optional[int] = False, serial_schedule_order: Optional[List[int]] = None):
        
        pass

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the input syndrome using belief propagation decoding algorithm.

        Parameters
        ----------
        syndrome : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input syndrome does not match the number of rows in the parity check matrix.
        """
        
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        cdef bool zero_syndrome = True
        DTYPE = syndrome.dtype
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]: zero_syndrome = False
        if zero_syndrome: return np.zeros(self.n,dtype=DTYPE)
        
        self.bpd.decode(self._syndrome)
        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out
        

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out


cdef class SoftInfoBpDecoder(BpDecoderBase):
    """
    A decoder that uses soft information belief propagation algorithm for decoding binary linear codes.

    This class implements a modified version of the belief propagation decoding algorithm that accounts for
    uncertainty in the syndrome readout using a serial belief propagation schedule. The decoder uses a minimum
    sum method as the belief propagation variant. For more information on the algorithm, please see the original
    research paper at https://arxiv.org/abs/2205.02341.

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity check matrix for the code.
    error_rate : Optional[float]
        The probability of a bit being flipped in the received codeword.
    error_channel : Optional[List[float]]
        A list of probabilities that specify the probability of each bit being flipped in the received codeword.
        Must be of length equal to the block length of the code.
    max_iter : Optional[int]
        The maximum number of iterations for the decoding algorithm.
    bp_method : Optional[str]
        The variant of belief propagation method to be used. The default value is 'minimum_sum'.
    ms_scaling_factor : Optional[float]
        The scaling factor used in the minimum sum method. The default value is 1.0.
    cutoff : Optional[float]
        The threshold value below which syndrome soft information is used.
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf):

        self.cutoff = cutoff
        self.schedule = "serial"
        self.bp_method = "minimum_sum"

        pass

    def __init__(self, pcm: Union[np.ndarray, spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[List[float]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[float] = 1.0, cutoff: Optional[float] = np.inf):
        
        pass

    def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the input syndrome using the soft information belief propagation decoding algorithm.

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
        
        self.bpd.soft_info_decode_serial(soft_syndrome,self.cutoff)

        out = np.zeros(self.n,dtype=np.uint8)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out
        
    @property
    def soft_syndrome(self) -> np.ndarray:
        """
        Returns the current soft syndrome.

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
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out






