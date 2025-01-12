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
    Bp Decoder base class
    """

    def __cinit__(self,pcm, **kwargs):

        error_rate=kwargs.get("error_rate",None)
        error_channel=kwargs.get("error_channel", None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        schedule=kwargs.get("schedule", 0)
        omp_thread_count = kwargs.get("omp_thread_count", 1)
        random_schedule_seed = kwargs.get("random_schedule_seed", 0)
        serial_schedule_order = kwargs.get("serial_schedule_order", None)
        channel_probs = kwargs.get("channel_probs", [None])
        
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
        self.bpd = new BpDecoderCpp(self.pcm[0],self._error_channel,0,PRODUCT_SUM,PARALLEL,1.0,1,self._serial_schedule_order,0,True,SYNDROME)

        ## set the decoder parameters
        self.bp_method = bp_method
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.serial_schedule_order = serial_schedule_order
        self.random_schedule_seed = random_schedule_seed
        self.omp_thread_count = omp_thread_count

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


        

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd
            del self.pcm

    @property
    def error_rate(self) -> np.ndarray:
        """
        Returns the current error rate vector.

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
        Sets the error rate for the decoder.

        Args:
            value (Optional[float]): The error rate value to be set. Must be a single float value.
        """
        if value is not None:
            if not isinstance(value, float):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
            for i in range(self.n):
                self.bpd.channel_probabilities[i] = value

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
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

        self.bpd.random_schedule_seed = value

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
        The scheduling method for belief propagation: 'parallel', 'serial', or 'serial_relative'. By default 'parallel'.
    omp_thread_count : Optional[int], optional
        The number of OpenMP threads to use, by default 1.
    random_schedule_seed : Optional[int], optional
        The seed for the random serial schedule, by default 0. If set to 0, the seed is set according the system clock.
    serial_schedule_order : Optional[List[int]], optional
        The custom order for serial scheduling, by default None.
    input_vector_type: str, optional
        Use this paramter to specify the input type. Choose either: 1) 'syndrome' or 2) 'received_vector' or 3) 'auto'.
        Note, it is only necessary to specify this value when the parity check matrix is square. When the
        parity matrix is non-square the input vector type is inferred automatically from its length.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], error_rate: Optional[float] = None,
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_iter: Optional[int] = 0, bp_method: Optional[str] = 'minimum_sum',
                 ms_scaling_factor: Optional[Union[float,int]] = 1.0, schedule: Optional[str] = 'parallel', omp_thread_count: Optional[int] = 1,
                 random_schedule_seed: Optional[int] = 0, serial_schedule_order: Optional[List[int]] = None, input_vector_type: str = "auto", **kwargs):

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
                 input_vector_type: str = "auto", **kwargs):
        
        pass

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input input_vector using belief propagation decoding algorithm.

        Parameters
        ----------
        input_vector : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input input_vector does not match the number of rows in the parity check matrix.
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

        self.bpd.soft_info_decode_serial(soft_syndrome,self.cutoff, self.sigma)

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






