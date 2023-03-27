#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class bp_decoder_base:

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
        
        '''
        Docstring test
        '''

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
        self.pcm = make_shared[bp_sparse](self.m,self.n,nonzero_count) #creates the C++ sparse matrix object

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
        self.bpd = new bp_decoder_cpp(self.pcm,self._error_channel,0,0,0.0,0,0,self._serial_schedule_order,0)

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

    # cpdef np.ndarray[np.uint8_t, ndim=1] decode(self,uint8_t[::1] input_syndrome):
    #     self._syndrome = &input_syndrome[0] #get pointer of memoryview
    #     self.bpd.decode(self._syndrome) 
    #     # cdef uint8_t[::1] output_view = <uint8_t[:self.n]> self.bpd.decoding #convert pointer array to memory view
    #     cdef np.ndarray[np.uint8_t, ndim=1] out = np.asarray(<uint8_t[:self.n]> self.bpd.decoding) #convert memory view to numpy
    #     return out #return numpy array

    @property
    def decoding(self):
        out = np.zeros(self.n).astype(int)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out

 

    @property
    def error_rate(self):
        out = np.zeros(self.n).astype(float)
        for i in range(self.n): out[i] = self.bpd.channel_probs[i]
        return out

    @error_rate.setter
    def error_rate(self, value):
        if value is not None:
            if not isinstance(value,float):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
            for i in range(self.n): self.bpd.channel_probs[i] = value


    @property
    def error_channel(self):
        out = np.zeros(self.n).astype(float)
        for i in range(self.n): out[i] = self.bpd.channel_probs[i]
        return out

    @error_channel.setter
    def error_channel(self, value):
        if value is not None:
            if len(value)!=self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n): self.bpd.channel_probs[i] = value[i]


    @property
    def log_prob_ratios(self):
        out=np.zeros(self.n)
        for i in range(self.n): out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def converge(self):
        return self.bpd.converge

    @property
    def iter(self):
        return self.bpd.iterations

    @property
    def check_count(self):
        """Returns the number of rows of the parity check matrix"""
        return self.pcm.get().m

    @property
    def bit_count(self):
        """Returns the number of columns of the parity check matrix"""
        return self.pcm.get().n

    @property
    def max_iter(self):
        """Returns the maximum number of iterations"""
        return self.bpd.max_iter

    @max_iter.setter
    def max_iter(self, value):
        """Sets the maximum number of iterations"""
        if not isinstance(value,int):
            raise ValueError("max_iter input parameter is invalid. This must be specified as a positive int.")
        if value<0:
            raise ValueError(f"max_iter input parameter must be a postive int. Not {value}.")
        self.bpd.max_iter = value if value != 0 else self.n

    @property
    def bp_method(self):
        """Returns the belief propagation method used"""
        if self.bpd.bp_method == 0:
            return 'product_sum'
        elif self.bpd.bp_method == 1:
            return 'minimum_sum'
        else:
            return self.bpd.bp_method

    @bp_method.setter
    def bp_method(self, value):
        """Sets the belief propagation method used"""
        if str(value).lower() in ['prod_sum','product_sum','ps','0','prod sum']:
            self.bpd.bp_method = 0
        elif str(value).lower() in ['min_sum','minimum_sum','ms','1','minimum sum','min sum']:
            self.bpd.bp_method = 1
        else:
            raise ValueError(f"BP method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'product_sum', 'minimum_sum'")

    @property
    def schedule(self):
        """Returns the scheduling method used"""
        if self.bpd.schedule == 0:
            return 'parallel'
        elif self.bpd.schedule == 1:
            return 'serial'
        else:
            return self.bpd.schedule

    @schedule.setter
    def schedule(self, value):
        """Sets the scheduling method used"""
        if str(value).lower() in ['parallel','p','0']:
            self.bpd.schedule = 0
        elif str(value).lower() in ['serial','s','1']:
            self.bpd.schedule = 1
        else:
            raise ValueError(f"The BP schedule method '{value}' is invalid. \
                    Please choose from the following methods: \
                    'schedule=parallel', 'schedule=serial'")

    @property
    def serial_schedule_order(self):
        """Returns the serial schedule order"""
        if self.bpd.serial_schedule_order.size() == 0:
            return None

        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.serial_schedule_order[i]
        return out

    @serial_schedule_order.setter
    def serial_schedule_order(self, value):
        """Sets the serial schedule order"""
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
    def ms_scaling_factor(self):
        """Returns the ms_scaling_factor used"""
        return self.bpd.ms_scaling_factor

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value):
        """Sets the ms_scaling_factor used"""
        if not isinstance(value, float):
            raise TypeError("The ms_scaling factor must be specified as a float")
        self.bpd.ms_scaling_factor = value

    @property
    def omp_thread_count(self):
        """Returns the omp_thread_count used"""
        return self.bpd.omp_thread_count

    @omp_thread_count.setter
    def omp_thread_count(self, value):
        """Sets the omp_thread_count used"""
        if not isinstance(value, int) or value < 1:
            raise TypeError("The omp_thread_count must be specified as a positive integer.")
        self.bpd.set_omp_thread_count(value)

    @property
    def random_serial_schedule(self):
        """Returns the value of random_serial_schedule"""
        return self.bpd.random_serial_schedule

    @random_serial_schedule.setter
    def random_serial_schedule(self, value):
        """Sets the value of random_serial_schedule"""
        if not isinstance(value, int) or value < 0 or value > 1:
            raise ValueError("The value of random_serial_schedule must be either 0 or 1.")
        self.bpd.random_serial_schedule = value
        

cdef class bp_decoder(bp_decoder_base):

    def __cinit__(self,pcm, error_rate=None, error_channel=None, max_iter=0, bp_method=1, ms_scaling_factor=1.0, schedule=0, omp_thread_count = 1, random_serial_schedule = 0, serial_schedule_order = None):
        pass

    def __init__(self,pcm, error_rate=None, error_channel=None, max_iter=0, bp_method=1, ms_scaling_factor=1.0, schedule=0, omp_thread_count = 1, random_serial_schedule = 0, serial_schedule_order = None):
        
        pass

    def decode(self,syndrome):
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
    def decoding(self):
        out = np.zeros(self.n).astype(int)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out


    # cpdef np.ndarray[np.uint8_t, ndim=1] decode(self,uint8_t[::1] input_syndrome):
    #     self._syndrome = &input_syndrome[0] #get pointer of memoryview
    #     self.bpd.decode(self._syndrome) 
    #     # cdef uint8_t[::1] output_view = <uint8_t[:self.n]> self.bpd.decoding #convert pointer array to memory view
    #     cdef np.ndarray[np.uint8_t, ndim=1] out = np.asarray(<uint8_t[:self.n]> self.bpd.decoding) #convert memory view to numpy
    #     return out #return numpy array





