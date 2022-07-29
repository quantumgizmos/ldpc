#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
from scipy.sparse import spmatrix

cdef class bp_decoder:
    '''
    A class implementing a belief propagation decoder for LDPC codes

    Parameters
    ----------
    parity_check_matrix: numpy.ndarray or spipy.sparse
        The parity check matrix of the binary code in numpy.ndarray or spipy.sparse format.
    error_rate: float64, optional
        The bit error rate.
    max_iter: int, optional
        The maximum number of iterations for the BP decoder. If max_iter==0, the BP algorithm
        will iterate n times, where n is the block length of the code.
    bp_method: str or int, optional
        The BP method. Currently three methods are implemented: 1) "ps": product sum updates;
        2) "ms": min-sum updates; 3) "msl": min-sum log updates
    ms_scaling_factor: float64, optional
        Sets the min-sum scaling factor for the min-sum BP method
    channel_probs: list, optional
        This parameter can be used to set the initial error channel across all bits.
    input_vector_type: str, optional
        Use this paramter to specify the input type. Choose either: 1) 'syndrome' or 2) 'received_vector'.
        Note, it is only necessary to specify this value when the parity check matrix is square. When the
        parity matrix is non-square the input vector type is inferred automatically from its length.
    '''

    def __cinit__(self,parity_check_matrix,**kwargs):

        #Load in optional parameters (and set defaults)
        error_rate=kwargs.get("error_rate",None)
        max_iter=kwargs.get("max_iter",0)
        bp_method=kwargs.get("bp_method",0)
        ms_scaling_factor=kwargs.get("ms_scaling_factor",1.0)
        channel_probs=kwargs.get("channel_probs",[None])
        input_vector_type=kwargs.get("input_vector_type",-1)


        self.MEM_ALLOCATED=False
        cdef i,j

        #check that parity_check_matrix is a numpy.ndarray or scipy.sparse.spmatrix object

        if isinstance(parity_check_matrix, np.ndarray) or isinstance(parity_check_matrix, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(parity_check_matrix)}")

        self.m=parity_check_matrix.shape[0]
        self.n=parity_check_matrix.shape[1]

        #input vector type
        if type(input_vector_type) is int:
            self.input_vector_type =  input_vector_type
        elif type(input_vector_type) is str and input_vector_type == "syndrome":
            self.input_vector_type = 0
        elif type(input_vector_type) is str and input_vector_type == "received_vector":
            self.input_vector_type = 1
        else:
            raise Exception(f"TypeError: input_vector type must be either 'syndrome' or 'received_vector'. Not {input_vector_type}")

        #Error rate
        if error_rate!=None:
            if error_rate<0 or error_rate>1.0:
                raise ValueError(f"The error rate must be in the range 0.0<error_rate<1.0")

        #BP iterations
        if max_iter<0: raise ValueError('The maximum number of iterations must a positive number')
        if max_iter==0: max_iter=self.n

        #BP method
        if str(bp_method).lower() in ['prod_sum','product_sum','ps','0','prod sum']:
            bp_method=0
        elif str(bp_method).lower() in ['min_sum','minimum_sum','ms','1','minimum sum','min sum']:
            bp_method=3 # method 1 is not working (see issue 1). Defaulting to the log version of bp.
        elif str(bp_method).lower() in ['prod_sum_log','product_sum_log','ps_log','2','psl']:
            bp_method=2
        elif str(bp_method).lower() in ['min_sum_log','minimum_sum_log','ms_log','3','minimum sum_log','msl']:
            bp_method=3
        else: raise ValueError(f"BP method '{bp_method}' is invalid.\
                            Please choose from the following methods:'product_sum',\
                            'minimum_sum', 'product_sum_log' or 'minimum_sum_log'")

        if channel_probs[0]!=None:
            if len(channel_probs)!=self.n:
                raise ValueError(f"The length of the channel probability vector must be eqaul to the block length n={self.n}.")

        #BP Settings
        self.max_iter=max_iter
        self.bp_method=bp_method
        self.ms_scaling_factor=ms_scaling_factor

        #memory allocation
        if isinstance(parity_check_matrix, np.ndarray):
            self.H=numpy2mod2sparse(parity_check_matrix) #parity check matrix in sparse form
        elif isinstance(parity_check_matrix, spmatrix):
            self.H=spmatrix2mod2sparse(parity_check_matrix)

        assert self.n==self.H.n_cols #validate number of bits in mod2sparse format
        assert self.m==self.H.n_rows #validate number of checks in mod2sparse format
        self.error=<char*>calloc(self.n,sizeof(char)) #error string
        self.synd=<char*>calloc(self.m,sizeof(char)) #syndrome string
        self.received_codeword=<char*>calloc(self.n,sizeof(char)) #received codeword
        self.bp_decoding_synd=<char*>calloc(self.m,sizeof(char)) #decoded syndrome string
        self.bp_decoding=<char*>calloc(self.n,sizeof(char)) #BP decoding
        self.channel_probs=<double*>calloc(self.n,sizeof(double)) #channel probs
        self.log_prob_ratios=<double*>calloc(self.n,sizeof(double)) #log probability ratios

        self.MEM_ALLOCATED=True

        #error channel setup
        if channel_probs[0]!=None:
            for j in range(self.n): self.channel_probs[j]=channel_probs[j]
            self.error_rate=np.mean(channel_probs)
        elif error_rate!=None:
            for j in range(self.n): self.channel_probs[j]=error_rate
            self.error_rate=error_rate

    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector):

        """
        Runs the BP decoder for a given input_vector.

        Parameters
        ----------

        input_vector: numpy.ndarray or scipy.sparse.spmatrix
            The syndrome to be decoded.

        Returns
        -------
        numpy.ndarray
            The belief propagation decoding in numpy.ndarray format.
        """

        cdef int input_length = input_vector.shape[0]
        cdef int i

        if self.n == self.m and self.input_vector_type == -1:
            raise Exception("TypeError: the block length is equal to the syndrome length. Please set the 'input_vector_type' parameter as either: 1) 'syndrome' or 2) 'received_vector'")

        if input_length == self.n and (self.input_vector_type == 1 or self.input_vector_type == -1):
            if isinstance(input_vector,spmatrix) and input_vector.shape[1]==1:
                self.received_codeword=spmatrix2char(input_vector,self.received_codeword)
            elif isinstance(input_vector,np.ndarray):
                self.received_codeword=numpy2char(input_vector,self.received_codeword)
            else:
                raise ValueError("The input to ldpc.decode must either be of type `np.ndarray` or `scipy.sparse.spmatrix`.")
            
            mod2sparse_mulvec(self.H,self.received_codeword,self.synd)
            self.bp_decode_cy()
            for i in range(self.n):
                self.bp_decoding[i]=self.bp_decoding[i]^self.received_codeword[i]

        elif input_length ==self.m and (self.input_vector_type == 0 or self.input_vector_type == -1):
            self.input_vector_type = 0
            if isinstance(input_vector,spmatrix) and input_vector.shape[1]==1:
                self.synd=spmatrix2char(input_vector,self.synd)
            elif isinstance(input_vector,np.ndarray):
                self.synd=numpy2char(input_vector,self.synd)
            else:
                raise ValueError("The input to ldpc.decode must either be of type `np.ndarray` or `scipy.sparse.spmatrix`.")
            self.bp_decode_cy()
        
        else:
            raise ValueError(f"The input to the ldpc.bp_decoder.decode must be either a received codeword (of length={self.n}) or a syndrome (of length={self.m}). The inputted vector has length={input_length}. Valid formats are `np.ndarray` or `scipy.sparse.spmatrix`.")
        
        return char2numpy(self.bp_decoding,self.n)


    def update_channel_probs(self,channel):
        """
        Function updates the channel probabilities for each bit in the BP decoder.

        Parameters
        ----------
        channel: numpy.ndarray
            A list of the channel probabilities for each bit

        Returns
        -------
        NoneType
        """
        cdef j
        for j in range(self.n): self.channel_probs[j]=channel[j]

    cdef char* bp_decode_cy(self):
        """
        Cython function for calling the BP decoder

        Notes
        -----
        This funciton accepts no parameters. The syndrome must be set beforehand:

        eg. self.synd=syndrome
        """
        if self.bp_method == 0 or self.bp_method == 1:
            self.bp_decode_prob_ratios()

        elif self.bp_method == 2 or self.bp_method==3:
            self.bp_decode_log_prob_ratios()

        else:
            ValueError("Specified BP method is invalid.")

    cdef int bp_decode_prob_ratios(self):
        """
        Cython function implementing belief propagation for probability ratios.

        Notes
        -----
        This function accepts no parameters. The syndrome must be set beforehand.
        """

        cdef mod2entry *e
        cdef int i, j, check,equal, iteration
        cdef double bit_to_check0, temp

        #initialisation

        for j in range(self.n):
            e=mod2sparse_first_in_col(self.H,j)
            while not mod2sparse_at_end(e):
                e.bit_to_check=self.channel_probs[j]/(1-self.channel_probs[j])
                e=mod2sparse_next_in_col(e)

        self.converge=0
        for iteration in range(1,self.max_iter+1):

            self.iter=iteration

            if self.ms_scaling_factor==0:
                alpha = 1.0 - 2**(-1*iteration/1.0)
            else: alpha = self.ms_scaling_factor

            #check-to-bit messages

            #product sum updates
            if self.bp_method==0:

                for i in range(self.m):

                    e=mod2sparse_first_in_row(self.H,i)
                    temp=((-1)**self.synd[i])
                    while not mod2sparse_at_end(e):
                        e.check_to_bit=temp
                        temp*=2/(1+e.bit_to_check) - 1
                        e=mod2sparse_next_in_row(e)

                    e=mod2sparse_last_in_row(self.H,i)
                    temp=1.0
                    while not mod2sparse_at_end(e):
                        e.check_to_bit*=temp
                        e.check_to_bit=(1-e.check_to_bit)/(1+e.check_to_bit)
                        temp*=2/(1+e.bit_to_check) - 1
                        e=mod2sparse_prev_in_row(e)

            #min-sum updates
            elif self.bp_method==1:
                for i in range(self.m):

                    e=mod2sparse_first_in_row(self.H,i)
                    temp=1e308

                    if self.synd[i]==1: sgn=1
                    else: sgn=0

                    while not mod2sparse_at_end(e):
                        e.check_to_bit=temp
                        e.sgn=sgn
                        if abs(abs(e.bit_to_check)-1)<temp:
                            temp=abs(e.bit_to_check)
                        if e.bit_to_check >=1: sgn+=1
                        e=mod2sparse_next_in_row(e)

                    e=mod2sparse_last_in_row(self.H,i)
                    temp=1e308
                    sgn=0
                    while not mod2sparse_at_end(e):
                        if temp < e.check_to_bit:
                            e.check_to_bit=temp
                        e.sgn+=sgn

                        e.check_to_bit=e.check_to_bit**(((-1)**e.sgn)*alpha)

                        if abs(e.bit_to_check)<temp:
                            temp=abs(e.bit_to_check)
                        if e.bit_to_check >=1: sgn+=1


                        e=mod2sparse_prev_in_row(e)

            # bit-to-check messages
            for j in range(self.n):

                e=mod2sparse_first_in_col(self.H,j)
                temp=self.channel_probs[j]/(1-self.channel_probs[j])

                while not mod2sparse_at_end(e):
                    e.bit_to_check=temp
                    temp*=e.check_to_bit
                    if isnan(temp):
                        temp=1.0
                    e=mod2sparse_next_in_col(e)

                self.log_prob_ratios[j]=log(1/temp)
                if temp >= 1:
                    self.bp_decoding[j]=1
                else: self.bp_decoding[j]=0

                e=mod2sparse_last_in_col(self.H,j)
                temp=1.0

                while not mod2sparse_at_end(e):
                    e.bit_to_check*=temp
                    temp*=e.check_to_bit
                    if isnan(temp):
                        temp=1.0
                    e=mod2sparse_prev_in_col(e)


            mod2sparse_mulvec(self.H,self.bp_decoding,self.bp_decoding_synd)

            equal=1
            for check in range(self.m):
                if self.synd[check]!=self.bp_decoding_synd[check]:
                    equal=0
                    break
            if equal==1:
                self.converge=1
                return 1

        return 0

    # Belief propagation with log probability ratios
    cdef int bp_decode_log_prob_ratios(self):
        """
        Cython function implementing belief propagation for log probability ratios.

        Notes
        -----
        This function accepts no parameters. The syndrome must be set beforehand.
        """

        cdef mod2entry *e
        cdef int i, j, check,equal, iteration, sgn
        cdef double bit_to_check0, temp, alpha

        #initialisation

        for j in range(self.n):
            e=mod2sparse_first_in_col(self.H,j)
            while not mod2sparse_at_end(e):
                e.bit_to_check=log((1-self.channel_probs[j])/self.channel_probs[j])
                e=mod2sparse_next_in_col(e)

        self.converge=0
        for iteration in range(1,self.max_iter+1):

            self.iter=iteration

            #product sum check_to_bit messages
            if self.bp_method==2:

                for i in range(self.m):

                    e=mod2sparse_first_in_row(self.H,i)
                    temp=1.0
                    while not mod2sparse_at_end(e):
                        e.check_to_bit=temp
                        temp*=tanh(e.bit_to_check/2)
                        e=mod2sparse_next_in_row(e)

                    e=mod2sparse_last_in_row(self.H,i)
                    temp=1.0
                    while not mod2sparse_at_end(e):
                        e.check_to_bit*=temp
                        e.check_to_bit=((-1)**self.synd[i])*log((1+e.check_to_bit)/(1-e.check_to_bit))
                        temp*=tanh(e.bit_to_check/2)
                        e=mod2sparse_prev_in_row(e)

            #min-sum check to bit messages
            if self.bp_method==3:

                if self.ms_scaling_factor==0:
                    alpha = 1.0 - 2**(-1*iteration/1.0)
                else: alpha = self.ms_scaling_factor

                for i in range(self.m):

                    e=mod2sparse_first_in_row(self.H,i)
                    temp=1e308

                    if self.synd[i]==1: sgn=1
                    else: sgn=0

                    while not mod2sparse_at_end(e):
                        e.check_to_bit=temp
                        e.sgn=sgn
                        if abs(e.bit_to_check)<temp:
                            temp=abs(e.bit_to_check)
                        if e.bit_to_check <=0: sgn+=1
                        e=mod2sparse_next_in_row(e)

                    e=mod2sparse_last_in_row(self.H,i)
                    temp=1e308
                    sgn=0
                    while not mod2sparse_at_end(e):
                        if temp < e.check_to_bit:
                            e.check_to_bit=temp
                        e.sgn+=sgn

                        e.check_to_bit*=((-1)**e.sgn)*alpha

                        if abs(e.bit_to_check)<temp:
                            temp=abs(e.bit_to_check)
                        if e.bit_to_check <=0: sgn+=1


                        e=mod2sparse_prev_in_row(e)

            # bit-to-check messages
            for j in range(self.n):

                e=mod2sparse_first_in_col(self.H,j)
                temp=log((1-self.channel_probs[j])/self.channel_probs[j])

                while not mod2sparse_at_end(e):
                    e.bit_to_check=temp
                    temp+=e.check_to_bit
                    # if isnan(temp): temp=0.0
                    e=mod2sparse_next_in_col(e)

                self.log_prob_ratios[j]=temp
                if temp <= 0: self.bp_decoding[j]=1
                else: self.bp_decoding[j]=0

                e=mod2sparse_last_in_col(self.H,j)
                temp=0.0
                while not mod2sparse_at_end(e):
                    e.bit_to_check+=temp
                    temp+=e.check_to_bit
                    # if isnan(temp): temp=0.0
                    e=mod2sparse_prev_in_col(e)


            mod2sparse_mulvec(self.H,self.bp_decoding,self.bp_decoding_synd)

            equal=1
            for check in range(self.m):
                if self.synd[check]!=self.bp_decoding_synd[check]:
                    equal=0
                    break
            if equal==1:
                self.converge=1
                return 1

        return 0


    @property
    def channel_probs(self):
        """
        numpy.ndarray: The initial error channel probabilities
        """
        probs=np.zeros(self.n).astype("float")
        for j in range(self.n):
            probs[j]=self.channel_probs[j]

        return probs

    # @property
    # def bp_probs(self):
    #     """
    #     Getter fo the soft-decision probabilities from the last round of BP decoding

    #     Returns
    #     -------
    #     numpy.ndarray
    #     """
    #     probs=np.zeros(self.n).astype("float")
    #     for j in range(self.n):
    #         probs[j]=self.log_prob_ratios[j]

    #     return probs

    @property
    def bp_method(self):
        """
        Getter for the BP method

        Returns
        -------
        str
        """
        if self.bp_method==0: return "product_sum"
        elif self.bp_method==1: return "minimum_sum"
        elif self.bp_method==2: return "product_sum_log"
        elif self.bp_method==3: return "minimum_sum_log"

    @property
    def iter(self):
        """
        Getter. Returns the number of iterations in the last round of BP decoding.

        Returns
        -------
        numpy.ndarray
        """
        return self.iter

    @property
    def ms_scaling_factor(self):
        """
        Getter. Returns the min-sum scaling factor.

        Returns
        -------
        float64
        """
        return self.ms_scaling_factor

    @property
    def max_iter(self):
        """
        Getter. Returns the maximum interation depth for the BP decoder.

        Returns
        -------
        int
        """
        return self.max_iter

    @property
    def converge(self):
        """
        Getter. Returns `1' if the last round of BP succeeded (converged) and `0' if it failed.

        Returns
        -------
        int
        """
        return self.converge

    @property
    def bp_decoding(self):
        """
        Getter. Returns the output from the last round of BP decoding.

        Returns
        -------
        numpy.ndarray
        """
        return char2numpy(self.bp_decoding,self.n)

    @property
    def log_prob_ratios(self):
        """
        Getter. Returns the soft-decision log probability ratios from the last round of BP
        decoding.

        Returns
        -------
        numpy.ndarray
        """
        return double2numpy(self.log_prob_ratios,self.n)

    # @property
    # def channel_probs(self):
    #     """
    #     Getter.

    #     Returns
    #     -------
    #     numpy.ndarray
    #     """
    #     return double2numpy(self.channel_probs,self.n)

    def __dealloc__(self):
            if self.MEM_ALLOCATED:
                free(self.error)
                free(self.synd)
                free(self.bp_decoding_synd)
                free(self.channel_probs)
                free(self.bp_decoding)
                free(self.log_prob_ratios)
                free(self.received_codeword)
                mod2sparse_free(self.H)


# cdef class bposd_decoder(bp_decoder):

#     def __cinit__(self,parity_check_matrix,**kwargs):
#         self.test=149
#         pass

#     def print_test(self):
#         return self.test















