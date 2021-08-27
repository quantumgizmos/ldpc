#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np

cdef class bp_decoder:

    '''
    This class implements a belief propagation decoder for LDPC codes   
    '''

    def __init__(self,mat, error_rate=None, max_iter=0, bp_method=0, ms_scaling_factor=1.0,channel_probs=[None]):

        pass

    def __cinit__(self,mat, error_rate=None, max_iter=0, bp_method=0, ms_scaling_factor=1.0,channel_probs=[None]):

        self.MEM_ALLOCATED=False

        cdef i,j
        self.m=mat.shape[0]
        self.n=mat.shape[1]

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
            bp_method=1
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
        self.H=pymod2sparse(mat) #parity check matrix in sparse form
        assert self.n==self.H.shape[1] #validate number of bits in mod2sparse format
        assert self.m==self.H.shape[0] #validate number of checks in mod2sparse format
        self.error=<char*>calloc(self.n,sizeof(char)) #error string
        self.synd=<char*>calloc(self.m,sizeof(char)) #syndrome string
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
        

    cdef char* bp_decode_cy(self):

        if self.bp_method == 0 or self.bp_method == 1:
            self.bp_decode_prob_ratios()

        # elif self.bp_method == 2 or self.bp_method==3:
        #     self.bp_decode_log_prob_ratios()

        else:
            print("Decoder not called")

    cpdef np.ndarray[np.int_t, ndim=1] bp_decode(self, np.ndarray[np.int_t, ndim=1] syndrome):
        self.synd=numpy2char(syndrome,self.synd)
        self.bp_decode_cy()
        return char2numpy(self.bp_decoding,self.n)

    # Belief propagation with probability ratios
    cdef int bp_decode_prob_ratios(self):
   
        cdef pymod2sparse e
        cdef int i, j, check,equal, iteration
        cdef double bit_to_check0, temp

        #initialisation

        for j in range(self.n):
            for e in self.H.iter_col(j,reverse_iterate=False):
                e.bit_to_check=self.channel_probs[j]/(1-self.channel_probs[j])

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
                    temp=((-1)**self.synd[i])
                    for e in self.H.iter_row(i,reverse_iterate=False):
                        e.check_to_bit=temp
                        temp*=2/(1+e.bit_to_check) - 1

                    temp=1.0
                    for e in self.H.iter_row(i,reverse_iterate=True):
                        e.check_to_bit*=temp
                        e.check_to_bit=(1-e.check_to_bit)/(1+e.check_to_bit)
                        temp*=2/(1+e.bit_to_check) - 1

            # #min-sum updates
            # elif self.bp_method==1:
            #     for i in range(self.m):

            #         e=mod2sparse_first_in_row(self.H,i)
            #         temp=1e308
                    
            #         if self.synd[i]==1: sgn=1
            #         else: sgn=0

            #         while not mod2sparse_at_end(e):
            #             e.check_to_bit=temp
            #             e.sgn=sgn
            #             if abs(abs(e.bit_to_check)-1)<temp:
            #                 temp=abs(e.bit_to_check)
            #             if e.bit_to_check <=0: sgn+=1
            #             e=mod2sparse_next_in_row(e)

            #         e=mod2sparse_last_in_row(self.H,i)
            #         temp=1e308
            #         sgn=0
            #         while not mod2sparse_at_end(e):
            #             if temp < e.check_to_bit:
            #                 e.check_to_bit=temp
            #             e.sgn+=sgn

            #             e.check_to_bit=e.check_to_bit**(((-1)**e.sgn)*alpha)

            #             if abs(e.bit_to_check)<temp:
            #                 temp=abs(e.bit_to_check)
            #             if e.bit_to_check <=0: sgn+=1


            #             e=mod2sparse_prev_in_row(e)

            # bit-to-check messages
            for j in range(self.n):

                temp=self.channel_probs[j]/(1-self.channel_probs[j])

                for e in self.H.iter_col(j,reverse_iterate=False):
                    e.bit_to_check=temp
                    temp*=e.check_to_bit
                    if isnan(temp):
                        temp=1.0

                # self.log_prob_ratios[j]=1/temp)
                if temp >= 1:
                    self.bp_decoding[j]=1
                else: self.bp_decoding[j]=0

                temp=1.0

                for e in self.H.iter_col(j,reverse_iterate=True):
                    e.bit_to_check*=temp
                    temp*=e.check_to_bit
                    if isnan(temp):
                        temp=1.0

            mod2sparse_mulvec(self.H.matrix,self.bp_decoding,self.bp_decoding_synd)

            equal=1
            for check in range(self.m):
                if self.synd[check]!=self.bp_decoding_synd[check]:
                    equal=0
                    break
            if equal==1:
                self.converge=1
                return 1

        return 0

    def update_channel_probs(self,channel):
        cdef j
        for j in range(self.n): self.channel_probs[j]=channel[j]

    @property
    def channel_probs(self):
        probs=np.zeros(self.n).astype("float")
        for j in range(self.n):
            probs[j]=self.channel_probs[j]

        return probs

    @property
    def bp_probs(self):
        probs=np.zeros(self.n).astype("float")
        for j in range(self.n):
            probs[j]=self.log_prob_ratios[j]

        return probs

    @property
    def bp_method(self):
        if self.bp_method==0: return "product_sum"
        elif self.bp_method==1: return "minimum_sum"
        elif self.bp_method==2: return "product_sum_log"
        elif self.bp_method==3: return "minimum_sum_log"

    @property
    def iter(self): return self.iter

    @property
    def ms_scaling_factor(self): return self.ms_scaling_factor

    @property
    def max_iter(self): return self.max_iter

    @property
    def converge(self): return self.converge

    @property
    def osd_order(self): return self.osd_order

    @property
    def osdw_decoding(self): return char2numpy(self.osdw_decoding,self.n)

    @property
    def bp_decoding(self): return char2numpy(self.bp_decoding,self.n)

    @property
    def osd0_decoding(self): return char2numpy(self.osd0_decoding,self.n)

    @property
    def log_prob_ratios(self): return double2numpy(self.log_prob_ratios,self.n)

    @property
    def channel_probs(self): return double2numpy(self.channel_probs,self.n)


    ###note dealloc is causing exception handling to fail. Don't know why this is happening?

    def __dealloc__(self):
            if self.MEM_ALLOCATED:
                free(self.error)
                free(self.synd)
                free(self.bp_decoding_synd)
                free(self.channel_probs)
                free(self.bp_decoding)
                free(self.log_prob_ratios)

        












