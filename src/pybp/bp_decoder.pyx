#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc,free
from libc.math cimport log, tanh, isnan, abs

from mod2sparse cimport *
from c_util cimport numpy2char, char2numpy, numpy2double, double2numpy

cdef class bp_decoder:
    cdef mod2sparse* H
    cdef int m, n
    cdef char* error
    cdef char* synd
    cdef char* bp_decoding_synd
    cdef char* bp_decoding
    cdef char* decoding
    cdef int iter
    cdef int converge
    cdef double* channel_probs
    cdef double* log_prob_ratios
    cdef double error_rate
    cdef int max_iter
    cdef int bp_method
    cdef double ms_scaling_factor

    def __cinit__(self,mat, error_rate=None, max_iter=0, bp_method=0, ms_scaling_factor=1.0,channel_probs=[None]):

        cdef i,j

        self.m=mat.shape[0]
        self.n=mat.shape[1]

        #BP Settings
        self.max_iter=max_iter
        self.bp_method=bp_method
        self.ms_scaling_factor=ms_scaling_factor

        #memory allocation
        self.H=numpy2mod2sparse(mat) #parity check matrix in sparse form
        assert self.n==self.H.n_cols #validate number of bits in mod2sparse format
        assert self.m==self.H.n_rows #validate number of checks in mod2sparse format
        self.error=<char*>calloc(self.n,sizeof(char)) #error string
        self.synd=<char*>calloc(self.m,sizeof(char)) #syndrome string
        self.bp_decoding_synd=<char*>calloc(self.m,sizeof(char)) #decoded syndrome string
        self.bp_decoding=<char*>calloc(self.n,sizeof(char)) #BP decoding
        self.channel_probs=<double*>calloc(self.n,sizeof(double)) #channel probs
        self.log_prob_ratios=<double*>calloc(self.n,sizeof(double)) #log probability ratios

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

        elif self.bp_method == 2 or self.bp_method==3:
            self.bp_decode_log_prob_ratios()

        else:
            print("Decoder not called")

    cpdef np.ndarray[np.int_t, ndim=1] bp_decode(self, np.ndarray[np.int_t, ndim=1] syndrome):
        self.synd=numpy2char(syndrome,self.synd)
        self.bp_decode_cy()
        return char2numpy(self.bp_decoding,self.n)

    # Belief propagation with probability ratios
    cdef int bp_decode_prob_ratios(self):
   
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
                        if e.bit_to_check <=0: sgn+=1
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
                        if e.bit_to_check <=0: sgn+=1


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

                # self.log_prob_ratios[j]=1/temp)
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
        elif self.bp_method==1: return "mininum_sum"
        elif self.bp_method==2: return "product_sum_log"
        elif self.bp_method==3: return "mininum_sum_log"

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



    def __dealloc__(self):
        
        mod2sparse_free(self.H)
        free(self.error)
        free(self.synd)
        free(self.bp_decoding_synd)
        free(self.channel_probs)
        free(self.bp_decoding)
        free(self.log_prob_ratios)










