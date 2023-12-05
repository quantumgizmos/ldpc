#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
from ldpc.bp_decoder import bp_decoder
from scipy.sparse import spmatrix
from scipy.special import comb as nCr

cdef class bposd_decoder(bp_decoder):
    '''
    A class implementing a belief propagation plus ordered statistics decoding for LDPC codes

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
    osd_order: str or int, optional
        Sets the OSD order.
    osd_method: str or int, optional
        The OSD method. Currently three methods are availbe: 1) "osd_0": Zero-oder OSD; 2) "osd_e": exhaustive OSD;
        3) "osd_cs": combination-sweep OSD.

    '''
    
    def __cinit__(self,parity_check_matrix,**kwargs):

        #OSD specific input parameters
        osd_method=kwargs.get("osd_method",1)
        osd_order=kwargs.get("osd_order",-1)

        self.MEM_ALLOCATED=False

        cdef i,j

        #memory allocation for OSD specific attributes
        self.osd0_decoding=<char*>calloc(self.n,sizeof(char)) #the OSD_0 decoding
        self.osdw_decoding=<char*>calloc(self.n,sizeof(char)) #the osd_w decoding

        #OSD setup

        # OSD method
        if str(osd_method).lower() in ['OSD_0','osd_0','0','osd0']:
            osd_method=0
            osd_order=0
        elif str(osd_method).lower() in ['osd_e','1','osde','exhaustive','e']:
            osd_method=1
            if osd_order>15:
                print("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        elif str(osd_method).lower() in ['osd_cs','2','osdcs','combination_sweep','combination_sweep','cs']:
            osd_method=2
        else:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")

        self.osd_order=int(osd_order)
        self.osd_method=int(osd_method)

        self.encoding_input_count=0
        
        if self.osd_order>-1:
            self.rank=mod2sparse_rank(self.H)
            try:
                assert self.osd_order<=(self.n - self.rank)
            except AssertionError:
                self.osd_order=-1
                raise ValueError(f"For this code, the OSD order should be set in the range 0<=osd_oder<={self.n - self.rank}.")
            self.cols=<int*>calloc(self.n,sizeof(int)) 
            self.orig_cols=<int*>calloc(self.n,sizeof(int))
            self.rows=<int*>calloc(self.m,sizeof(int))
            self.k=self.n-self.rank

        if self.osd_order>0:
            self.y=<char*>calloc(self.n,sizeof(char))
            self.g=<char*>calloc(self.m,sizeof(char))
            self.Htx=<char*>calloc(self.m,sizeof(char))
            self.Ht_cols=<int*>calloc(self.k,sizeof(int)) 

        if osd_order==0: pass
        elif self.osd_order>0 and self.osd_method==1: self.osd_e_setup()
        elif self.osd_order>0 and self.osd_method==2: self.osd_cs_setup()
        elif self.osd_order==-1: pass
        else: raise Exception(f"ERROR: OSD method '{osd_method}' invalid")

        self.MEM_ALLOCATED=True

    cdef void osd_e_setup(self):

        self.encoding_input_count=int(2**self.osd_order)
        self.osdw_encoding_inputs=<char**>calloc(self.encoding_input_count,sizeof(char*))
        for i in range(self.encoding_input_count):
            self.osdw_encoding_inputs[i] = decimal_to_binary_reverse(i, self.n - self.rank)

    cdef void osd_cs_setup(self):

        cdef int kset_size=self.n-self.rank

        assert self.osd_order<=kset_size


        self.encoding_input_count=kset_size+nCr(self.osd_order,2)
        
        self.osdw_encoding_inputs=<char**>calloc(self.encoding_input_count,sizeof(char*))
        cdef int total_count=0
        for i in range(kset_size):
            self.osdw_encoding_inputs[total_count] = <char*>calloc(kset_size,sizeof(char))
            self.osdw_encoding_inputs[total_count][i]=1
            total_count+=1

        for i in range(self.osd_order):
            for j in range(self.osd_order):
                if i<j:
                    self.osdw_encoding_inputs[total_count] = <char*>calloc(kset_size,sizeof(char))
                    self.osdw_encoding_inputs[total_count][i]=1
                    self.osdw_encoding_inputs[total_count][j]=1
                    total_count+=1

        assert total_count==self.encoding_input_count


    cdef char* decode_cy(self, char* syndrome):

        self.synd=syndrome

        # print(char2numpy(self.synd,self.m))

        self.bp_decode_cy()

        # print(double2numpy(self.log_prob_ratios,self.n))


        if self.osd_order==-1: return self.bp_decoding

        #if BP has converged, return the BP solution
        if self.converge==1:
            for j in range(self.n): self.osd0_decoding[j]=self.osdw_decoding[j]=self.bp_decoding[j]
            return self.osd0_decoding

        #if BP doesn't converge, run OSD post-processing
        self.osd()

        if self.osd_order==0:
            for j in range(self.n): self.osdw_decoding[j]=self.osd0_decoding[j]
            return self.osd0_decoding
        else:
            return self.osdw_decoding

    
    cpdef np.ndarray[np.int_t, ndim=1] decode(self, input_vector):

        """
        Runs the BP+OSD decoder for a given syndrome.

        Parameters
        ----------

        input_vector: numpy.ndarray or scipy.sparse.spmatrix
            The syndrome to be decoded.

        Returns
        -------
        numpy.ndarray
            The BP+OSD decoding in numpy.ndarray format.
        """

        cdef int input_length = input_vector.shape[0]
        cdef int i

        if input_length ==self.m:
            if isinstance(input_vector,spmatrix) and input_vector.shape[1]==1:
                self.synd=spmatrix2char(input_vector,self.synd)
            elif isinstance(input_vector,np.ndarray):
                self.synd=numpy2char(input_vector,self.synd)
            else:
                raise ValueError("The input to ldpc.decode must either be of type `np.ndarray` or `scipy.sparse.spmatrix`.")
            
            self.decode_cy(self.synd)
        
        else:
            raise ValueError(f"The input to the ldpc.bp_decoder.decode must be a syndrome (of length={self.m}). The inputted vector has length={input_length}. Valid formats are `np.ndarray` or `scipy.sparse.spmatrix`.")
        
        if self.osd_order==-1: return char2numpy(self.bp_decoding,self.n)
        else: return char2numpy(self.osdw_decoding,self.n)

    #OSD Post-processing
    cdef int osd(self):
        cdef int i, j
        cdef long int l
        cdef mod2sparse *L
        cdef mod2sparse *U

        #allocating L and U matrices 
        L=mod2sparse_allocate(self.m,self.rank)
        U=mod2sparse_allocate(self.rank,self.n)

        #sort the columns on the basis of the soft decisions
        soft_decision_col_sort(self.log_prob_ratios,self.cols, self.n)

        #save the original sorted column order
        for i in range(self.n):
            self.orig_cols[i]=self.cols[i]

        #find the LU decomposition of the ordered matrix
        mod2sparse_decomp_osd(
            self.H,
            self.rank,
            L,
            U,
            self.rows,
            self.cols)


        #solve the syndrome equation with most probable full-rank submatrix
        LU_forward_backward_solve(
            L,
            U,
            self.rows,
            self.cols,
            self.synd,
            self.osd0_decoding)


        if self.osd_order==0:
            mod2sparse_free(U)
            mod2sparse_free(L)
            return 1


        #return the columns outside of the information set to their orginal ordering (the LU decomp scrambles them)
        cdef int check, counter, in_pivot
        cdef mod2sparse* Ht=mod2sparse_allocate(self.m,self.k)

        counter=0

        for i in range(self.n):
            check=self.orig_cols[i]
            in_pivot=0
            for j in range(self.rank):
                if self.cols[j]==check:
                    in_pivot=1
                    break
            
            if in_pivot==0:
                self.cols[counter+self.rank]=check
                counter+=1

        #create the HT matrix
        for i in range(self.k):
            self.Ht_cols[i]=self.cols[i+self.rank]

        mod2sparse_copycols(self.H,Ht,self.Ht_cols)



        # cdef osd_0_weight=bin_char_weight(self.osd0_decoding,self.n)

        cdef double osd_min_weight=0
        for i in range(self.n):
            # osd_min_weight+=self.osd0_decoding[i]*log(1/self.channel_probs[i])
            # osd_min_weight+=self.osd0_decoding[i]
            if self.osd0_decoding[i]==1:
                osd_min_weight+=log(1/self.channel_probs[i])


        for i in range(self.n):
            self.osdw_decoding[i]=self.osd0_decoding[i]

        cdef double solution_weight
        cdef char *x



        for l in range(self.encoding_input_count):
            x=self.osdw_encoding_inputs[l]
            mod2sparse_mulvec(Ht,x,self.Htx)
            for i in range(self.m):
                self.g[i]=self.synd[i]^self.Htx[i]

            LU_forward_backward_solve(
                L,
                U,
                self.rows,
                self.cols,
                self.g,
                self.y)

            for i in range(self.k):
                self.y[self.Ht_cols[i]]=x[i]

            solution_weight=0.0
            for i in range(self.n):
                # solution_weight+=self.y[i]*log(1/self.channel_probs[i])
                # solution_weight+=self.y[i]
                if self.y[i]==1:
                    solution_weight+=log(1/self.channel_probs[i])

            if solution_weight<osd_min_weight:
                osd_min_weight=solution_weight
                for i in range(self.n):
                    self.osdw_decoding[i]=self.y[i]

        mod2sparse_free(Ht)
        mod2sparse_free(U)
        mod2sparse_free(L)
        return 1


    @property
    def osd_method(self):
        """
        Getter. Returns the OSD method.
        
        Returns
        -------
        str
        """
        if self.osd_order==-1: return None
        if self.osd_method==0: return "osd_0"
        if self.osd_method==1: return "osd_e"
        if self.osd_method==2: return "osd_cs"
    
    @property
    def osd_order(self):
        """
        Getter. Returns the OSD order.
        
        Returns
        -------
        int
        """
        return self.osd_order

    @property
    def osdw_decoding(self):
        """
        Getter. Returns the recovery vector from the last round of BP+OSDW decoding.
        
        Returns
        -------
        numpy.ndarray
        """        
        return char2numpy(self.osdw_decoding,self.n)

    @property
    def osd0_decoding(self):
        """
        Getter. Returns the recovery vector from the last round of BP+OSD0 decoding.
        
        Returns
        -------
        numpy.ndarray
        """        
        return char2numpy(self.osd0_decoding,self.n)


    def __dealloc__(self):
        
        if self.MEM_ALLOCATED==True:
    
            free(self.osd0_decoding)
            free(self.osdw_decoding)

            if self.osd_order>-1:
                free(self.cols)
                free(self.rows)
                free(self.orig_cols)

            if self.osd_order>0:
                free(self.Htx)
                free(self.g)
                free(self.y)
                free(self.Ht_cols)

            if self.encoding_input_count!=0:
                for i in range(self.encoding_input_count):
                    free(self.osdw_encoding_inputs[i])
















