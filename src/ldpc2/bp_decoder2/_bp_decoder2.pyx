#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class bp_decoder:

    def __init__(self,parity_check_matrix, error_rate, max_iter, ms_scaling_factor):
        pass

    def __cinit__(self, parity_check_matrix, error_rate, max_iter, ms_scaling_factor):

        cdef int m,n, max_col_nnz, max_row_nnz, i,j
        self.m = parity_check_matrix.shape[0]
        self.n = parity_check_matrix.shape[1]
        self.syndrome.resize(self.m)
        max_row_nnz = max(np.sum(parity_check_matrix != 0, axis=1))
        max_col_nnz = max(np.sum(parity_check_matrix != 0, axis=0))

        # print(self.m,self.n,max_row_nnz,max_col_nnz)

        self.pcm = new gf2csr(self.m,self.n,max_row_nnz,max_col_nnz)

        for i in range(self.m):
            for j in range(self.n):
                if parity_check_matrix[i,j]:
                    self.pcm.insert_entry(i,j)
        

        self.error_rate = error_rate
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor

        # self.pcm.print()
        # print(self.pcm.row_count, self.pcm.col_count)

        self.bpd = new bp_decoder_cpp(self.pcm, self.error_rate, self.max_iter, self.ms_scaling_factor)

    def decode(self,syndrome):
        # if not len(syndrome)==self.m:
        #     raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        cdef bool zero_syndrome = True
        DTYPE = syndrome.dtype
        
        for i in range(self.m):
            self.syndrome[i] = syndrome[i]
            if self.syndrome[i]: zero_syndrome = False
        if zero_syndrome: return np.zeros(self.n,dtype=DTYPE)

        # print("hello")
        
        self.bpd.decode(self.syndrome)

        # print("hello")

        out = np.zeros(self.n,dtype=DTYPE)
        # print(out)
        for i in range(self.n):
            # print(i)
            out[i] = self.bpd.decoding[i]
        return out

    @property
    def converge(self):
        return self.bpd.converge

    @property
    def log_prob_ratios(self):
        out=np.zeros(self.n)
        for i in range(self.n): out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def iter(self):
        return self.bpd.iterations


        
       




