#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix

cdef class UnionFindDecoder:

    def __cinit__(self,pcm, error_rate=None, error_channel=None, bits_per_step = 1):

        cdef i, j
        self.MEMORY_ALLOCATED=False

        #check the parity check matrix is the right type
        if isinstance(pcm, np.ndarray) or isinstance(pcm, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        #MEMORY ALLOCATION
        self.pcm = new bp_sparse(self.m,self.n) #createst the C++ sparse matrix object
        self.error_channel.resize(self.n) #C vector for the error channel
        self.syndrome.resize(self.m) #C vector for the syndrome
        self.bit_weights.resize(self.n)


        ## error channel setup
        # if error_rate is None:
        #     if error_channel is None:
        #         raise ValueError("Please specify the error channel. Either: 1) error_rate: float or 2) error_channel: list of floats of length equal to the block length of the code {self.n}.")

        if error_rate is not None:
            if error_channel is None:
                if not isinstance(error_rate,float):
                    raise ValueError("The `error_rate` parameter must be specified as a single float value.")
                for i in range(self.n): self.error_channel[i] = error_rate

        if error_channel is not None:
            if len(error_channel)!=self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(error_channel)}.")
            for i in range(self.n): self.error_channel[i] = error_channel[i]


        #fill sparse matrix
        if isinstance(pcm,np.ndarray):
            for i in range(self.m):
                for j in range(self.n):
                    if pcm[i,j]==1:
                        self.pcm.insert_entry(i,j,1)
        elif isinstance(pcm,spmatrix):
            rows, cols = pcm.nonzero()
            for i in range(len(rows)):
                self.pcm.insert_entry(rows[i], cols[i], pcm[rows[i], cols[i]])
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")


        self.bits_per_step = bits_per_step


        self.ufd = new uf_decoder_cpp(self.pcm)
        self.MEMORY_ALLOCATED=True

    def decode(self,syndrome, bit_weights = None):
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        DTYPE = syndrome.dtype
        for i in range(self.m): self.syndrome[i] = syndrome[i]

        if bit_weights is not None:
            for i in range(self.n): self.bit_weights[i] = bit_weights[i]
            self.ufd.peel_decode(self.syndrome, self.bit_weights, self.bits_per_step)
        else:
            self.ufd.peel_decode(self.syndrome, NULL_DOUBLE_VECTOR, 1)
            
        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.ufd.decoding[i]
        return out

    def matrix_decode(self,syndrome, bit_weights = None):
        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        cdef int i
        DTYPE = syndrome.dtype
        for i in range(self.m): self.syndrome[i] = syndrome[i]

        if bit_weights is not None:
            for i in range(self.n): self.bit_weights[i] = bit_weights[i]
            self.ufd.matrix_decode(self.syndrome, self.bit_weights, self.bits_per_step)
        else:
            self.ufd.matrix_decode(self.syndrome, NULL_DOUBLE_VECTOR, 1)
            
        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.ufd.decoding[i]
        return out

    # def matrix_decode(self,syndrome):
    #     if not len(syndrome)==self.m:
    #         raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
    #     cdef int i
    #     DTYPE = syndrome.dtype
    #     for i in range(self.m): self.syndrome[i] = syndrome[i]
    #     self.ufd.matrix_decode(self.syndrome)
    #     out = np.zeros(self.n,dtype=DTYPE)
    #     for i in range(self.n): out[i] = self.ufd.decoding[i]
    #     return out

    # def bposd_decode(self,syndrome):
    #     if not len(syndrome)==self.m:
    #         raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
    #     cdef int i
    #     DTYPE = syndrome.dtype
    #     for i in range(self.m): self.syndrome[i] = syndrome[i]
    #     self.ufd.bposd_decode(self.syndrome)
    #     out = np.zeros(self.n,dtype=DTYPE)
    #     for i in range(self.n): out[i] = self.ufd.decoding[i]
    #     return out