#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: language = c++
import numpy as np
from typing import Tuple, Dict, List, Union
import warnings
import scipy

cdef class mbp_decoder:

    def __cinit__(self,
        Hgf4: Union[scipy.sparse, np.ndarray] = None,
        HX_CSS: Union[scipy.sparse, np.ndarray] = None,
        HZ_CSS: Union[scipy.sparse, np.ndarray] = None,
        error_rate: float = None,
        xyz_bias: List[float] = [1,1,1],
        error_channel: List[List[float]]=None,
        max_iter: int = 0,
        alpha_parameter: np.ndarray[np.ndarray[float]] = None,
        beta_parameter: double = 0.0,
        bp_method: int = 0,
        gamma_parameter: float = 1.0):

        self.MEMORY_ALLOCATED=False
        cdef i, j

        if Hgf4 is not None:
            self.stab_count,self.qubit_count = Hgf4.shape[0], Hgf4.shape[1]
            self.OUTPUT_TYPE = 4
        elif HX_CSS is not None and HZ_CSS is not None:
            self.OUTPUT_TYPE = 2
            self.stab_count = HZ_CSS.shape[0] + HX_CSS.shape[0]
            self.qubit_count = HX_CSS.shape[1]
            try:
                assert HX_CSS.shape[1] == HZ_CSS.shape[1]
            except AssertionError:
                raise ValueError("The number of columns in HX_CSS should be equal to the number of columns in HZ_CSS.")
        else:
            raise ValueError("Please enter either the GF4 parity check matrix, or the GF2 HX and HZ CSS parity check components.")

        self.max_iter = max_iter
        if max_iter== 0: self.max_iter = self.qubit_count

        self.syndrome.resize(self.stab_count)
        self.error_channel.resize(3)
        for i in range(3):
            self.error_channel[i].resize(self.qubit_count)

        xyz_bias = np.array(xyz_bias)
        if np.sum(xyz_bias) == 0:
            for i in range(3): self.xyz_bias[i] = 0
        elif np.sum(xyz_bias)>0:
            xyz_bias=xyz_bias/np.sum(xyz_bias)
            for i in range(3): self.xyz_bias[i] = xyz_bias[i]

        if error_rate is not None and error_channel is None:
            self.error_rate = error_rate
            px, py, pz = xyz_bias*error_rate

            for i in range(self.qubit_count):
                self.error_channel[0][i] = px
                self.error_channel[1][i] = py
                self.error_channel[2][i] = pz

        elif error_channel is not None:
            if error_rate is not None:
                warnings.warn(f"An error channel has been provided as input. This will override the 'error_rate={error_rate}' parameter that has also been inputted.")
            for i in range(self.qubit_count):
                self.error_channel[0][i] = error_channel[0][i]
                self.error_channel[1][i] = error_channel[1][i]
                self.error_channel[2][i] = error_channel[2][i]

        self.update_alpha(alpha_parameter)

        self.pcm = new mbp_sparse(self.stab_count,self.qubit_count)
        self.pcmX = new bp_sparse(HX_CSS.shape[0],self.qubit_count)
        self.pcmZ = new bp_sparse(HZ_CSS.shape[0],self.qubit_count)

        if Hgf4 is not None:
            if isinstance(Hgf4, np.ndarray):
                for i in range(self.stab_count):
                    for j in range(self.qubit_count):
                        if Hgf4[i,j]!=0:
                            self.pcm.insert_entry(i,j,Hgf4[i,j])
            elif scipy.sparse.issparse(Hgf4):
                rows, cols = Hgf4.nonzero()
                for i in range(len(rows)):
                    self.pcm.insert_entry(rows[i], cols[i], Hgf4[rows[i], cols[i]])


        elif HX_CSS is not None and HZ_CSS is not None:
            if isinstance(HZ_CSS, np.ndarray) and isinstance(HX_CSS, np.ndarray):
                for i in range(HX_CSS.shape[0]):
                    for j in range(self.qubit_count):
                        if HX_CSS[i,j]!=0:
                            self.pcmX.insert_entry(i,j,1)
                            self.pcm.insert_entry(i+HZ_CSS.shape[0],j,1)
                for i in range(HZ_CSS.shape[0]):
                    for j in range(self.qubit_count):
                        if HZ_CSS[i,j]!=0:
                            self.pcmZ.insert_entry(i,j,1)
                            self.pcm.insert_entry(i,j,3)
            elif scipy.sparse.issparse(HX_CSS) and scipy.sparse.issparse(HZ_CSS):
                rows, cols = HX_CSS.nonzero()
                for i in range(len(rows)):
                    self.pcmX.insert_entry(rows[i],cols[i],1)
                    self.pcm.insert_entry(rows[i]+HZ_CSS.shape[0], cols[i], 1)
                rows, cols = HZ_CSS.nonzero()
                for i in range(len(rows)):
                    self.pcmZ.insert_entry(rows[i],cols[i],1)
                    self.pcm.insert_entry(rows[i], cols[i], 3)
            else:
                raise ValueError("The HX_CSS and HZ_CSS matrices must be either of type np.ndarray or scipy.sparse.")
        else:
            raise ValueError("Please enter either the GF4 parity check matrix, or the GF2 HX and HZ CSS parity check components.")

        # self.pcm.print()


        if str(bp_method).lower() in ['prod_sum','product_sum','ps','0','prod sum']:
            self.bp_method=0
        elif str(bp_method).lower() in ['min_sum','minimum_sum','ms','1','minimum sum','min sum']:
            self.bp_method=1
        else: raise ValueError(f"BP method '{bp_method}' is invalid.\
                            Please choose from the following methods:'product_sum',\
                            'minimum_sum' or 'product_sum'")

        self.beta_parameter=beta_parameter
        self.gamma_parameter=gamma_parameter


        self.bpd = new mbp_decoder_cpp(self.pcm,self.error_channel,self.max_iter,self.alpha,self.beta_parameter,self.bp_method,self.gamma_parameter)
        self.uf_bpdX = new uf_decoder_cpp(self.pcmZ)
        self.uf_bpdZ = new uf_decoder_cpp(self.pcmX)

        self.MEMORY_ALLOCATED = True

    def __del__(self):
        cdef i
        if self.MEMORY_ALLOCATED:
            del self.pcm
            del self.bpd
            del self.uf_bpd

    def update_alpha(self,alpha):
        cdef int i
        cdef int j
        if alpha is not None:
            if isinstance(alpha,np.ndarray):

                if(alpha.size == 3):
                    self.alpha.resize(3)
                    for i in range(3):
                        self.alpha[i].resize(self.qubit_count)
                        for j in range(self.qubit_count):
                            self.alpha[i][j] = alpha[i]

                elif(alpha.size == 3*self.qubit_count):
                    self.alpha.resize(3)
                    for i in range(3):
                        self.alpha[i].resize(self.qubit_count)
                        for j in range(self.qubit_count):
                            self.alpha[i][j] = alpha[i,j]
                else:
                    raise ValueError(f"The 'alpha' input must be either a single double variable or a 3xN np.ndarray of doubles, where N is the qubit count. The current input has dimensions {alpha.shape}.")

            elif isinstance(alpha,(float,int)):
                self.alpha.resize(3)
                for i in range(3):
                    self.alpha[i].resize(self.qubit_count)
                    for j in range(self.qubit_count):
                        self.alpha[i][j] = alpha

            else:
                raise ValueError(f"The 'alpha' input must be either a single double variable or a 3xN np.ndarray of doubles, where N is the qubit count. The current input has of type {type(alpha)}.")

    @property
    def alpha(self):
        cdef int i
        cdef int j
        out = np.zeros((3,self.qubit_count))
        for i in range(3):
            for j in range(self.qubit_count):
                out[i,j] = self.alpha[i][j]
        return out

    def decode(self,syndrome = None, sx = None, sz = None):
        cdef int i
        if syndrome is not None:
            if not len(syndrome)==self.stab_count:
                raise ValueError(f"The syndrome must have length {self.stab_count}. Not {len(syndrome)}.")
            DTYPE = syndrome.dtype
            for i in range(self.stab_count): self.syndrome[i] = syndrome[i]
        elif sx is not None and sz is not None:
            DTYPE = sx.dtype
            for i in range(len(sx)):
                self.syndrome[i] = sx[i]
            for i in range(len(sz)):
                self.syndrome[i+len(sx)] = sz[i]
        else:
            raise ValueError("Invalid syndrome input.")
        
        self.bpd.decode(self.syndrome)
        
        if self.OUTPUT_TYPE == 0: #GF4 output
            out = np.zeros(self.qubit_count,dtype=DTYPE)
            for i in range(self.qubit_count): out[i] = self.bpd.decoding[i]
            return out
        else: # GF2 output
            outx = np.zeros(self.qubit_count,dtype=DTYPE)
            outz = np.zeros(self.qubit_count,dtype=DTYPE)
            for i in range(self.qubit_count):
                if self.bpd.decoding[i] == 0:
                    pass
                elif self.bpd.decoding[i] == 1:
                    outx[i] = 1
                elif self.bpd.decoding[i] == 2:
                    outx[i] = 1
                    outz[i] = 1
                elif self.bpd.decoding[i] == 3:
                    outz[i] = 1
                else:
                    raise TypeError("mbp output invalid. Check C++.")
            return outx, outz

        
    def uf_decode(self, sx = None, sz = None):
        
        cdef int i
        cdef vector[double] uf_weightsX
        cdef vector[double] uf_weightsZ
        uf_weightsX.resize(self.qubit_count)
        uf_weightsZ.resize(self.qubit_count)
        
        out = self.decode(sx=sx,sz=sz)
        if self.converge: return out

        for i in range(self.qubit_count):
            uf_weightsZ[i] = 1/(np.exp(self.bpd.log_prob_ratios[1][i])+1) + 1/(np.exp(self.bpd.log_prob_ratios[2][i])+1)
            if uf_weightsZ[i] == 0:
                uf_weightsZ[i] = np.inf
            else:
                uf_weightsZ[i] = np.log((1-uf_weightsZ[i])/uf_weightsZ[i])

            uf_weightsX[i] = 1/(np.exp(self.bpd.log_prob_ratios[1][i])+1) + 1/(np.exp(self.bpd.log_prob_ratios[0][i])+1)
            if uf_weightsX[i] == 0:
                uf_weightsX[i] = np.inf
            else:
                uf_weightsX[i] = np.log((1-uf_weightsX[i])/uf_weightsX[i])

        self.uf_bpdX.matrix_decode(sx,uf_weightsX,1)
        self.uf_bpdZ.matrix_decode(sz,uf_weightsZ,1)

        DTYPE = sx.dtype

        outx = np.zeros(self.qubit_count,dtype=DTYPE)
        outz = np.zeros(self.qubit_count,dtype=DTYPE)

        for i in range(self.qubit_count):
            outx = self.uf_bpdX.decoding[i]
            outz = self.uf_bpdZ.decoding[i]

        return outx, outz


    @property
    def log_prob_ratios(self):
        cdef int i, w
        out = np.zeros((3,self.qubit_count))
        for w in range(3):
            for i in range(self.qubit_count):
                out[w,i] = self.bpd.log_prob_ratios[w][i]
        return out
        

    @property
    def decoding(self):
        out = np.zeros(self.qubit_count).astype(int)
        for i in range(self.qubit_count): out[i] = self.bpd.decoding[i]
        return out

    @property
    def syndrome(self):
        cdef int i
        out = np.zeros(self.stab_count).astype(int)
        for i in range(self.stab_count): out[i] = self.syndrome[i]

        return out

    @property
    def error_channel(self):
        out = np.zeros((3,self.qubit_count)).astype(float)
        for i in range(3):
            for j in range(self.qubit_count):
                out[i,j]=self.bpd.channel_probs[i][j]
        return out

    @property
    def max_iter(self):
        return self.bpd.max_iter

    @property
    def alpha_parameter(self):
        return self.bpd.alpha

    @property
    def beta_parameter(self):
        return self.bpd.beta

    @property
    def iterations(self):
        return self.bpd.iterations

    @property
    def converge(self):
        return self.bpd.converge

    @property
    def iter(self):
        return self.bpd.iterations












