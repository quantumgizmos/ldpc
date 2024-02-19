#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
from scipy.sparse import spmatrix


cdef BpSparse* Py2BpSparse(pcm):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input\
        a np.ndarray or spmatrix object, not {type(pcm)}")

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef BpSparse* cpcm = new BpSparse(m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.insert_entry(i,j)
    elif isinstance(pcm,spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.insert_entry(rows[i], cols[i])
    
    return cpcm

cdef class UnionFindDecoder:
    """
    A decoder class that implements the Union Find Decoder (UFD) algorithm to decode binary linear codes. 
    The decoder operates on a provided parity-check matrix (PCM) and can function with or without soft information 
    from a channel. The UFD algorithm can be run in two modes: matrix solve and peeling, controlled by the 
    `uf_method` flag. 

    Parameters
    ----------
    pcm : Union[np.ndarray, spmatrix]
        The parity-check matrix (PCM) of the code. This should be either a dense matrix (numpy ndarray) 
        or a sparse matrix (scipy sparse matrix).
    uf_method : bool, optional
        If True, the decoder operates in matrix solve mode. If False, it operates in peeling mode. 
        Default is False.
    """ 
 
    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], uf_method: str = False):
        
        self.MEMORY_ALLOCATED=False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)
 
        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        self.ufd = new uf_decoder_cpp(self.pcm[0])
        self._syndrome.resize(self.m) #C vector for the syndrome
        self.uf_llrs.resize(self.n) #C vector for the log-likehood ratios
        self.uf_method = uf_method
        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.ufd

    def decode(self, syndrome: np.ndarray, llrs: np.ndarray = None, bits_per_step: int = 0) -> np.ndarray:
        """
        Decodes the given syndrome to find an estimate of the transmitted codeword.

        Parameters
        ----------
        syndrome : np.ndarray
            The syndrome to be decoded.
        llrs : np.ndarray, optional
            Log-likelihood ratios (LLRs) of the received bits. If provided, these are used to guide 
            the decoding process. Default is None.
        bits_per_step : int, optional
            The number of bits to be added to clusters in each step of the decoding process. 
            If 0, all neigbouring bits are added in one step. Default is 0.

        Returns
        -------
        np.ndarray
            The estimated codeword.

        Raises
        ------
        ValueError
            If the length of the syndrome or the length of the llrs (if provided) do not match the dimensions 
            of the parity-check matrix.
        """

        if not len(syndrome)==self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")

        if llrs is not None:
            if not len(llrs) == self.n:
                raise ValueError(f"The llrs must have length {self.n}. Not {len(llrs)}.")
        
        
        cdef int i
        DTYPE = syndrome.dtype
        
        cdef zero_syndrome = True
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            return np.zeros(self.n,dtype=DTYPE)


        if llrs is not None:
            for i in range(self.n):
                self.uf_llrs[i] = llrs[i] 


        if bits_per_step == 0:
            self.bits_per_step = self.n
        else:
            self.bits_per_step = bits_per_step

        if self.uf_method:
            if llrs is not None:
                self.ufd.decoding = self.ufd.matrix_decode(self._syndrome, self.uf_llrs,self.bits_per_step)
            else:
                self.ufd.decoding = self.ufd.matrix_decode(self._syndrome, EMPTY_DOUBLE_VECTOR,self.bits_per_step)

        else:
            if llrs is not None:
              
                self.ufd.decoding = self.ufd.peel_decode(self._syndrome, self.uf_llrs,self.bits_per_step)
            else:
                self.ufd.decoding = self.ufd.peel_decode(self._syndrome, EMPTY_DOUBLE_VECTOR,self.bits_per_step)
        

        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n):
            out[i] = self.ufd.decoding[i]
        
        return out

    @property
    def decoding(self):

        out = np.zeros(self.n,dtype=np.uint8)
        for i in range(self.n):
            out[i] = self.ufd.decoding[i]
