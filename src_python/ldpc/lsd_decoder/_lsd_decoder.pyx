#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from ldpc.bposd_decoder cimport OsdMethod
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

cdef class LsdDecoder():
    """
    A class implementing the Localised Statistics Decoder (LSD) in isolation.

    The LsdDecoder class provides an interface to directly decode a syndrome using the LSD algorithm, without a preceding Belief Propagation (BP) stage. The user provides the syndrome and bit weights as inputs to produce the decoded output.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code.
    bits_per_step : int, optional
        Specifies the number of bits added to the cluster in each step of the LSD algorithm. The default value is `1`.
    lsd_order: int, optional
        The order of the LSD algorithm applied to each cluster. Must be greater than or equal to 0, by default 0.
    lsd_method: str or int, optional
        The LSD method of the LSD algorithm applied to each cluster. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'} or {0, 1, 2}. By default 'LSD_0'.

    Notes
    -----
    The LsdDecoder class leverages soft information (bit weights) provided by the user to guide the cluster growth in the LSD algorithm. The number of bits added to the cluster in each step is controlled by the `bits_per_step` parameter.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], bits_per_step: int = 1,
                  lsd_order: int = 0,
                  lsd_method: Union[str, int] = 0, **kwargs):

        # compatability with osd_method/osd_order
        if "osd_method" in kwargs:
            lsd_method = kwargs["osd_method"]
        else:
            lsd_method = lsd_method

        if "osd_order" in kwargs:
            lsd_order = kwargs["osd_order"]
        else:
            lsd_order = lsd_order
        if(lsd_order < 0):
            raise ValueError(f"lsd_order must be greater than or equal to 0. Not {lsd_order}.")

        if isinstance(lsd_method, str):
            if lsd_method.lower() not in ['osd_0', 'osd_e', 'osd_cs', 'osde', 'osdcs', 'osd0', 'lsd_0', 'lsd_e','lsd_cs','lsd0','lsdcs','lsde']:
                raise ValueError(f"lsd_method must be one of 'LSD_0', 'LSD_E', 'LSD_CS'. Not {lsd_method}.")
        elif isinstance(lsd_method, int):
            if lsd_method not in [0, 1, 2]:
                raise ValueError(f"lsd_method must be one of 0, 1, 2. Not {lsd_method}.")
        else:
            raise ValueError(f"lsd_method must be one of 'LSD_0' (0), 'LSD_E' (1), 'LSD_CS' (2). Not {lsd_method}.")


        self.MEMORY_ALLOCATED = False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)

        ## set vector lengths
        self._syndrome.resize(self.pcm.m)
        self._bit_weights.resize(self.pcm.n)
        self.decoding.resize(self.pcm.n)

        self.lsd = new LsdDecoderCpp(pcm=self.pcm[0], lsd_method=OsdMethod.OSD_0, lsd_order=lsd_order)
        self.lsd_method = lsd_method

        if bits_per_step == 0:
            self.bits_per_step = pcm.shape[1]
        else:
            self.bits_per_step = bits_per_step
        self.MEMORY_ALLOCATED=True

    def __dealloc__(self):
        if self.MEMORY_ALLOCATED:
            del self.lsd
            del self.pcm

    def decode(self,syndrome,bit_weights):
        """
        Decodes the input syndrome using the LSD algorithm in isolation.

        This method directly invokes the LSD decoding routine without attempting any BP decoding first.
        The provided bit weights are used as input for the LSD decoder.

        Parameters:
            syndrome : np.ndarray
                A 1D numpy array (dtype=np.uint8) representing the syndrome. Its length must equal the number of rows in the parity check matrix.
            bit_weights : list or np.ndarray
                A list or 1D numpy array of doubles, with length equal to the number of columns in the parity check matrix.

        Returns:
            np.ndarray
                A 1D numpy array (dtype=np.uint8) containing the decoded output.

        Raises:
            ValueError: If the length of syndrome or bit_weights does not match the expected dimensions.
        """

        if not len(bit_weights)==self.pcm.n:
            raise ValueError(f"The bit weights must have length {self.pcm.n}. Not {len(bit_weights)}.")

        if not len(syndrome)==self.pcm.m:
            raise ValueError(f"The syndrome must have length {self.pcm.m}. Not {len(syndrome)}.")
        cdef int i
        DTYPE = syndrome.dtype
        
        for i in range(self.pcm.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            self.bpd.converge = True
            return np.zeros(self.pcm.n,dtype=DTYPE)

        for i in range(self.pcm.n):
            self._bit_weights[i] = bit_weights[i]

        self.lsd.decoding = self.lsd.lsd_decode(self._syndrome, self._bit_weights,self.bits_per_step, True)

        out = np.zeros(self.pcm.n,dtype=DTYPE)
        for i in range(self.pcm.n):
            out[i] = self.lsd.decoding[i]
        
        return out

    @property
    def lsd_method(self) -> Optional[str]:
        """
        The Localized Statistic Decoding (LSD) method used.

        Returns
        -------
        Optional[str]
            A string representing the LSD method used. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}. If no LSD method
            has been set, returns `None`.
        """
        if self.lsd.lsd_method == OsdMethod.OSD_0:
            return 'lSD_0'
        elif self.lsd.lsd_method == OsdMethod.EXHAUSTIVE:
            return 'LSD_E'
        elif self.lsd.lsd_method == OsdMethod.COMBINATION_SWEEP:
            return 'LSD_CS'
        elif self.lsd.lsd_method == OsdMethod.OSD_OFF:
            return 'LSD_OFF'
        else:
            return None

    @lsd_method.setter
    def lsd_method(self, method: Union[str, int, float]) -> None:
        """
        Sets the LSD method used. That is, the OSD method per cluster.

        Parameters
        ----------
        method : Union[str, int, float]
            A string, integer or float representing the OSD method to use. Must be one of {'LSD_0', 'LSD_E', 'LSD_CS'}, corresponding to
            LSD order-0, LSD Exhaustive or LSD-Cominbation-Sweep.
        """
        # OSD method
        if str(method).lower() in ['osd_0', '0', 'osd0', 'lsd_0', 'lsd0']:
            self.lsd.lsd_method = OsdMethod.OSD_0
            self.lsd.lsd_order = 0
        elif str(method).lower() in ['osd_e', 'e', 'exhaustive', 'lsd_e', 'lsde']:
            self.lsd.lsd_method = OsdMethod.EXHAUSTIVE
        elif str(method).lower() in ['osd_cs', '1', 'cs', 'combination_sweep', 'lsd_cs']:
            self.lsd.lsd_method = OsdMethod.COMBINATION_SWEEP
        elif str(method).lower() in ['off', 'osd_off', 'deactivated', -1, 'lsd_off']:
            self.lsd.lsd_method = OsdMethod.OSD_OFF
        else:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:\
                'LSD_0', 'LSD_E' or 'LSD_CS'.")


    @property
    def lsd_order(self) -> int:
        """
        The LSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """
        return self.lsd.lsd_order


    @lsd_order.setter
    def lsd_order(self, order: int) -> None:
        """
        Set the order for the LSD method.

        Parameters
        ----------
        order : int
            The order for the OSD method. Must be a positive integer.

        Raises
        ------
        ValueError
            If order is less than 0.

        Warns
        -----
        UserWarning
            If the LSD method is 'OSD_E' and the order is greater than 15.

        """
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")

        if self.lsd.lsd_method == OsdMethod.OSD_0 and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")

        if self.lsd.lsd_method == OsdMethod.EXHAUSTIVE and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not "
                        "recommended. Use the 'osd_cs' method instead.")

        self.lsd.lsd_order = order

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.pcm.n).astype(int)
        for i in range(self.pcm.n):
            out[i] = self.bpd.decoding[i]
        return out