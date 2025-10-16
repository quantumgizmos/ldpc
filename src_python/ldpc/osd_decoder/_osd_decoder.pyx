#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import warnings
from scipy.sparse import spmatrix
from typing import Union, List, Optional
import scipy.sparse
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


cdef class OsdDecoder():
    """
    Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    This class provides an implementation of Ordered Statistic Decoding (OSD) for binary linear codes.
    OSD is a post-processing technique that can be used after belief propagation (BP) decoding to 
    improve decoding performance by performing an exhaustive search over the most reliable bit positions.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix for the code. Must be a binary matrix.
    osd_method : Union[str, int, float], optional
        The OSD method to use. Must be one of:
        - 'OSD_0' or 0: OSD order-0 (fastest, least powerful)
        - 'OSD_E' or 'exhaustive': Exhaustive search (most powerful, slowest)
        - 'OSD_CS' or 'combination_sweep': Combination sweep method (balanced performance)
        Default is 0 (OSD_0).
    osd_order : int, optional
        The order of the OSD algorithm, which determines the number of least reliable
        bits to consider for higher-order osd. Must be a non-negative integer.
        For OSD_0, this must be 0. For OSD_E, values > 15 are not recommended.
        Default is 0.
    channel_probabilities : np.ndarray, optional
        Channel error probabilities for the intial error distribution. Used in the weighted sum
        during the higher-order OSD search. Must have length equal to the number of columns.

    Notes
    -----
    This class uses the C++ implementation `ldpc::osd::OsdDecoderCpp` for efficient
    OSD decoding. The decoder requires log probability ratios from a previous BP
    decoding attempt to determine the reliability ordering of bits.

    The OSD algorithm works by:
    1. Ordering bits by their reliability (from log probability ratios)
    2. Selecting the most reliable independent set of bits
    3. Performing Gaussian elimination to solve for the remaining bits
    4. Optionally searching over error patterns in the least reliable bits
    """

    def __cinit__(self, pcm: Union[np.ndarray, spmatrix],  osd_method: Union[str, int, float] = 0,
                 osd_order: int = 0, channel_probabilities: np.ndarray = None):
        
        self.MEMORY_ALLOCATED=False


        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)

        self.m = self.pcm.m
        self.n = self.pcm.n

        self._channel_probabilities.resize(pcm.shape[1])
        cdef int i
        for i in range(pcm.shape[1]):
            self._channel_probabilities[i] = channel_probabilities[i]

        ## set vector lengths
        self._syndrome.resize(self.pcm.m)
        self._bit_weights.resize(self.pcm.n)
        self.decoding.resize(self.pcm.n)

        ## set up OSD with default values and channel probs from BP
        self.osdD = new OsdDecoderCpp(self.pcm[0], OSD_OFF, 0, self._channel_probabilities)
        self.osd_method=osd_method
        self.osd_order=osd_order

        self.osdD.osd_setup()

        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.osdD

    def decode(self, syndrome: np.ndarray, bit_weights: np.ndarray) -> np.ndarray:
        """
        Decodes the input syndrome using the belief propagation and OSD decoding methods.

        This method takes an input syndrome and decodes ins using OSD.

        Parameters
        ----------
        syndrome : np.ndarray
            The input syndrome to decode.

        bit_weights : np.ndarray
            The bit weights (reliabilities) used for OSD decoding. The OSD columns are ordered by this weighting from smallest to largest.

        Returns
        -------
        np.ndarray
            A numpy array containing the decoded output.

        Raises
        ------
        ValueError
            If the length of the input syndrome is not equal to the length of the code.

        Notes
        -----
        This method first checks if the input syndrome is all zeros. If it is, it returns an array of zeros of the same
        length as the codeword. The OSD method used is specified by the `osd_method` parameter passed to the class
        constructor. The OSD order used is specified by the `osd_order` parameter passed to the class constructor.

        """

        cdef int i

        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")

        if not len(bit_weights) == self.n:
            raise ValueError(f"The bit_weights must have length {self.n}. Not {len(bit_weights)}.")
        
        zero_syndrome = True
        
        for i in range(self.m):
            self._syndrome[i] = syndrome[i]
            if self._syndrome[i]:
                zero_syndrome = False
        if zero_syndrome:
            return np.zeros(self.n, dtype=syndrome.dtype)

        for i in range(self.n):
            self._bit_weights[i] = bit_weights[i]
        
        self.osdD.decode(self._syndrome, self._bit_weights)

        out = np.zeros(self.n, dtype=syndrome.dtype)
        for i in range(self.n):
            out[i] = self.osdD.osdw_decoding[i]

        return out

    @property
    def osd_method(self) -> Optional[str]:
        """
        The Ordered Statistic Decoding (OSD) method used.

        Returns
        -------
        Optional[str]
            A string representing the OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}. If no OSD method
            has been set, returns `None`.
        """
        if self.osdD.osd_method == OSD_0:
            return 'OSD_0'
        elif self.osdD.osd_method == EXHAUSTIVE:
            return 'OSD_E'
        elif self.osdD.osd_method == COMBINATION_SWEEP:
            return 'OSD_CS'
        elif self.osdD.osd_method == OSD_OFF:
            return 'OSD_OFF'
        else:
            return None

    @osd_method.setter
    def osd_method(self, method: Union[str, int, float]) -> None:
        """
        Sets the OSD method used.

        Parameters
        ----------
        method : Union[str, int, float]
            A string, integer or float representing the OSD method to use. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}, corresponding to
            OSD order-0, OSD Exhaustive or OSD-Cominbation-Sweep.
        """
        # OSD method
        if str(method).lower() in ['osd_0', '0', 'osd0']:
            self.osdD.osd_method = OSD_0
            self.osdD.osd_order = 0
        elif str(method).lower() in ['osd_e', 'e', 'exhaustive']:
            self.osdD.osd_method = EXHAUSTIVE
        elif str(method).lower() in ['osd_cs', '1', 'cs', 'combination_sweep']:
            self.osdD.osd_method = COMBINATION_SWEEP
        elif str(method).lower() in ['off', 'osd_off', 'deactivated', -1]:
            self.osdD.osd_method = OSD_OFF
        else:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:\
                'OSD_0', 'OSD_E' or 'OSD_CS'.")


    @property
    def osd_order(self) -> int:
        """
        The OSD order used.

        Returns
        -------
        int
            An integer representing the OSD order used.
        """
        return self.osdD.osd_order


    @osd_order.setter
    def osd_order(self, order: int) -> None:
        """
        Set the order for the OSD method.

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
            If the OSD method is 'OSD_E' and the order is greater than 15.

        """
        # OSD order
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")

        if self.osdD.osd_method == OSD_0 and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")

        if self.osdD.osd_method == EXHAUSTIVE and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not "
                        "recommended. Use the 'osd_cs' method instead.")

        self.osdD.osd_order = order

    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.osD.osdw_decoding[i]
        return out

    @property
    def osd0_decoding(self) -> np.ndarray:
        """
        Returns the current OSD-0 decoding output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)

        for i in range(self.n):
            out[i] = self.osdD.osd0_decoding[i]
        return out

    @property
    def osdw_decoding(self) -> np.ndarray:
        """
        Returns the current OSD-W decoding output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)

        for i in range(self.n):
            out[i] = self.osdD.osdw_decoding[i]
        return out

