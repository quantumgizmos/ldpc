import numpy as np
from scipy.sparse import spmatrix

class UnionFindDecoder:
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
 
    def __cinit__(self, pcm: Union[np.ndarray, spmatrix], uf_method: str = False): ...

    def __del__(self): ...

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

    @property
    def decoding(self): ...
