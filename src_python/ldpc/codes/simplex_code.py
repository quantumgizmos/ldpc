import numpy as np
from ldpc.codes import hamming_code
from ldpc.mod2 import kernel
import scipy.sparse

def simplex_code(m: int) -> scipy.sparse.csr_matrix:
    """
    Outputs the parity check matrix of a binary simplex code.

    Simplex codes are a family of binary linear codes that are duals of Hamming codes.
    For a given integer m , the binary simplex code has parameters [ 2^m - 1 , m , 2^m - 1 ] .
    These codes are notable for their large minimum distance and large autormorphism groups that are useful for compiling logic in quantum stabiliser codes.

    Parameters
    ----------
        m (int): Dimension of the simplex code.

    Returns
    ----------
        H_simplex (np.ndarray): Parity check matrix of shape (2^m - 1 - m, 2^m - 1).
    
    Raises
    ------
    TypeError
        If the input variable 'dimension' is not of type 'int'.
    """

    # Implement function here
    if not isinstance(m, int):
        raise TypeError("The input variable 'rank' must be of type 'int'.")
    H_hamming = hamming_code(m)

    H_simplex = kernel(H_hamming)


    return H_simplex