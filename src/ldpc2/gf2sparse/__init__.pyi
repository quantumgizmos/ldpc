from typing import Tuple
import numpy as np
import scipy.sparse

cdef class gf2sparse:
    """
    A class representing a binary (GF(2)) sparse matrix with basic linear algebra operations.

    Parameters
    ----------
    pcm : Tuple[np.ndarray,scipy.sparse.csr_matrix], optional
        A binary parity check matrix in the form of a tuple containing a numpy.ndarray and a
        scipy.sparse.csr_matrix. Default is None.
    empty : bool, optional
        If True, an empty instance of the class is created. If False, an input parity check matrix
        must be provided. Default is False.

    Attributes
    ----------
    pcm : cygf2_sparse
        A Cython wrapper for a binary sparse matrix.
    m : int
        Number of rows of the parity check matrix.
    n : int
        Number of columns of the parity check matrix.
    PCM_ALLOCATED : bool
        A flag indicating if the PCM memory has been allocated.

    Methods
    -------
    __init__(self, pcm=None, empty=False)
        Initializes an instance of the class.
    toarray(self, type="0")
        Returns the parity check matrix in a numpy.ndarray or a scipy.sparse.csr_matrix format.
    to_numpy(self)
        Returns the parity check matrix in a numpy.ndarray format.
    to_scipy_sparse(self)
        Returns the parity check matrix in a scipy.sparse.csr_matrix format.
    lu_decomposition(self, reset_cols=True, full_reduce=False)
        Computes the LU decomposition of the parity check matrix.
    __repr__(self)
        Returns a string representation of the parity check matrix.
    lu_solve(self, y)
        Solves a linear system of equations using the LU decomposition.
    kernel(self)
        Computes the kernel (nullspace) of the parity check matrix.
    transpose(self)
        Computes the transpose of the parity check matrix.
    T
        Property that returns the transpose of the parity check matrix.
    rank
        Property that returns the rank of the parity check matrix.
    rows
        Property that returns the rows of the parity check matrix.
    cols
        Property that returns the columns of the parity check matrix.
    inv_rows
        Property that returns the inverse rows of the parity check matrix.
    inv_cols
        Property that returns the inverse columns of the parity check matrix.
    L
        Property that returns the lower triangular factor of the LU decomposition.
    U
        Property that returns the upper triangular factor of the LU decomposition.
    """

    def __init__(self, pcm: Tuple[np.ndarray,scipy.sparse.csr_matrix] = None, empty: bool = False):
        """
        Initializes an instance of the class.

        Parameters
        ----------
        pcm : Tuple[np.ndarray,scipy.sparse.csr_matrix], optional
            A binary parity check matrix in the form of a tuple containing a numpy.ndarray and a
            scipy.sparse.csr_matrix. Default is None.
        empty : bool, optional
            If True, an empty instance of the class is created. If False, an input parity check matrix
            must be provided. Default is False.
        """