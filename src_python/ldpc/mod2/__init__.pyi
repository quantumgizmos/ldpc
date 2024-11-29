import numpy as np
import scipy.sparse
import ldpc.mod2._legacy_v1
import ldpc.helpers.scipy_helpers
from typing import Tuple, Union, List
from libc.stdint import uintptr_t

def csc_to_scipy_sparse(vector[vector[int]]& col_adjacency_list):
    """
    Converts CSC matrix to sparse matrix
    """

def rank(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method: str = "dense") -> int:
    """
    Calculate the rank of a given parity check matrix.

    This function calculates the rank of the parity check matrix (pcm) using either a dense or sparse method. 
    The dense method is used by default.

    Parameters
    ----------
    pcm : Union[scipy.sparse.spmatrix, np.ndarray]
        The parity check matrix to be ranked.
    method : str, optional
        The method to use for calculating the rank. Options are "dense" or "sparse". Defaults to "dense".

    Returns
    -------
    int
        The rank of the parity check matrix.
    """

def nullspace(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method = "dense") -> scipy.sparse.spmatrix:
    """
    Calculate the kernel of a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for kernel calculation.
        
    Returns:
        scipy.sparse.spmatrix: The kernel of the parity check matrix.
    """

def kernel(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method = "dense") -> scipy.sparse.spmatrix:
    """
    Calculate the kernel of a given parity check matrix (same as the nullspace).
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for kernel calculation.
        
    Returns:
        scipy.sparse.spmatrix: The kernel of the parity check matrix.
    """

def row_complement_basis(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """
    Calculate the row complement basis of a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for row complement basis calculation.
        
    Returns:
        scipy.sparse.spmatrix: The row complement basis of the parity check matrix.
    """

def pivot_rows(mat: Union[np.ndarray,scipy.sparse.spmatrix]):
    """
    Find the pivot rows of a given matrix.

    This function finds the pivot rows of the input matrix. The input matrix can be either a dense numpy array or a sparse scipy matrix.
    The function first converts the input matrix to a CSR list, then calls the C++ function `pivot_rows_cpp` to find the pivot rows.

    Parameters
    ----------
    mat : Union[np.ndarray, scipy.sparse.spmatrix]
        The input matrix.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the pivot rows of the input matrix.
    """

def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    """
    Test function
    """

def estimate_code_distance(pcm: Union[scipy.sparse.spmatrix,np.ndarray], timeout_seconds: float = 0.025, number_of_words_to_save = 10):

    """
    Estimate the code distance of a binary matrix representing a parity-check matrix.

    This function estimates the minimum distance of a code defined by the given parity-check matrix (PCM).
    The calculation runs until either the specified timeout is reached or an estimate is found.

    Parameters
    ----------
    pcm : Union[scipy.sparse.spmatrix, np.ndarray]
        The parity-check matrix representing the code, provided as a scipy sparse matrix or a numpy ndarray.

    timeout_seconds : float, optional (default=0.025)
        The maximum time in seconds allowed for estimating the code distance.

    number_of_words_to_save : int, optional (default=10)
        The number of minimum-weight codewords to save in the returned matrix.

    Returns
    -------
    min_distance : int
        The estimated minimum distance of the code.

    samples_searched : int
        The number of samples that were searched to find the minimum distance.

    min_weight_words_matrix : scipy.sparse.csr_matrix
        A sparse matrix containing the minimum-weight codewords found, up to `number_of_words_to_save`.
    """

def row_span(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) -> scipy.sparse.spmatrix:
    """
    Compute the row span of a given parity check matrix.

    This function computes the row span of the input parity check matrix (pcm). The input matrix can be either a dense numpy array or a sparse scipy matrix.
    The function first converts the input matrix to a CSR list, then calls the C++ function `row_span_cpp` to compute the row span.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The input parity check matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        The row span of the input matrix.
    """

def compute_exact_code_distance(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):

    """
    Compute the exact code distance of a binary matrix representing a parity-check matrix.

    This function computes the exact minimum distance of a code defined by the given parity-check matrix (PCM).
    Unlike the estimation function, this function guarantees the precise minimum distance, though it may be computationally intensive.

    Parameters
    ----------
    pcm : Union[scipy.sparse.spmatrix, np.ndarray]
        The parity-check matrix representing the code, provided as a scipy sparse matrix or a numpy ndarray.

    Returns
    -------
    distance : int
        The exact minimum distance of the code.
    """

def row_basis(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) -> scipy.sparse.spmatrix:
    """
    Compute the row basis of a given parity check matrix.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The input parity check matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        The row basis of the input matrix.
    """

def row_echelon(matrix: Union[np.ndarray,scipy.sparse.spmatrix], full: bool = False) -> List[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarry or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to `True', Gaussian elimination is only performed on the rows below
        the pivot. If set to `False' Gaussian eliminatin is performed on rows above
        and below the pivot. 

    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    Examples
    --------
    >>> H=np.array([[1, 1, 1],[1, 1, 1],[0, 1, 0]])
    >>> re_matrix=row_echelon(H)[0]
    >>> print(re_matrix)
    [[1 1 1]
        [0 1 0]
        [0 0 0]]

    >>> re_matrix=row_echelon(H,full=True)[0]
    >>> print(re_matrix)
    [[1 0 1]
        [0 1 0]
        [0 0 0]]

    """

def reduced_row_echelon(matrix: Union[np.ndarray, scipy.sparse.spmatrix]) -> List[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Converts matrix to reduced row echelon form such that the output has
    the form rre=[I,A]

    Parameters
    ----------
    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format

    Returns
    -------
    reduced_row_echelon_from: numpy.ndarray
        The reduced row echelon form of the inputted matrix in the form rre=[I,A]
    matrix_rank: int
        The binary rank of the matrix
    transform_matrix_rows: numpy.ndarray
        The transformation matrix for row permutations
    transform_matrix_cols: numpy.ndarray
        The transformation matrix for the columns

    Examples
    --------
    >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    >>> rre=reduced_row_echelon(H)[0]
    >>> print(rre)
    [[1 0 0 1 1 0 1]
     [0 1 0 1 0 1 1]
     [0 0 1 0 1 1 1]]

    """


def inverse(matrix: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """
    Computes the left inverse of a full-rank matrix.

    Notes
    -----

    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows::

        Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form::

        row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to::

        Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P

    Parameters
    ----------
    matrix: numpy.ndarray
        The binary matrix to be inverted in numpy.ndarray format. This matrix must either be
        square full-rank or rectangular with full-column rank.

    Returns
    -------
    numpy.ndarray
        The inverted binary matrix


    Examples
    --------

    >>> # full-rank square matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> # full-column rank matrix
    >>> mat=np.array([[1,1,0],[0,1,0],[0,0,1],[0,1,1]])
    >>> i_mat=inverse(mat)
    >>> print(i_mat@mat%2)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    """


class PluDecomposition():
    """
    Initialise the PLU Decomposition with a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for PLU Decomposition.
        full_reduce (bool, optional): Flag to indicate if full row reduction is required. Default is False.
        lower_triangular (bool, optional): Flag to indicate if the result should be in lower triangular form. Default is True.
    """


    def __init__(self, pcm: Union[scipy.sparse.spmatrix, np.ndarray], full_reduce: bool = False, lower_triangular: bool = True) -> None: ...

    def __cinit__(self, pcm: Union[scipy.sparse.spmatrix,np.ndarray], full_reduce: bool = False, lower_triangular: bool = True): ...


    def lu_solve(self, y: np.ndarray) -> np.ndarray:
        """
        Solve the LU decomposition problem for a given array 'y'.
        
        Parameters:
            y (np.ndarray): Array to be solved.
            
        Returns:
            np.ndarray: Solution array.
        """

    @property
    def L(self) -> scipy.sparse.spmatrix:
        """
        Get the lower triangular matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Lower triangular matrix.
        """

    @property
    def U(self) -> scipy.sparse.spmatrix:
        """
        Get the upper triangular matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Upper triangular matrix.
        """

    @property
    def P(self) -> scipy.sparse.spmatrix:
        """
        Get the permutation matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Permutation matrix.
        """

    @property
    def rank(self) -> int:
        """
        Get the rank of the matrix used for the LU decomposition.
        
        Returns:
            int: Rank of the matrix.
        """

    @property
    def pivots(self) -> np.ndarray:
        """
        Get the pivot positions used during the LU decomposition.
        
        Returns:
            np.ndarray: Array of pivot positions.
        """

    def __del__(self): ...
