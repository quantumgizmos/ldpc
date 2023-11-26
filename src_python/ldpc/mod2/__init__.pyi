import numpy as np
import scipy.sparse
import ldpc.helpers.scipy_helpers
from typing import Tuple, Union
from libc.stdint cimport uintptr_t
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


class PluDecomposition():
    """
    Initialise the PLU Decomposition with a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for PLU Decomposition.
        full_reduce (bool, optional): Flag to indicate if full row reduction is required. Default is False.
        lower_triangular (bool, optional): Flag to indicate if the result should be in lower triangular form. Default is True.
    """


    def __init__(self, pcm: Union[scipy.sparse.spmatrix, np.ndarray], full_reduce: bool = False, lower_triangular: bool = True) -> None:
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

