#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
import ldpc.mod2._legacy_v1
import ldpc.helpers.scipy_helpers
from typing import Tuple, Union, List
from libc.stdint cimport uintptr_t

cdef void print_sparse_matrix(GF2Sparse& mat):

    cdef int m = mat.m
    cdef int n = mat.n

    cdef int i
    cdef int j

    out = np.zeros((m,n)).astype(np.uint8)

    cdef GF2Entry e

    for i in range(m):
        for j in range(n):
            e = mat.get_entry(i,j)
            if not e.at_end():
                out[i,j] = 1

    print(out)

cdef GF2Sparse* Py2GF2Sparse(pcm):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

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
    cdef GF2Sparse* cpcm = new GF2Sparse(m,n) #creates the C++ sparse matrix object

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
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix.spmatrix object, not {type(pcm)}")
    
    return cpcm


cdef coords_to_scipy_sparse(vector[vector[int]]& entries, int m, int n, int entry_count):

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef csr_to_scipy_sparse(vector[vector[int]]& row_adjacency_list, int m, int n, int entry_count = -9999):
    """
    Converts CSR matrix to sparse matrix
    """
    cdef int i
    cdef int j

    if entry_count == -9999:
        entry_count = 0
        for i in range(m):
                entry_count += row_adjacency_list[i].size()

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    cdef int row_n = 0
    cdef entry_i = 0
    for i in range(m):
        row_n = row_adjacency_list[i].size()
        for j in range(row_n):
            rows[entry_i] = i
            cols[entry_i] = row_adjacency_list[i][j]
            entry_i += 1

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

def csc_to_scipy_sparse(vector[vector[int]]& col_adjacency_list):
    """
    Converts CSC matrix to sparse matrix
    """
    # Determine column count
    cdef int n = col_adjacency_list.size()

    # Determine row count and entry count
    cdef int m = 0
    cdef int entry_count = 0
    cdef int col_n = 0
    for j in range(n):
        col_n = col_adjacency_list[j].size()
        entry_count += col_n
        for i in range(col_n):
            if col_adjacency_list[j][i] >= m:
                m = col_adjacency_list[j][i] + 1

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    cdef int entry_i = 0
    for j in range(n):
        col_n = col_adjacency_list[j].size()
        for i in range(col_n):
            rows[entry_i] = col_adjacency_list[j][i]
            cols[entry_i] = j
            entry_i += 1

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef GF2Sparse2Py(GF2Sparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat

cdef vector[vector[int]] Py2CscList(pcm: Union[scipy.sparse.spmatrix, np.ndarray]):

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)
    
    cdef int rows
    cdef int cols
    cdef vector[vector[int]] csc_list
    cdef int[:] indices
    cdef int[:] indptr
    # cdef int[:] data
    cdef int col, row, idx

    csc = scipy.sparse.csc_matrix(pcm)

    # Get the dimensions of the matrix
    rows = csc.shape[0]
    cols = csc.shape[1]

    # Get the data, indices, and indptr from the CSC matrix
    indices = csc.indices
    indptr = csc.indptr
    # data = csc.data

    # Fill the vector of vectors with the data from the CSC matrix
    
    for col in range(cols):
        csc_list.push_back(vector[int]())
        for idx in range(indptr[col], indptr[col + 1]):
            csc_list[col].push_back(indices[idx])

    return csc_list

cdef vector[vector[int]] Py2CsrList(pcm: Union[scipy.sparse.spmatrix, np.ndarray]):

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    cdef int rows
    cdef int cols
    cdef vector[vector[int]] csr_list
    cdef int[:] indices
    cdef int[:] indptr
    cdef int row, col, idx

    csr = scipy.sparse.csr_matrix(pcm)  # Convert to CSR format

    # Get the dimensions of the matrix
    rows = csr.shape[0]
    cols = csr.shape[1]

    # Get the indices and indptr from the CSR matrix
    indices = csr.indices
    indptr = csr.indptr

    # Fill the vector of vectors with the data from the CSR matrix
    for row in range(rows):
        csr_list.push_back(vector[int]())
        for idx in range(indptr[row], indptr[row + 1]):
            csr_list[row].push_back(indices[idx])

    return csr_list

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

    cdef vector[vector[int]] pcm_csc
    cdef GF2Sparse* cpcm
    cdef RowReduce* rr
    cdef int rank
    
    if method == "dense":
        pcm_csc = Py2CscList(pcm)
        return rank_cpp(pcm.shape[0], pcm.shape[1], pcm_csc)

    elif method == "sparse":

        cpcm = Py2GF2Sparse(pcm)
        rr = new RowReduce(cpcm[0])
        rr.rref(False,False)
        rank = rr.rank
        del rr
        del cpcm
        return rank
    
    else:
        raise ValueError(f"Invalid method. Please use 'dense' or 'sparse', not {method}")

def nullspace(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method = "dense") -> scipy.sparse.spmatrix:
    """
    Calculate the kernel of a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for kernel calculation.
        
    Returns:
        scipy.sparse.spmatrix: The kernel of the parity check matrix.
    """

    cdef vector[vector[int]] csr_list
    cdef vector[vector[int]] csr_ker
    cdef GF2Sparse* cpcm
    cdef CsrMatrix csrf

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # get the parity check dimensions
    row_count, col_count = pcm.shape[0], pcm.shape[1]

    if method == "dense":

        csr_list = Py2CsrList(pcm)
        csr_ker = gf2dense_kernel(row_count, col_count, csr_list)
        return csc_to_scipy_sparse(csr_ker)

    elif method == "sparse":
        cpcm = Py2GF2Sparse(pcm)
        csr = cy_kernel(cpcm)
        del cpcm
        return csr_to_scipy_sparse(csr.row_adjacency_list, csr.m, csr.n, csr.entry_count)

    else:
        raise ValueError(f"Invalid method. Please use 'dense' or 'sparse'")

def kernel(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method = "dense") -> scipy.sparse.spmatrix:
    """
    Calculate the kernel of a given parity check matrix (same as the nullspace).
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for kernel calculation.
        
    Returns:
        scipy.sparse.spmatrix: The kernel of the parity check matrix.
    """
    return nullspace(pcm, method)

def row_complement_basis(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """
    Calculate the row complement basis of a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for row complement basis calculation.
        
    Returns:
        scipy.sparse.spmatrix: The row complement basis of the parity check matrix.
    """
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    cdef CsrMatrix csr = cy_row_complement_basis(cpcm)
    del cpcm
    return csr_to_scipy_sparse(csr.row_adjacency_list, csr.m, csr.n, csr.entry_count)

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
    cdef int i
    cdef vector[vector[int]] mat_csr = Py2CsrList(mat)
    cdef vector[int] pivots = pivot_rows_cpp(mat.shape[0], mat.shape[1], mat_csr)
    out = np.zeros(pivots.size()).astype(int)
    for i in range(pivots.size()):
        out[i] = pivots[i]
    return out

def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    """
    Test function
    """
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    output = GF2Sparse2Py(cpcm)
    del cpcm
    return output

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
  
    cdef int row_count = pcm.shape[0]
    cdef int col_count = pcm.shape[1]
    cdef vector[vector[int]] csr_list = Py2CsrList(pcm)

    cdef DistanceStruct dist_struct = estimate_code_distance_cpp(row_count, col_count, csr_list, timeout_seconds, 10)

    cdef int min_distance = dist_struct.min_distance
    cdef int samples_searched = dist_struct.samples_searched
    cdef vector[vector[int]] min_weight_words = dist_struct.min_weight_words

    min_weight_words_matrix = csr_to_scipy_sparse(min_weight_words, number_of_words_to_save, col_count)

    return min_distance, samples_searched, min_weight_words_matrix

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

    cdef int row_count = pcm.shape[0]
    cdef int col_count = pcm.shape[1]
    cdef vector[vector[int]] csr_list = Py2CsrList(pcm)

    cdef vector[vector[int]] rs = row_span_cpp(row_count, col_count, csr_list)

    return csr_to_scipy_sparse(rs, int(2**row_count), col_count)

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
 
    cdef int row_count = pcm.shape[0]
    cdef int col_count = pcm.shape[1]
    cdef vector[vector[int]] csr_list = Py2CsrList(pcm)

    cdef int distance = compute_exact_code_distance_cpp(row_count, col_count, csr_list)

    return distance

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

    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    pivots = pivot_rows(pcm)

    return pcm[pivots,:]

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
    #convert to numpy
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    
    return ldpc.mod2._legacy_v1.row_echelon(matrix, full)

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
    #convert to numpy
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    
    return ldpc.mod2._legacy_v1.reduced_row_echelon(matrix)


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
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    
    return ldpc.mod2._legacy_v1.inverse(matrix)


cdef class PluDecomposition():
    """
    Initialise the PLU Decomposition with a given parity check matrix.
    
    Parameters:
        pcm (Union[scipy.sparse.spmatrix, np.ndarray]): The parity check matrix for PLU Decomposition.
        full_reduce (bool, optional): Flag to indicate if full row reduction is required. Default is False.
        lower_triangular (bool, optional): Flag to indicate if the result should be in lower triangular form. Default is True.
    """


    def __init__(self, pcm: Union[scipy.sparse.spmatrix, np.ndarray], full_reduce: bool = False, lower_triangular: bool = True) -> None:
        pass

    def __cinit__(self, pcm: Union[scipy.sparse.spmatrix,np.ndarray], full_reduce: bool = False, lower_triangular: bool = True):

        self._MEM_ALLOCATED = False
        self.L_cached = False
        self.U_cached = False
        self.P_cached = False
        self.Lmat = scipy.sparse.csr_matrix((0,0))
        self.Umat = scipy.sparse.csr_matrix((0,0))
        self.Pmat = scipy.sparse.csr_matrix((0,0))
        self.cpcm = Py2GF2Sparse(pcm)
        self.rr = new RowReduce(self.cpcm[0])
        self._MEM_ALLOCATED = True
        self.full_reduce = full_reduce
        self.lower_triangular = full_reduce
        self.rr.rref(full_reduce,lower_triangular)


    def lu_solve(self, y: np.ndarray) -> np.ndarray:
        """
        Solve the LU decomposition problem for a given array 'y'.
        
        Parameters:
            y (np.ndarray): Array to be solved.
            
        Returns:
            np.ndarray: Solution array.
        """

        if self.full_reduce == True or self.lower_triangular == False:
            self.rr.rref(False,True)
        
        cdef int i
        cdef vector[uint8_t] y_c
        
        y_c.resize(len(y))
        for i in range(len(y)):
            y_c[i] = y[i]
        
        cdef vector[uint8_t] x = self.rr.lu_solve(y_c)
        cdef np.ndarray[uint8_t, ndim=1] x_np = np.zeros(x.size(), dtype=np.uint8)
        for i in range(x.size()):
            x_np[i] = x[i]

        return x_np

    @property
    def L(self) -> scipy.sparse.spmatrix:
        """
        Get the lower triangular matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Lower triangular matrix.
        """
        cdef vector[vector[int]] coords
        if not self.L_cached:
            coords = self.rr.L.nonzero_coordinates()
            self.Lmat = coords_to_scipy_sparse(coords,self.rr.L.m,self.rr.L.n,self.rr.L.entry_count())
        
        self.L_cached = True
        
        return self.Lmat

    @property
    def U(self) -> scipy.sparse.spmatrix:
        """
        Get the upper triangular matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Upper triangular matrix.
        """
        cdef vector[vector[int]] coords
        if not self.U_cached:
            coords = self.rr.U.nonzero_coordinates()
            self.Umat = coords_to_scipy_sparse(coords,self.rr.U.m,self.rr.U.n,self.rr.U.entry_count())
        
        self.U_cached = True
        
        return self.Umat

    @property
    def P(self) -> scipy.sparse.spmatrix:
        """
        Get the permutation matrix from the LU decomposition.
        
        Returns:
            scipy.sparse.spmatrix: Permutation matrix.
        """
        cdef vector[vector[int]] coords
        if not self.P_cached:
            self.rr.build_p_matrix()
            coords = self.rr.P.nonzero_coordinates()
            self.Pmat = coords_to_scipy_sparse(coords,self.rr.P.m,self.rr.P.n,self.rr.P.entry_count())
        
        self.P_cached = True
        
        return self.Pmat

    @property
    def rank(self) -> int:
        """
        Get the rank of the matrix used for the LU decomposition.
        
        Returns:
            int: Rank of the matrix.
        """
        return self.rr.rank

    @property
    def pivots(self) -> np.ndarray:
        """
        Get the pivot positions used during the LU decomposition.
        
        Returns:
            np.ndarray: Array of pivot positions.
        """
        cdef int i
        cdef int count = 0
        out = np.zeros(self.rr.rank, dtype=np.int32)

        for i in range(self.rr.pivots.size()):
            if self.rr.pivots[i] == 1:
                out[count] = i
                count+=1

        return out

    def __del__(self):
        if self._MEM_ALLOCATED:    
            del self.rr
            del self.cpcm