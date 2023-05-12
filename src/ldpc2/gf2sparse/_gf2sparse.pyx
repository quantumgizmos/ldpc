#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Tuple, Union

cdef shared_ptr[GF2Sparse] Py2GF2Sparse(pcm):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse.spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef shared_ptr[GF2Sparse] cpcm = make_shared[GF2Sparse](m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.get().insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse.spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.get().insert_entry(rows[i], cols[i])
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix.spmatrix object, not {type(pcm)}")
    

    return cpcm

cdef GF2Sparse2Py(shared_ptr[GF2Sparse] cpcm):
    cdef int entry_count = cpcm.get().entry_count()
    cdef vector[vector[int]] entries = cpcm.get().nonzero_coordinates()

    cdef int m = cpcm.get().m
    cdef int n = cpcm.get().n

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    cdef int i

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)

    return smat


def rank(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) ->int:
    cdef shared_ptr[GF2Sparse] cpcm = Py2GF2Sparse(pcm)
    cdef RowReduce* rr = new RowReduce(cpcm)
    rr.rref(False,False)
    cdef int rank = rr.rank
    del rr
    return rank

def kernel(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) -> scipy.sparse.spmatrix:
    cdef shared_ptr[GF2Sparse] cpcm = Py2GF2Sparse(pcm)
    cdef vector[vector[np.uint8_t]] ckernel = cy_kernel(cpcm)
    cdef int k = ckernel.size()
    cdef int n = cpcm.get().n
    cdef int i
    # ctypedef np.uint8_t dtype_t
    ker = np.zeros((k,n), dtype=np.uint8)
    for i in range(k):
        for j in range(n):
            ker[i,j] = ckernel[i][j]

    return ker

cdef class LuDecomposition():

    def __cinit__(self, pcm, full_reduce=False, lower_triangular=False):
        self.m = pcm.shape[0]
        self.n = pcm.shape[1]
        self.cpcm = Py2GF2Sparse(pcm)
        self.rr = make_shared[RowReduce](self.cpcm)
        self.rr.get().rref(full_reduce,lower_triangular)

    def solve(self, y):
        cdef vector[uint8_t] cy
        cdef vector[uint8_t] cx

        cy.resize(self.m)
        # cx.resize(self.n)

        cdef int i
        for i in range(self.m): cy[i] = y[i]

        cx = self.rr.get().lu_solve(cy)

        cdef np.ndarray[np.uint8_t, ndim=1] x = np.zeros(self.n, dtype=np.uint8)
        for i in range(self.n): x[i] = cx[i]
        
        return x