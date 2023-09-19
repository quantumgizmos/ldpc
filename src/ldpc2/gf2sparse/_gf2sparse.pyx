#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Tuple, Union
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

cdef csr_to_scipy_sparse(vector[vector[int]]& row_adjacency_list, int m, int n, int entry_count):

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

cdef GF2Sparse2Py(GF2Sparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat


def rank(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) ->int:
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    cdef RowReduce* rr = new RowReduce(cpcm[0])
    rr.rref(False,False)
    cdef int rank = rr.rank
    del rr
    del cpcm
    return rank

def kernel(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) -> scipy.sparse.spmatrix:
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    cdef CsrMatrix csr = cy_kernel(cpcm)
    del cpcm
    return csr_to_scipy_sparse(csr.row_adjacency_list, csr.m, csr.n, csr.entry_count)

def row_complement_basis(pcm: Union[scipy.sparse.spmatrix,np.ndarray]) -> scipy.sparse.spmatrix:
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    cdef CsrMatrix csr = cy_row_complement_basis(cpcm)
    del cpcm
    return csr_to_scipy_sparse(csr.row_adjacency_list, csr.m, csr.n, csr.entry_count)

def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
    output = GF2Sparse2Py(cpcm)
    del cpcm
    return output

cdef class PluDecomposition():

    def __cinit__(self, pcm: Union[scipy.sparse.spmatrix,np.ndarray], full_reduce: bool = False, lower_triangular: bool = True):

        self.MEM_ALLOCATED = False
        self.L_cached = False
        self.U_cached = False
        self.P_cached = False
        self.Lmat = scipy.sparse.csr_matrix((0,0))
        self.Umat = scipy.sparse.csr_matrix((0,0))
        self.Pmat = scipy.sparse.csr_matrix((0,0))
        cdef GF2Sparse* cpcm = Py2GF2Sparse(pcm)
        self.rr = new RowReduce(cpcm[0])
        self.MEM_ALLOCATED = True
        self.full_reduce = full_reduce
        self.lower_triangular = full_reduce
        self.rr.rref(full_reduce,lower_triangular)


    def lu_solve(self, y: np.ndarray)->np.ndarray:

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
    def L(self):
        cdef vector[vector[int]] coords
        if not self.L_cached:
            coords = self.rr.L.nonzero_coordinates()
            self.Lmat = coords_to_scipy_sparse(coords,self.rr.L.m,self.rr.L.n,self.rr.L.entry_count())
        
        self.L_cached = True
        
        return self.Lmat

    @property
    def U(self):
        cdef vector[vector[int]] coords
        if not self.U_cached:
            coords = self.rr.U.nonzero_coordinates()
            self.Umat = coords_to_scipy_sparse(coords,self.rr.U.m,self.rr.U.n,self.rr.U.entry_count())
        
        self.U_cached = True
        
        return self.Umat

    @property
    def P(self):
        cdef vector[vector[int]] coords
        if not self.P_cached:
            self.rr.build_p_matrix()
            coords = self.rr.P.nonzero_coordinates()
            self.Pmat = coords_to_scipy_sparse(coords,self.rr.P.m,self.rr.P.n,self.rr.P.entry_count())
        
        self.P_cached = True
        
        return self.Pmat

    @property
    def rank(self):
        return self.rr.rank

    @property
    def pivots(self):
        cdef int i
        cdef int count = 0
        out = np.zeros(self.rr.rank, dtype=np.int32)

        for i in range(self.rr.pivots.size()):
            if self.rr.pivots[i] == 1:
                out[count] = i
                count+=1

        return out

    def __del__(self):
        if self.MEM_ALLOCATED:    
            del self.rr
            del self.cpcm