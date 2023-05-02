#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Tuple, Union

cdef shared_ptr[GF2Sparse] Py2GF2Sparse(pcm: Union[scipy.sparse,np.ndarray]):
    
    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef shared_ptr[GF2Sparse] cpcm = make_shared[GF2Sparse](m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.get().insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.get().insert_entry(rows[i], cols[i])
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
    

    return cpcm

cdef GF2Sparse2Py(shared_ptr[GF2Sparse] cpcm):
    cdef int entry_count = cpcm.get().entry_count()
    cdef vector[vector[int]] entries = cpcm.get().nonzero_coordinates()

    cdef int m
    cdef int n

    out = scipy.sparse.csr_matrix((m,n),dtype=np.int8)
    for i in range(entry_count):
        out[entries[i][0],entries[i][1]] = 1

    return out

def rank(pcm: Union[scipy.sparse,np.ndarray]) ->int:
    cdef shared_ptr[GF2Sparse] cpcm = Py2GF2Sparse(pcm)
    rank: int = NotImplemented
    return rank