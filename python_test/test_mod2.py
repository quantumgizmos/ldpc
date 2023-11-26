import pytest
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from ldpc.mod2 import io_test, rank, kernel, PluDecomposition, pivot_rows
from ldpc.codes import rep_code, ring_code, hamming_code

def test_constructor_rep_code():

    for i in range(2,10):

        H = rep_code(i)
        out = io_test(H)
        assert np.array_equal(out.toarray(),H.toarray()) == True

    for i in range(2,10):
        H = rep_code(i).toarray()
        out = io_test(H)
        assert np.array_equal(out.toarray(),H) == True

    for i in range(2,100):

        H = ring_code(i)
        out = io_test(H)
        assert np.array_equal(out.toarray(),H.toarray()) == True

    for i in range(2,100):
        H = ring_code(i).toarray()
        out = io_test(H)
        assert np.array_equal(out.toarray(),H) == True


def test_rank_rep_code():
    
    for i in range(2,10):
        H = rep_code(i)
        assert rank(H) == i-1

    for i in range(2,10):
        H = ring_code(i)
        assert rank(H) == i-1

    for i in range(2,10):
        H = rep_code(i).T
        assert rank(H) == i-1

    assert rank(rep_code(1000).T) == 999

def test_rank_rep_code_sparse():
    
    for i in range(2,10):
        H = rep_code(i)
        assert rank(H,method="sparse") == i-1

    for i in range(2,10):
        H = ring_code(i)
        assert rank(H,method="sparse") == i-1

    for i in range(2,10):
        H = rep_code(i).T
        assert rank(H,method="sparse") == i-1

    assert rank(rep_code(1000).T) == 999

def test_kernel_rep_code():

    for i in range(2,10):
        H = rep_code(i)
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    for i in range(2,10):
        H = ring_code(i)
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    for i in range(2,10):
        H = rep_code(i).T
        assert kernel(H).nnz == 0

    for i in range(2,10):
        H = ring_code(i).T
        assert kernel(H).shape == (1, i)
        assert not ((H@kernel(H).T).data % 2).any()

    assert kernel(rep_code(1000)).shape == (1, 1000)

def test_kernel_hamming_code_dense():
    for i in range(3,12):
        H = hamming_code(i)
        ker = kernel(H, method = "dense")
        assert ker.shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@ker.T).data % 2).any()

def test_kernel_hamming_code_sparse():
    for i in range(3,12):
        H = hamming_code(i)
        ker = kernel(H, method = "sparse")
        assert ker.shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@ker.T).data % 2).any()

from ldpc.mod2 import nullspace as np_nullspace
def test_kernel_hamming_code_numpy():
    for i in range(3,12):
        H = hamming_code(i)
        ker = np_nullspace(H)
        assert ker.shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@ker.T) % 2).any()


def test_kernel_rep_code_dense():
    for i in range(3,100):
        H = rep_code(i)
        # assert kernel(H, method = "dense").shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H, method="dense").T).data % 2).any()

def test_kernel_rep_code_sparse():
    for i in range(3,100):
        H = rep_code(i)
        # assert kernel(H, method = "sparse").shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H, method="sparse").T).data % 2).any()

from ldpc.mod2.mod2_numpy import nullspace as np_nullspace
def test_kernel_rep_code_numpy():
    for i in range(3,100):
        H = rep_code(i)
        # assert np_nullspace(H).shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@np_nullspace(H).T) % 2).any()

def test_plu_decomposition():
    
    CODES = [rep_code,ring_code,hamming_code]
    
    for code in CODES:
        for d in range(2,10):
            H = code(d)
            plu = PluDecomposition(H)
            P = plu.P
            L = plu.L
            U = plu.U
            assert np.array_equal( (P@L@U).toarray() %2, H.toarray() )

def test_lu_solve():

    CODES = [rep_code,ring_code,hamming_code]
    
    for code in CODES:
        for d in range(2,10):
            H = code(d)
            plu = PluDecomposition(H)
            x = np.random.randint(2, size=H.shape[1])
            y = H@x % 2
            x_solution = plu.lu_solve(y)
            assert np.array_equal(H@x_solution % 2, y)

def test_pivot_rows1():
    # Create a dense numpy array
    dense_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Create a sparse scipy matrix
    sparse_mat = csr_matrix(dense_mat)

    # Test with dense numpy array
    pivots_dense = pivot_rows(dense_mat)
    assert np.array_equal(pivots_dense, np.array([0, 1, 2]))

    # Test with sparse scipy matrix
    pivots_sparse = pivot_rows(sparse_mat)
    assert np.array_equal(pivots_sparse, np.array([0, 1, 2]))


def test_pivot_rows():

    H = hamming_code(3)
    mat = scipy.sparse.vstack([np.zeros(shape=(3,7),dtype=np.uint8),H])

    pivots = pivot_rows(mat)

    assert np.array_equal(pivots, np.array([3,4,5]))

def test_rank_case2():
    
    mat = np.array([[0, 0, 1, 0],
       [0, 0, 1, 1],
       [1, 1, 0, 0],
       [0, 1, 0, 0]])
    
    mat = scipy.sparse.csr_matrix(mat)
    
    assert rank(mat) == 4

def test_rank_case3():
    mat = np.array([[1, 0, 1, 0, 1],
       [0, 1, 0, 1, 1]], dtype=np.uint8)
    
    mat = scipy.sparse.csr_matrix(mat)  
    
    assert rank(mat) == 2

    ker = kernel(mat)
    print(ker.toarray())

    print(ker@mat.T)
    




if __name__ == "__main__":
    test_rank_case3()