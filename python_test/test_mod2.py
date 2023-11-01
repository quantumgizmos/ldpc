import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc.gf2sparse import io_test, rank, kernel, PluDecomposition
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
        assert kernel(H, method = "dense").shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H).T).data % 2).any()

def test_kernel_hamming_code_sparse():
    for i in range(3,12):
        H = hamming_code(i)
        assert kernel(H, method = "sparse").shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H).T).data % 2).any()

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


if __name__ == "__main__":
    pytest.main([__file__])