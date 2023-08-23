import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc2.codes import rep_code, ring_code, hamming_code
from ldpc2.gf2sparse import io_test, rank, kernel

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

def test_kernel_hamming_code():
    for i in range(3,10):
        H = hamming_code(i)
        assert kernel(H).shape == (2**i - 1 - i, 2**i - 1)
        assert not ((H@kernel(H).T).data % 2).any()



if __name__ == "__main__":
    pytest.main([__file__])