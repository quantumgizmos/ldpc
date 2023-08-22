import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ldpc2.codes import rep_code, ring_code
from ldpc2.gf2sparse import io_test, rank

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




if __name__ == "__main__":
    pytest.main([__file__])