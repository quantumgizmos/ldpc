from ldpc.codes import hamming_code, rep_code, ring_code
import scipy.sparse
import numpy as np
from ldpc import mod2
from ldpc import code_util

def test_construct_generator_matrix():

    for i in range(3,10):

        H = hamming_code(i)
        G = code_util.construct_generator_matrix(H)
        assert not ((H@G.T).data %2).any()

