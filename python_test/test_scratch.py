import numpy as np

import ldpc.codes
import ldpc.code_util

import scipy.sparse

import numpy as np
import ldpc.codes
from ldpc import BpDecoder
from ldpc import BpOsdDecoder
from ldpc import BeliefFindDecoder
from ldpc.monte_carlo_simulation import MonteCarloBscSimulation
import ldpc.mod2

from qec.codes import ToricCode



if __name__ == "__main__":



    H1 = ldpc.codes.hamming_code(3)
    H2 = ldpc.codes.hamming_code(3)

    H3 = scipy.sparse.hstack([H1, scipy.sparse.csr_matrix((H1.shape[0], H2.shape[1]))])
    H4 = scipy.sparse.hstack([scipy.sparse.csr_matrix((H2.shape[0], H1.shape[1])), H2])

    H = scipy.sparse.vstack([H3, H4]).astype(np.uint8)

    H5 = scipy.sparse.csr_matrix((H.shape[0], H.shape[1]))


    vec= np.zeros(H.shape[1]).astype(np.uint8)
    vec[[6,7]] = 1
    vec = scipy.sparse.csr_matrix(vec)

    H = scipy.sparse.vstack([H, vec])

    print(H.toarray())


    c1 = H[:,:7]
    c2 = H[:,7:]

    # print(c1.toarray())
    # print(c2.toarray())

    # print(vec.toarray())


    plu1 = ldpc.mod2.PluDecomposition(c1)
    plu2 = ldpc.mod2.PluDecomposition(c2)

    print(plu1.U.toarray())
    print(plu2.L.toarray())