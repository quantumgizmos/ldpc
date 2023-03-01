import numpy as np
import scipy.sparse
from ldpc2.gf2sparse import gf2sparse

if __name__ == "__main__":
    print("Hello world")

    pcm = np.identity(5)

    pcm = scipy.sparse.identity(10,format='csr')

    # print(isinstance(pcm,scipy.sparse.s))

    test = gf2sparse(pcm)
    test.T
    a = test.kernel()
    print(test)
    print(a)
    a=test.to_scipy_sparse()
    print(a)


