import numpy as np
import scipy.sparse
from ldpc2.gf2sparse import gf2sparse

H = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
])
zero = np.zeros(H.shape).astype(int)
H = np.hstack([zero,H])


H =np.array([[1, 1, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 1]], dtype=np.uint8)



if __name__ == "__main__":
    print("Hello world")

    print(H)

    pcm = np.identity(5)

    # pcm = scipy.sparse.identity(10,format='csr')

    # print(isinstance(pcm,scipy.sparse.s))

    test = gf2sparse(H)

    test.lu_decomposition(full_reduce=True)

    print(test.U)

    P = test.L

    print(P@H%2)


