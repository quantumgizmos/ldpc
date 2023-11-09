import numpy as np

import ldpc.codes
import ldpc.code_util


if __name__ == "__main__":

    H = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]])
    

    n,k,d = ldpc.code_util.compute_code_parameters(H,timeout_seconds=0.1)
    print(f"n = {n}, k = {k}, d = {d}")

    # exit(22)


    for _ in range(10000):
        H = ldpc.codes.random_binary_code(8,15,4, variance=1)
        n,k,d = ldpc.code_util.compute_code_parameters(H,timeout_seconds=0.001)

        if d > 4:
            print(f"n = {n}, k = {k}, d = {d}")
            print(H.toarray().__repr__())