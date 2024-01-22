import numpy as np

import ldpc.codes
import ldpc.code_util
from ldpc.bplsd_decoder import BpLsdDecoder
import scipy

if __name__ == "__main__":

    syndrome = np.array([  3,   6,   8,  13,  36,  40,  60,  72,  84, 104, 120, 156, 180, 188])
    pcm = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz").toarray()

    output_file = open("cpp_test/test_inputs/qdlpc_test.csv", "w")

    csr = "["

    for row in pcm:
        csr += "["
        for i, col in enumerate(row):
            if col == 1:
                csr += str(i) + ","
        csr = csr[:-1]
        csr += "],"
    csr = csr[:-1]
    csr += "]"

    # print(csr)

    m,n = pcm.shape

    print(f"{m};{n};{csr}", file=output_file)

    
    
    




    