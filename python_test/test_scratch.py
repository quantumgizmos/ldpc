import numpy as np

import ldpc.codes
import ldpc.code_util
from ldpc.bplsd_decoder import BpLsdDecoder
import scipy

if __name__ == "__main__":

    syndrome = np.array([  3,   6,   8,  13,  36,  40,  60,  72,  84, 104, 120, 156, 180, 188])
    pcm = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
    print(pcm.shape)
    




    