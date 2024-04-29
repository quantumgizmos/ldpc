import numpy as np

import ldpc.codes
import ldpc.code_util
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bpk_decoder import BpKruskalDecoder
import scipy

if __name__ == "__main__":

    syndrome_i = np.array([  3,   6,   8,  13,  36,  40,  60,  72,  84, 104, 120, 156, 180, 188])
    pcm = scipy.sparse.load_npz("python_test/pcms/hx_400_16_6.npz")
    # print(pcm.shape)

    syndrome = np.zeros(pcm.shape[0], dtype=np.uint8)
    syndrome[syndrome_i] = 1

    bpd = BpKruskalDecoder(pcm, error_rate=0.1, max_iter=100, bp_method="ms", ms_scaling_factor=0.1)

    bpd.decode(syndrome)
    print(bpd.converge)

    




    