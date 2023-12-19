import numpy as np

import ldpc.codes
import ldpc.code_util
from ldpc.bplsd_decoder import BpLsdDecoder


if __name__ == "__main__":

    n = 10

    h = ldpc.codes.hamming_code(n)

    lsd = BpLsdDecoder(h, error_rate = 0.1, max_iter = 3, bits_per_step = 1)

    print(h.toarray())

    synd =  np.random.randint(0,2,n)

    print(synd)

    decoding = lsd.decode(synd)

    print(h@decoding % 2)



    