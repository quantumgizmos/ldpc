# from ldpc.bp_decoder import BpDecoder
# # from udlr.gf2sparse import PluDecomposition
# import ldpc

# print(ldpc.__file__)

# from ldpc.codes import hamming_code

# print(hamming_code(3).toarray().__repr__())

# LDPC Package BP+OSD demo, Hamming(7,4,3) code
import numpy as np
from ldpc import BpOsdDecoder
from ldpc.noise_models import generate_bsc_error

H = np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 1]])

decoder = BpOsdDecoder(H, error_rate=0.1)

syndrome = np.array([1, 0, 1])

recovery = decoder.decode(syndrome)

print(recovery)

>>> [0 0 0 0 1 0 0]

# Monte Carlo Simulation, BSC Channel

decode_fail_count = 0

for _ in range(1000):

    error = generate_bsc_error(7, 0.01)
    syndrome = H @ error % 2

    recovery = decoder.decode(syndrome)

    if not np.array_equal(recovery, error):
        decode_fail_count += 1

logical_error_rate = decode_fail_count / 100



