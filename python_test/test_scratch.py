import numpy as np

from ldpc.codes import hamming_code, rep_code

from ldpc.gf2sparse import nullspace

H = hamming_code(3)

ker = nullspace(H, method='dense')

nulls = H@ker.T

print(nulls.data %2)

from ldpc.bp_decoder import bp_decoder, BpDecoder

H = rep_code(3)

# input_vector = np.array([1,0,1,0,1])

input_vector = np.array([1,0])

bpd = BpDecoder(H,error_rate=0.1, input_vector_type='syndrome')

print(bpd.decode(input_vector))

print(bpd.input_vector_type)
