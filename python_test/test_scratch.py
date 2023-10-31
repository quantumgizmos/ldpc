import numpy as np

from ldpc.codes import hamming_code

from ldpc.gf2sparse import kernel

H = hamming_code(3)

ker = kernel(H, method='dense')

nulls = H@ker.T

print(nulls.data %2)
