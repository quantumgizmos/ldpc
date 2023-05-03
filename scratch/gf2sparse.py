from ldpc2.gf2sparse import rank, kernel
import numpy as np

from ldpc2.codes import hamming_code,rep_code

H = rep_code(1000)

print(H.shape)

print(rank(H))

ker =kernel(H)

print(ker)

# print((H@ker.T).toarray()%2)

