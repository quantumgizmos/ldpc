from ldpc2.gf2sparse import rank, kernel
import numpy as np
import scipy.sparse as sp

from ldpc2.codes import hamming_code,rep_code
from ldpc2.gf2sparse import PluDecomposition, rank, kernel

# from ldpc.mod2 import rank as rank_v1
# from ldpc.mod2 import nullspace as kernel_v1
# from ldpc2.gf2sparse import kernel as kernel_v2

H = hamming_code(5)

# r=rank(H.toarray())
# print(r)


plu = PluDecomposition(H)
L = plu.L
print(L.toarray())
U = plu.U
print(U.toarray())
P = plu.P
print(P.toarray())

PLU  = (P@L@U) 
print(PLU.toarray()%2)

