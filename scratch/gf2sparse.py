from ldpc2.gf2sparse import rank, kernel, LuDecomposition
import numpy as np
import scipy.sparse as sp

from ldpc2.codes import hamming_code,rep_code
# from ldpc2.gf2sparse import rank as rank_v2

# from ldpc.mod2 import rank as rank_v1
# from ldpc.mod2 import nullspace as kernel_v1
# from ldpc2.gf2sparse import kernel as kernel_v2

H = rep_code(3)

# r=rank(H.toarray())
# print(r)


a = kernel(H.toarray())
print(a.toarray())
