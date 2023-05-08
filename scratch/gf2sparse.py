from ldpc2.gf2sparse import rank, kernel, LuDecomposition
import numpy as np
import scipy.sparse as sp

from ldpc2.codes import hamming_code,rep_code

H = rep_code(3)


H = sp.vstack([H, H])


print(H.toarray())

print(rank(H))

lud = LuDecomposition(H)



error = np.zeros(H.shape[1], dtype=np.uint8)

error[0] = 1

syndrome = H@error%2


x =lud.solve(syndrome)

print(x)