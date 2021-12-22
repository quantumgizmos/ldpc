import numpy as np
from scipy.sparse import coo, coo_matrix,csr_matrix
from ldpc import bposd_decoder
from ldpc.codes import hamming_code

H=hamming_code(6)
m,n=H.shape
bpd=bposd_decoder(H,error_rate=0.1,osd_order=2)

error=np.zeros(n).astype(int)

error[0]=1

s=H@error%2
print(s)
bpd.decode(s)

print(bpd.osdw_decoding)

