import numpy as np
from ldpc2.bp_decoder import BpDecoder

H = np.array([[1,1,0],[0,1,1]])
decoder = BpDecoder(H, error_rate = 0.1)

error = np.array([1,0,0])
syndrome = H@error %2

decoding = decoder.decode(syndrome)
print(decoding)

