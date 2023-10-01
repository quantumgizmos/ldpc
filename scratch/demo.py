from ldpc.bp_decoder import BpDecoder
from ldpc.codes import rep_code
import numpy as np


H = rep_code(3)
decoder = BpDecoder(
    H, error_rate = 0.1, bp_method = "ps")

decoding = decoder.decode(np.array([1,0]))
print(decoding)




