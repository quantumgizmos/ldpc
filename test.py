import numpy as np
from ldpc.codes import rep_code
from ldpc.bp_decoder import bp_decoder

H = rep_code(3)
print(H)

bpd = bp_decoder(H, error_rate = 0.1, max_iter=0, bp_method='ps', ms_scaling_factor=0.625)

syndrome = np.array([1,0])

decoding = bpd.decode(syndrome)
print(decoding)
print(bpd.bp_decoding)
print(bpd.log_prob_ratios)
