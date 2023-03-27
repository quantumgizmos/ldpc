from ldpc2.bp_decoder import bp_decoder
import numpy as np

bpd=bp_decoder(np.array([[1]]), error_rate=0.1, bp_method='min_sum', ms_scaling_factor=1.0)