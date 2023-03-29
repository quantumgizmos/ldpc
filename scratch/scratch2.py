from ldpc2.codes import rep_code, ring_code
from ldpc2.noise_models import generate_bsc_error

H = ring_code(2)

print(H.toarray())

# error = generate_bsc_error(1, 0.2)
# print(error)

# print(H.toarray())

from ldpc2.bp_decoder import BpDecoder, bp_decoder

BpDecoder(H, error_rate=0.2, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=100,omp_thread_count=1)

bp_decoder()