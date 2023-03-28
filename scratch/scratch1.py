from ldpc2.bp_decoder import bp_decoder
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from ldpc2.noise_models import generate_bsc_error


from ldpc2.codes import rep_code
run_count = 1000
error_rate = 0.2
np.random.seed(0)
H = rep_code(500)
bpd=bp_decoder(H, error_rate=error_rate, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=100,omp_thread_count=1)

m, n = H.shape

fail = 0
for _ in tqdm(range(run_count)):

    error = generate_bsc_error(H.shape[1], error_rate)
    z = H@error%2
    x = bpd.decode(z)

    if not np.array_equal(x, error):
        fail+=1

print(fail/run_count)
        

