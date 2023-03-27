from ldpc2.bp_decoder import bp_decoder
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


from ldpc.codes import rep_code
run_count = 1000
error_rate = 0.2
np.random.seed(0)
H = csr_matrix(rep_code(10000))
bpd=bp_decoder(H, error_rate=error_rate, bp_method='ps', schedule = "serial", ms_scaling_factor=1.0, max_iter=100,omp_thread_count=1)

m, n = H.shape

error = np.zeros(H.shape[1]).astype(np.uint8)
fail = 0
for _ in tqdm(range(run_count)):

    for i in range(n):
        if np.random.rand() < error_rate:
            error[i] = 1
        else:
            error[i] = 0

    z = H@error%2

    x = bpd.decode(z)

    if not np.array_equal(x, error):
        fail+=1

print(fail/run_count)
        

