from ldpc2.bp_decoder import bp_decoder
import numpy as np
from tqdm import tqdm

from ldpc.codes import rep_code
run_count = 1000
error_rate = 0.1
np.random.seed(0)
H = rep_code(1000)
bpd=bp_decoder(H, error_rate=error_rate, bp_method='ps', schedule = "parallel", ms_scaling_factor=0.8, max_iter=10,omp_thread_count=4)

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
        

