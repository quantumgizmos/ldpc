from ldpc2.bp_decoder import bp_decoder
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from ldpc2.noise_models import generate_bsc_error
from ldpc import bp_decoder as bp_decoder_og


from ldpc2.codes import rep_code
run_count = 10000
error_rate = 0.01
H = rep_code(500)
bpd=bp_decoder(H, error_rate=error_rate, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=10,omp_thread_count=1)
bpd_og=bp_decoder_og(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=10)



m, n = H.shape



for DECODER in [bpd,bpd_og]:
    np.random.seed(42)
    fail = 0
    for _ in tqdm(range(run_count)):

        error = generate_bsc_error(H.shape[1], error_rate)
        z = H@error%2
        x = DECODER.decode(z)

        if not np.array_equal(x, error):
            fail+=1

    print(fail/run_count)
        

