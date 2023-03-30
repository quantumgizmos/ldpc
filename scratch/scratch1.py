from ldpc2.bp_decoder import BpDecoder
from ldpc2.bposd_decoder import BpOsdDecoder
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from ldpc2.noise_models import generate_bsc_error
from ldpc import bp_decoder as bp_decoder_og
from ldpc import bposd_decoder as bposd_decoder_og


from ldpc2.codes import rep_code
run_count = 10000
error_rate = 0.1
H = rep_code(500)
bpd=BpDecoder(H, error_rate=error_rate, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=10,omp_thread_count=1)
bpd_og=bp_decoder_og(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=10)
bposd_og=bposd_decoder_og(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=10,osd_method=1,osd_order=0)
osdD=BpOsdDecoder(H, error_rate=error_rate, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=10,omp_thread_count=1,osd_order=0,osd_method=1)



m, n = H.shape



for DECODER in [osdD,bposd_og]:
# for DECODER in [bpd,bpd_og]:
    np.random.seed(42)
    fail = 0
    converge_fail = 0
    syndrome_converge = 0
    for _ in tqdm(range(run_count)):

        error = generate_bsc_error(H.shape[1], error_rate)
        z = H@error%2
        x = DECODER.decode(z)

        if not DECODER.converge: converge_fail+=1
        if not np.array_equal(x, error):
            fail+=1

        zc = H@x%2

        if np.array_equal(zc, z):
            syndrome_converge+=1

    print(f"ler: {fail/run_count}", f"converge failure rate: {converge_fail/run_count}", f"Syndrome match failure rate: {1.0-syndrome_converge/run_count}")
        

