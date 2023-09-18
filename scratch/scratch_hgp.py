import numpy as np
import scipy.sparse as sp

from ldpc2.bp_decoder import BpDecoder
from ldpc2.bposd_decoder import BpOsdDecoder
from ldpc import bposd_decoder as bposd_decoder_og
from ldpc import bp_decoder as bp_decoder_og
from tqdm import tqdm
from ldpc.codes import ring_code

from ldpc2.noise_models import generate_bsc_error

from bposd.css import css_code
from bposd.hgp import hgp

from ldpc2.monte_carlo_simulation import McSim

h = np.loadtxt("scratch/16_4_6.txt", dtype=int)
# h=ring_code(30)
qcode = hgp(h,h)
qcode.test()

hx = sp.csr_matrix(qcode.hx, dtype=np.uint8)
hz = sp.csr_matrix(qcode.hz, dtype=np.uint8)
lx = sp.csr_matrix(qcode.lx, dtype=np.uint8)
lz = sp.csr_matrix(qcode.lz, dtype=np.uint8)

run_count = 1000
error_rate = 0.05

osd = BpOsdDecoder(hx,error_rate=error_rate, bp_method='ms', schedule="parallel", ms_scaling_factor=0.625, max_iter=10,omp_thread_count=1,osd_order=5,osd_method="osd_e",random_schedule_seed=0)
bp = BpDecoder(hx,error_rate=error_rate, bp_method='ms', schedule="parallel", ms_scaling_factor=0.93, max_iter=20,omp_thread_count=1, random_schedule_seed = 0)

osd_og = bposd_decoder_og(hx,error_rate=error_rate, bp_method='ms', ms_scaling_factor=0.625, max_iter=10,osd_order=5,osd_method="osd_e")
bp_og = bp_decoder_og(hx,error_rate=error_rate, bp_method='ms', ms_scaling_factor=0.625, max_iter=10)
# bpd = BpDecoder(hx,error_rate=error_rate, bp_method='ms', schedule="serial", ms_scaling_factor=0.625, max_iter=50,omp_thread_count=1)

# McSim(hx, error_rate=error_rate, Decoder=bpd, target_run_count=run_count,seed=42)
seed = 43
# seed = np.random.randint(0,1000000)

for DECODER in [osd_og, osd]:
    np.random.seed(seed)
    fail = 0

    for i in tqdm(range(run_count)):

        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2

        decoding = DECODER.decode(z)

        # assert np.array_equal(hx@decoding%2, z)

        residual = (decoding + error) %2

        if DECODER == bp:
            if not DECODER.converge:
                fail+=1
                continue

        if np.any((lx@residual)%2):
            fail+=1

    print(f"ler: {fail/run_count}")


exit(22)

for DECODER in [bp_og,bp]:
    np.random.seed(seed)
    fail = 0

    for i in tqdm(range(run_count)):

        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2

        decoding = DECODER.decode(z)

        residual = (decoding + error) %2

        if DECODER.converge:
            if np.any((lx@residual)%2):
                fail+=1
                # print(np.nonzero(error))
                # print("Logical failure")
        else:
            fail+=1
            # print(np.nonzero(error))
            # print("Convergence failure")

    print(f"ler: {fail/run_count}")


    

# for DECODER in [bp_og,bp]:

#     print(DECODER)

#     error = np.zeros(hx.shape[1]).astype(np.uint8)
    
#     error[[332, 333, 338]] = 1

#     z = hx@error%2

#     decoding = DECODER.decode(z)

#     residual = (decoding + error) %2

#     if DECODER.converge:
#         if np.any((lx@residual)%2):
#             print(np.nonzero(error))
#             print("Logical failure")
#         else:
#             print(np.nonzero(hx@residual%2))
#             print("Logical success")
#     else:
#         print(np.nonzero(error))
#         print("Convergence failure")




