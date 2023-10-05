import numpy as np
import scipy.sparse as sp

from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from tqdm import tqdm
from ldpc.codes import ring_code

from ldpc.bp_decoder import bp_decoder as bp_og_syntax_decoder
from ldpc.bposd_decoder import bposd_decoder as osd_og_syntax_decoder

from ldpc.noise_models import generate_bsc_error

from qec.css import CssCode
from qec.hgp import HyperGraphProductCode
from qec.codes import ToricCode

from ldpc.monte_carlo_simulation import McSim

h = np.loadtxt("scratch/16_4_6.txt", dtype=int)
# h=ring_code(30)
qcode = HyperGraphProductCode(h,h)

qcode = ToricCode(20)

print(qcode)

hx = qcode.hx
hz = qcode.hz
lx = qcode.lx
lz = qcode.lz

run_count = 1000
error_rate = 0.01

bp = BpDecoder(hx,error_rate=error_rate, bp_method='ms', schedule="parallel", ms_scaling_factor=0.625, max_iter=10,omp_thread_count=1, random_schedule_seed = 0)
osd = BpOsdDecoder(hx,error_rate=error_rate, bp_method='ms', schedule="parallel", ms_scaling_factor=0.9, max_iter=10,omp_thread_count=1,osd_order=0,osd_method="osd_cs",random_schedule_seed=10)

from python_bp import PyBp

pybp = PyBp(hx.toarray(),error_rate=error_rate, max_iter=50)

# osd_og_syntax = osd_og_syntax_decoder(hx,error_rate=error_rate, bp_method='ps', ms_scaling_factor=0.625, max_iter=10,osd_order=5,osd_method="osd_e")
# bp_og_syntax = bp_og_syntax_decoder(hx,error_rate=error_rate, bp_method='ms', ms_scaling_factor=0.625, max_iter=10)

# McSim(hx, error_rate=error_rate, Decoder=bpd, target_run_count=run_count,seed=42)
seed = 40


for DECODER in [bp,osd]:
    np.random.seed(seed)
    fail = 0

    for i in tqdm(range(run_count)):

        error = generate_bsc_error(hx.shape[1], error_rate)
        z = hx@error%2

        # print(np.count_nonzero(z))

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


# exit(22)

for DECODER in []:
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




