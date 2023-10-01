from ldpc.bp_decoder import BpDecoder
from ldpc.bposd_decoder import BpOsdDecoder
import numpy as np
from ldpc import bp_decoder as bp_decoder_v1
from ldpc.codes import rep_code
from ldpc.monte_carlo_simulation import McSim

run_count = 1000
error_rate = 0.40
H = rep_code(1000)
bpd=BpDecoder(H, error_rate=error_rate, bp_method='ms', schedule = "serial", ms_scaling_factor=1, max_iter=0,omp_thread_count=1, random_schedule_seed = 5)
bpd_v1=bp_decoder_v1(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=20)
seed = np.random.randint(2**32 -1)
seed = 33
# McSim(H, error_rate=error_rate, Decoder=bpd_v1, target_run_count=run_count,seed=seed, run=True)
McSim(H, error_rate=error_rate, Decoder=bpd, target_run_count=run_count,seed=seed, run=True)

print(bpd.random_schedule_seed)

from ldpc.bp_flip import BpFlipDecoder