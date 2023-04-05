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
error_rate = 0.01
H = rep_code(5)
bpd=BpDecoder(H, error_rate=error_rate, bp_method='ms', schedule = "parallel", ms_scaling_factor=1.0, max_iter=10,omp_thread_count=1)
bpd_og=bp_decoder_og(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=10)

bposd_og=bposd_decoder_og(H, error_rate=error_rate, bp_method='ms', ms_scaling_factor=1.0, max_iter=10,osd_method=1,osd_order=0)
osdD=BpOsdDecoder(H, error_rate=error_rate, bp_method='ms', schedule = "serial", ms_scaling_factor=1.0, max_iter=10,omp_thread_count=1,osd_order=0,osd_method=1)


from ldpc2.monte_carlo_simulation import McSim

# McSim(H, error_rate=error_rate, Decoder=bpd, target_run_count=10000,seed=42)
# McSim(H, error_rate=error_rate, Decoder=bpd_og, target_run_count=10000,seed=42)


from ldpc2.bp_decoder import SoftInfoBpDecoder

sbpd = SoftInfoBpDecoder(H, error_rate = 0.1, max_iter = 10, cutoff = 10)

soft_syndrome = 10*np.ones(4)
soft_syndrome[0] = -10
soft_syndrome[1] = -1*5

print(soft_syndrome)

print(sbpd.decode(soft_syndrome))
print(sbpd.soft_syndrome)
print(sbpd.schedule)
print(sbpd.bp_method)

pcm = np.eye(3, dtype=int)
pcm += np.roll(pcm, 1, axis=1)
print(pcm)