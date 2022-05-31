import numpy as np
from ldpc.codes import rep_code, ring_code, hamming_code
from ldpc import bp_decoder
from tqdm import tqdm

n=10
h=ring_code(n)

print(h)

bpd = bp_decoder(h,error_rate=0.1, max_iter=n,bp_method="ms",ms_scaling_factor=1.0, schedule=1, input_vector_type='syndrome')

error =np.zeros(h.shape[1]).astype(int)
error[[0,1,5,6,7]]=1

syndrome =  h@error%2

print(syndrome)

# bpd.set_inactivated_checks([0])

decoding=bpd.si_decode(syndrome)

# import time
# start = time.process_time()
# for _ in tqdm(range(100)):
#     bpd.decode(syndrome)   
# print(time.process_time() - start)



print("iter",bpd.iter)
print(bpd.log_prob_ratios)
print(bpd.bp_decoding)
print("converge",bpd.converge)
print("output_synd", h@decoding%2)
print(h.shape)
print(bpd.ms_scaling_factor)