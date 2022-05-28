import numpy as np
from mbp import bp_decoder as bp_decoder2
from tqdm import tqdm
from qec.quantum_codes import toric_code

hx=np.loadtxt("hgp_hx.txt").astype(np.uint8)
# lx=np.loadtxt("hgp_lx.txt").astype(np.uint8)


from ldpc import bp_decoder as bp_decoder1

er=0.05
runs=500

converge=0

n=hx.shape[1]

error=np.zeros(n,dtype=np.uint8)

bpd1=bp_decoder1(hx,error_rate=er,bp_method="ms", schedule=1, max_iter=10,ms_scaling_factor=0.625)

error_channel=er*np.ones(n)
bpd2=bp_decoder2(hx,error_channel,bp_method=1, max_iter=10,schedule=1,ms_scaling_factor=0.625)

converge=0
np.random.seed(42)
for i in tqdm(range(runs)):

    for j in range(n):
        if np.random.random()<er: error[j]=1
        else: error[j]=0

    syndrome = hx@error%2

    decoding = bpd2.decode(syndrome)
    candidate_syndrome = hx@decoding%2
    # print(syndrome)
    # print(candidate_syndrome)
    # print((candidate_syndrome+syndrome)%2)
    if bpd2.converge ==1: converge+=1

print(1-converge/runs)

converge=0
np.random.seed(42)
for i in tqdm(range(runs)):

    for j in range(n):
        if np.random.random()<er: error[j]=1
        else: error[j]=0

    syndrome = hx@error%2

    bpd1.decode(syndrome)
    if bpd1.converge ==1: converge+=1

print(1-converge/runs)

print(bpd1.schedule)
print(bpd1.bp_method)