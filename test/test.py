import numpy as np
from ldpc import codes, mod2
from ldpc import bposd_decoder as bp_decoder
from tqdm import tqdm
from scipy.sparse import csr_matrix

N=500
H=codes.rep_code(N)
# H=csr_matrix(H)
error_rate=0.3

bpd=bp_decoder(H,error_rate=error_rate,max_iter=N,bp_method="product_sum")
error=np.ones(N).astype(int)
print(error.shape[0])

print(bpd.print_test())

# for i,_ in enumerate(error):
#     if np.random.random()<error_rate:
#         error[i]=(error[i]+1)%2
#     # else:
#     #     error[i]=0

# print()
# s=H@error%2

# r=bpd.decode(np.array([1,1,1,0]))

# print(r)



# exit(22)


for _ in tqdm(range(10000)):

    for i,_ in enumerate(error):
        if np.random.random()<error_rate:
            error[i]=(error[i]+1 )%2
            # error[i]=1
        else:
            error[i]=0

    # error=csr_matrix(error).T

    # s=H@error
    r=bpd.decode(error)

    # print(r)
