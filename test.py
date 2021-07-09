import numpy as np

from pybp import bp_decoder

from pybp.classical_codes import rep_code

H=rep_code(10,False)
bpd=bp_decoder(H,0.2,max_iter=-1)

for i in range(1):
    error=np.zeros(H.shape[1]).astype(int)
    error[np.random.randint(H.shape[1])]=1
    print(error)
    s=H@error %2
    print(bpd.bp_decode(s))
    print()










