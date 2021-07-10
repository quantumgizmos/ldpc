import numpy as np

from pybp import bp_decoder

from pybp.classical_codes import rep_code

from pybp.mod2sparse import pymod2sparse

H=rep_code(1000,False)


# H = pymod2sparse(H)

# print(H)

# H.check_to_bit=1

# a=H.check_to_bit

# print(a)

# print(H.shape)

from pybp.bp_decoder2 import bp_decoder

bpd=bp_decoder(H,0.1,max_iter=10,bp_method="0")

for i in range(1):
    error=np.zeros(H.shape[1]).astype(int)
    error[np.random.randint(H.shape[1])]=1
    print(error)
    s=H@error %2
    print(bpd.bp_decode(s))
    print()










