import numpy as np
from ldpc.belief_find_decoder import BeliefFindDecoder

pcm = np.zeros((2,7)).astype(int)
pcm[0,0] = 1
pcm[0,1] = 1
pcm[0,2] = 1
pcm[0,3] = 1
pcm[1,3] = 1
pcm[1,4] = 1
pcm[1,5] = 1
pcm[1,6] = 1

print(pcm)

syndrome = np.array([0,1])

bpd = BeliefFindDecoder(pcm,error_rate = 0.1,uf_method = 'peeling',max_iter = 0)

decoding = bpd.decode(syndrome)
print(bpd.converge)

print(decoding)

