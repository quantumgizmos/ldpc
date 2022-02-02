import numpy as np 
from ldpc import protograph as pt

a=pt.array([[(0),(1)]])

print(repr(a))

print(repr(np.kron(a,a)))
print(repr(np.hstack([a,a]).view(pt.array)))


print(repr(pt.hstack([a,a])))

R=pt.RingOfCirculantsF2((0,-1))

print(repr(R))

print(a)

pt.identity(3)