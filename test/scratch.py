import numpy as np
from scipy.sparse import coo, coo_matrix,csr_matrix

def test(c,**kwargs):

    a=kwargs.get("a",0)
    b=kwargs.get("b",1)

    print(a,b,c)

test(3,b=6)


z=np.zeros(4)

b=csr_matrix([[1,1],[1,0]])
a=csr_matrix(np.array([1,1])).T

print(a.shape)

b=np.array([[1,1],[1,0]])

a=np.array([1,1])

print(b@a)

print(a.shape[0])