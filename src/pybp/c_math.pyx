import numpy as np

cdef double fast_tanh_c(double x):
    return tanh(x)

def fast_tanh(x):

    '''
    A fast inplementation of the tanh function making use of the c math libary.

    INPUT: x, double
    OUTPUT, tanh(x), double
    
    '''

    return fast_tanh_c(x)

cdef class vector:

    def __init__(self,length):

        pass

    def __cinit__(self,length):

        if length<0:
            raise ValueError("The vector length must be a positive integer")

        self.length=length
        self.values=<double*>calloc(length,sizeof(double))

        for i in range(length):
            self.values[i]=np.random.random()

    cdef public void shuffle(self):
        for i in range(self.length):
            self.values[i]=np.random.random()

    def __repr__(self):
        out=np.zeros(self.length)

        for i in range(self.length):
            out[i]=self.values[i]
        return str(out)

