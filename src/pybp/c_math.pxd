cimport numpy as np
from libc.stdlib cimport malloc, calloc,free
from libc.math cimport log, tanh, isnan, abs

cdef double fast_tanh_c(double)

cdef class vector:
    cdef double *values
    cdef public int length
    cdef public void shuffle(self)

#cpdef extern Shrubbery standard_shrubbing()
