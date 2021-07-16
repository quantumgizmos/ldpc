
cdef class pymod2sparse():

    cdef mod2sparse *matrix
    cdef mod2entry *e
    cdef int m,n,iter_axis,reverse_iterate, row_index, col_index,start
    cdef char *vec_n
    cdef char *vec_m

    cpdef iter_row(self,int row_index,int reverse_iterate=False)
    cpdef iter_col(self,int col_index,int reverse_iterate=False)
    cpdef np.ndarray[np.int_t, ndim=1] mul(self, np.ndarray[np.int_t, ndim=1] vector)

