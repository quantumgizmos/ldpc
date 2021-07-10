
cdef class pymod2sparse():

    cdef mod2sparse *matrix
    cdef mod2entry *e
    cdef int m,n,iter_axis,reverse_iterate, row_index, col_index,start
    cdef char *vec_n
    cdef char *vec_m


    def __cinit__(self, mat):

        self.m,self.n=mat.shape
        self.matrix=numpy2mod2sparse(mat)
        self.iter_axis=-1
        self.vec_n=<char*>calloc(self.n,sizeof(char)) #error string
        self.vec_m=<char*>calloc(self.m,sizeof(char)) #error string


    def __iter__(self):
        self.start=True
        return self
    
    def __next__(self):
        
        if self.iter_axis==1 and self.reverse_iterate==False:
            if not self.start:
                self.e=mod2sparse_next_in_row(self.e)
            else:
                self.start=False
            if not mod2sparse_at_end(self.e):
                return self
            else:
                raise StopIteration
        
        elif self.iter_axis==1 and self.reverse_iterate==True:
            if not self.start:
                self.e=mod2sparse_prev_in_row(self.e)
            self.start=False
            if not mod2sparse_at_end(self.e):
                return self
            else:
                raise StopIteration

        if self.iter_axis==0 and self.reverse_iterate==False:
            if not self.start:
                self.e=mod2sparse_next_in_col(self.e)
            else:
                self.start=False
            if not mod2sparse_at_end(self.e):
                return self
            else:
                raise StopIteration
        
        elif self.iter_axis==0 and self.reverse_iterate==True:
            if not self.start:
                self.e=mod2sparse_prev_in_col(self.e)
            self.start=False
            if not mod2sparse_at_end(self.e):
                return self
            else:
                raise StopIteration

    cpdef iter_row(self,int row_index,int reverse_iterate=False):
        
        self.iter_axis=1
        self.reverse_iterate=reverse_iterate
        self.row_index=row_index
        
        if self.reverse_iterate==False:
            self.e=mod2sparse_first_in_row(self.matrix,self.row_index)
        elif self.reverse_iterate==True:
            self.e=mod2sparse_last_in_row(self.matrix,self.row_index)
        
        return self

    cpdef iter_col(self, int col_index, int reverse_iterate=False):
        
        self.iter_axis=0
        self.reverse_iterate=reverse_iterate
        self.col_index=col_index
        
        if self.reverse_iterate==False:
            self.e=mod2sparse_first_in_col(self.matrix,self.col_index)
        elif self.reverse_iterate==True:
            self.e=mod2sparse_last_in_col(self.matrix,self.col_index)
        
        return self

    cpdef np.ndarray[np.int_t, ndim=1] mul(self, np.ndarray[np.int_t, ndim=1] vector):
        
        if len(vector)!=self.n:
            raise ValueError(f'Dimension mismatch. The supplied vector of length {len(vector)} cannot be multiplied by a matrix with dimesnions ({self.m},{self.n})!')

        self.vec_n=numpy2char(vector,self.vec_n)

        mod2sparse_mulvec(self.matrix,self.vec_n,self.vec_m)

        return char2numpy(self.vec_m,self.m)

    def __matmul__(self, vector):

        return self.mul(vector)

    @property
    def check_to_bit(self):
        return self.e.check_to_bit
    
    @check_to_bit.setter
    def check_to_bit(self,value):
        self.e.check_to_bit=value

    @property
    def bit_to_check(self):
        return self.e.bit_to_check
    @bit_to_check.setter
    def bit_to_check(self,value):
        self.e.bit_to_check=value

    @property
    def sgn(self):
        return self.e.sgn
    @sgn.setter
    def sgn(self,value):
        self.e.sgn=value

    @property
    def shape(self):
        return (self.m,self.n)
    
    def __dealloc__(self):
        mod2sparse_free(self.matrix)
        free(self.vec_n)
        free(self.vec_m)