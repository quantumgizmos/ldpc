#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from scipy.sparse import spmatrix
import numpy as np

cdef mod2sparse* numpy2mod2sparse(mat):
    
    cdef mod2sparse* sparse_mat
    cdef int i,j,m,n
    m=mat.shape[0]
    n=mat.shape[1]
    sparse_mat=mod2sparse_allocate(m,n)

    for i in range(m):
        for j in range(n):
            if mat[i,j]:
                mod2sparse_insert(sparse_mat,i,j)

    return sparse_mat

cdef mod2sparse* spmatrix2mod2sparse(mat):
    
    cdef mod2sparse* sparse_mat
    cdef int i,j,n_rows,n_cols
    m=mat.shape[0]
    n=mat.shape[1]
    sparse_mat=mod2sparse_allocate(m, n)

    for i, j in zip(*mat.nonzero()):
        mod2sparse_insert(sparse_mat, i, j)

    return sparse_mat

cdef mod2sparse* alist2mod2sparse(fname):

    cdef mod2sparse* sparse_mat

    alist_file = np.loadtxt(fname, delimiter='\n',dtype=str)
    matrix_dimensions=alist_file[0].split()
    m=int(matrix_dimensions[0])
    n=int(matrix_dimensions[1])

    sparse_mat=mod2sparse_allocate(m,n)

    for i in range(m):
        for item in alist_file[i+4].split():
            if item.isdigit():
                column_index = int(item)
                mod2sparse_insert(sparse_mat,i,column_index)

    return sparse_mat

def lu_decomp(mat):

    cdef mod2sparse* H = numpy2mod2sparse(mat)
    cdef mod2sparse* L
    cdef mod2sparse* U
    cdef m,n,i,rank,submatrix_size,nnf
    m=mod2sparse_rows(H)
    n=mod2sparse_cols(H)

    if m==n: submatrix_size=m
    elif n>m: submatrix_size=m
    elif m>n: submatrix_size=n

    L=mod2sparse_allocate(m,submatrix_size)
    U=mod2sparse_allocate(submatrix_size,n)
    cdef int* cols = <int*>calloc(n,sizeof(int))
    cdef int* rows = <int*>calloc(m,sizeof(int))

    for i in range(n): cols[i]=i
    for i in range(m): rows[i]=i
    

    nnf=mod2sparse_decomp_osd(H,m,L,U,rows,cols)
    rank=submatrix_size-nnf

    out_rows=np.zeros(m,dtype=int)
    out_cols=np.zeros(n,dtype=int)

    for i in range(n): out_cols[i]=cols[i]
    for i in range(m): out_rows[i]=rows[i]

    free(rows)
    free(cols)
    mod2sparse_free(H)
    mod2sparse_free(L)
    mod2sparse_free(U)

    return [rank,out_rows,out_cols]


cdef class pymod2sparse():

    def __cinit__(self, mat):

        self.MEM_ALLOCATED=False
        self.m,self.n=mat.shape
        self.matrix=numpy2mod2sparse(mat)
        self.iter_axis=-1
        self.vec_n=<char*>calloc(self.n,sizeof(char)) 
        self.vec_m=<char*>calloc(self.m,sizeof(char))
        self.e=mod2sparse_first_in_col(self.matrix,0)
        self.MEM_ALLOCATED=True

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

    cpdef iter_row(self,int row_index,int reverse_iterate):
        
        self.iter_axis=1
        self.reverse_iterate=reverse_iterate
        self.row_index=row_index
        
        if self.reverse_iterate==False:
            self.e=mod2sparse_first_in_row(self.matrix,self.row_index)
        elif self.reverse_iterate==True:
            self.e=mod2sparse_last_in_row(self.matrix,self.row_index)
        
        return self

    cpdef iter_col(self, int col_index, int reverse_iterate):
        
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
    def check_to_bit(self, double value):
        self.e.check_to_bit=value

    @property
    def bit_to_check(self):
        return self.e.bit_to_check
    @bit_to_check.setter
    def bit_to_check(self,double value):
        self.e.bit_to_check=value

    @property
    def sgn(self):
        return self.e.sgn
    @sgn.setter
    def sgn(self,int value):
        self.e.sgn=value

    @property
    def shape(self):
        return (self.m,self.n)
    
    def __dealloc__(self):
        if self.MEM_ALLOCATED:
            mod2sparse_free(self.matrix)
            free(self.vec_n)
            free(self.vec_m)
