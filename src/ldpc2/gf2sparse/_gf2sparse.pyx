#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse

cdef class gf2sparse:
    """
    Test test test test
    """

    def __init__(self, pcm: np.ndarray = None, empty: bool = False):
        """
        Test test test test
        """

        pass

    def __cinit__(self, pcm=None, empty=False):

        self.PCM_ALLOCATED = False

        if pcm is not None:

            self.m, self.n = pcm.shape[0], pcm.shape[1]
            self.pcm = new cygf2_sparse(self.m,self.n)
            self.PCM_ALLOCATED = True

            if isinstance(pcm, np.ndarray):
                for i in range(self.m):
                    for j in range(self.n):
                        if pcm[i,j]!=0:
                            self.pcm.insert_entry(i,j,pcm[i,j])
            elif isinstance(pcm, scipy.sparse):
                rows, cols = pcm.nonzero()
                for i in range(len(rows)):
                    self.pcm.insert_entry(rows[i], cols[i], pcm[rows[i], cols[i]])
            else:
                raise Exception(f"InputError: The 'pcm' input parameter must be either a `numpy.ndarray` or a `scipy.sparse` matrix. Not {type(pcm)}")
        
        elif empty==False:
            raise ValueError("Please provide a parity check matric as input to this funciton.")
        
        else:
            pass

    cdef void c_object_init(self,cygf2_sparse* mat):
        self.pcm = mat
        self.PCM_ALLOCATED = True
        self.m = mat.m
        self.n = mat.n

    def __del__(self):
        if self.PCM_ALLOCATED:
            del self.pcm


    def lu_decomposition(self, reset_cols: bool = True, full_reduce: bool = False):

        """
        LU TEST IRUOJFKLJSLKJF
        """

        return self.pcm.lu_decomposition(reset_cols,full_reduce)

    def __repr__(self):

        out_matrix = np.zeros((self.m,self.n)).astype(int)
        cdef cygf2_entry *bpe
        for i in range(self.m):
            for j in range(self.n):
                bpe = self.pcm.get_entry(i,j)
                if not bpe.at_end():
                    # print(i,j,bpe.row_index, bpe.at_end())
                    out_matrix[i,j] = int(bpe.value)
        
        return str(out_matrix)


    def lu_solve(self, y):

        cdef int i

        cdef vector[uint8_t] input_y
        cdef vector[uint8_t] output_x
        output_x.resize(self.n)

        for i in range(self.m): input_y[i] = y[i]


        self.pcm.lu_solve(input_y,output_x)

        output = np.zeros(self.n).astype(int)

        for i in range(self.n): output[i] = output_x[i]

        return output

    def kernel(self):

        cdef cygf2_sparse* kern_cpp = self.pcm.kernel()

        kern = gf2sparse(empty=True)
        kern.c_object_init(kern_cpp)

        return kern

    def transpose(self):

        cdef cygf2_sparse* pcmT = self.pcm.transpose()

        py_pcmT = gf2sparse(empty=True)
        py_pcmT.c_object_init(pcmT)

        return py_pcmT

    @property
    def T(self):
        return self.transpose()

    @property
    def rank(self):
        return self.pcm.rank

    @property
    def rows(self):
        cdef int i
        out=np.zeros(self.m).astype(int)
        for i in range(self.m):
            out[i] = int(self.pcm.rows[i])
        return out

    @property
    def cols(self):
        cdef int i
        out=np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = int(self.pcm.cols[i])
        return out

    @property
    def inv_rows(self):
        cdef int i
        out=np.zeros(self.m).astype(int)
        for i in range(self.m):
            out[i] = int(self.pcm.inv_rows[i])
        return out

    @property
    def inv_cols(self):
        cdef int i
        out=np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = int(self.pcm.inv_cols[i])
        return out

    @property
    def L(self):

        cdef int m
        cdef int n
        cdef int i, j, ii
        cdef cygf2_sparse* L
        
        L=self.pcm.L

        m = L.m
        n = L.n
        out_matrix = np.zeros((m,n)).astype(int)
        cdef cygf2_entry *bpe
        for i in range(m):
            for j in range(n):
                bpe = L.get_entry(i,j)
                if not bpe.at_end():
                    ii=self.pcm.rows[i]
                    out_matrix[ii,j] = int(bpe.value)
        
        return out_matrix

    @property
    def U(self):

        cdef int m
        cdef int n, i, j, ii, jj
        cdef cygf2_sparse* U
        
        U=self.pcm.U

        m = U.m
        n = U.n
        out_matrix = np.zeros((m,n)).astype(int)
        cdef cygf2_entry *bpe
        for i in range(m):
            for j in range(n):
                bpe = U.get_entry(i,j)
                if not bpe.at_end():
                    ii = self.pcm.rows[i]
                    jj = self.pcm.cols[j]
                    out_matrix[ii,jj] = int(bpe.value)
        
        return out_matrix