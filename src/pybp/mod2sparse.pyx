#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

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
