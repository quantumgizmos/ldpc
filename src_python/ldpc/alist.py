import numpy as np

def save_alist(name, mat, j=None, k=None):

    '''
    Saves a numpy array as an alist file.

    Parameters
    ----------
    name : string
        The output filename. 
    mat : numpy.ndarray
        The input np.ndarray.
    j : int, optional
        The maximum column weight of the matrix to be save.
    k : int, optional
        The maximum row-weight of the matrix to the saved.
    
    Returns
    -------
    f : file
        Function saves an alist file to disk.
    
    '''
    
    H=np.copy(mat)
    H=H.T
    
    if j is None:
        j=int(max(H.sum(axis=0)))


    if k is None:
        k=int(max(H.sum(axis=1)))


    m, n = H.shape # rows, cols
    f = open(name, 'w')
    print(n, m, file=f)
    print(j, k, file=f)

    for col in range(n):
        print( int(H[:, col].sum()), end=" ", file=f)
    print(file=f)
    for row in range(m):
        print( int(H[row, :].sum()), end=" ", file=f)
    print(file=f)

    for col in range(n):
        for row in range(m):
            if H[row, col]:
                print( row+1, end=" ", file=f)
        print(file=f)

    for row in range(m):
        for col in range(n):
            if H[row, col]:
                print(col+1, end=" ", file=f)
        print(file=f)
    f.close()

def numpy2alist(name, mat, j=None, k=None):
    return save_alist(name, mat, j, k)

def alist2numpy(fname):
    alist_file = np.loadtxt(fname, delimiter='\n',dtype=str)
    matrix_dimensions=alist_file[0].split()
    m=int(matrix_dimensions[0])
    n=int(matrix_dimensions[1])

    mat=np.zeros((m,n)).astype(int)
    
    for i in range(m):
        columns=[]
        for item in alist_file[i+4].split():
            if item.isdigit(): columns.append(item)
        columns=np.array(columns).astype(int)
        columns=columns-1 #convert to zero indexing
        mat[i,columns]=1

    return mat
