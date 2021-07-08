import numpy as np

def permutation_matrix(n,shift):
    '''
    Outputs a size-n permutation matrix.


    Inputs:
        n: matrix dimension
        shift: the shift parameter
    Returns:
        mat: nxn matrix shifted by `shift' columns to the left
    '''
    return np.roll(np.identity(n),-1*shift,axis=1).astype(int)

def polynomial_to_circulant_matrix(n,non_zero_coefficients):
    '''
    Converts a polynomial into a circulant matrix

    Inputs:
        n: int, matrix dimension
        non_zero_coefficieents: list of ints (can be np.ndarray), list of the non-zero coefficients of the polynomial
    Returns:
        mat: an nxn circulant matrix corresponding to the inputted polynomial
    '''
    mat=np.zeros((n,n)).astype(int)
    for shift in non_zero_coefficients:
        mat+=permutation_matrix(n,shift)
    return mat % 2

def polynomial_transpose(lift_parameter, polynomial):

    polynomial_transpose=set()
    for coefficient in polynomial:
        polynomial_transpose.add((lift_parameter-coefficient)%lift_parameter)
    
    return polynomial_transpose

def empty_proto(shape):

    '''
    Returns an empty protograh
    '''

    m,n=shape
    row=np.array([{}]*n)
    return np.vstack([row]*m)

def kron_mat_proto(mat,proto):

    '''
    Kronecker product of a numpy matrix and a protograph
    '''

    mat_m,mat_n=mat.shape
    zero_proto=empty_proto(proto.shape)

    # print(zero_proto)

    out=[]
    for i in range(mat_m):
        row=[]
        for j in range(mat_n):
            if mat[i,j]:
                row.append(proto)
            else:
                row.append(zero_proto)
        out.append(np.hstack(row))

    return np.vstack(out)

def kron_proto_mat(proto,mat):

    '''
    Tensor product of a protograph and numpy matrix
    '''

    proto_m,proto_n=proto.shape
    mat_m,mat_n=mat.shape
    zero_proto=empty_proto(mat.shape)

    out=[]
    for i in range(proto_m):
        row=[]
        for j in range(proto_n):
            if len(proto[i,j])==0:
                row.append(zero_proto)
            else:
                temp=np.copy(zero_proto)
                for k in range(temp.shape[0]):
                    for l in range(temp.shape[1]):
                        if mat[k,l]==1:
                            temp[k,l]=proto[i,j]
                        else:
                            temp[k,l]={}
                
                row.append(temp)

        out.append(np.hstack(row))
    
    return np.vstack(out)


def protograph_transpose(lift_parameter,protograph):
    '''
    Returns the transpose of a protograph.

    Input:
        lift_paramter: int, the lift parameter for the protograph
        protograph: np.ndarray, the protograph
    Return:
        protograph_transpose: np.ndarray, the transpose of the inputted protograph for the given lift parameter
    '''

    m,n=protograph.shape
    protograph_transpose=empty_proto((n,m))

    for i in range(m):
        for j in range(n):
            coeff_list=set()
            # print(protograph_transpose[i,j])
            for coefficient in protograph[i,j]:
                # print(coefficient)
                coeff_list.add((lift_parameter-coefficient)%lift_parameter)

            protograph_transpose[j,i]=coeff_list

    return protograph_transpose


def protograph_to_qc_code(n,protograph):
    
    '''
    Generates the parity check matrix of a quasicyclic code from a matrix of polynomials.

    Inputs:
        n: int, lift parameter
        protograph: np.ndarray, polynomial matrix
    Returns:
        qc_matrix: np.ndarray, quasi-cyclic code corresponding to the inputted polynomial matrix
    '''
    
    qc_matrix=[]
    
    for row in protograph:
        qc_row=[]
        for polynomial in row:
            qc_row.append( polynomial_to_circulant_matrix(n,polynomial) )
        qc_row=np.hstack(qc_row)
        qc_matrix.append(qc_row)
        
    qc_matrix=np.vstack(qc_matrix)
    return qc_matrix


