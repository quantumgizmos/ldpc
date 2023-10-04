import numpy as np
import pandas as pd
from tqdm import tqdm

'''
Belief progagation decoder for binary codes
Syndrome decoding version.
Follows method outlined in Mackay. Implementation also closely follows python pyldpc module
'''


class ParityCheckMatrix:

    def __init__(self, H):
        self.shape = np.shape(H)
        self.m, self.n = self.shape

        self.H = H

        self.k = self.n - self.m
        self.bit_neighbourhoods, self.check_neighbourhoods = self.calc_neighbourhoods()

    def calc_neighbourhoods(self):
        bit_neighbourhoods = []
        check_neighbourhoods = []

        max_p = max([self.m, self.n])

        for i in range(max_p):
            if i < self.n:
                column = self.H[:, i]
                bit_neighbourhoods.append(np.nonzero(column)[0])

            if i < self.m:
                row = self.H[i]
                check_neighbourhoods.append(np.nonzero(row)[0])

        return [bit_neighbourhoods, check_neighbourhoods]

    def __repr__(self):
        return np.array2string(self.H)


def p0(R):
    if R >= 1000:
        return 1.0
    elif R <= -1000:
        return 0.0
    else:
        return (2 ** R) / (1 + 2 ** R)


def bp_decoder(H, z, p, iterations, full_diagnostics=False):
    """
    Input H: Parity check matrix
    Type: "parity_check_matrix" OR "np.ndarray"
    Code below: checks whether H is (Type: "parity_check_matrix" OR "np.ndarray").
    Converts to Type "parity_check_matrix" if Type: "np.ndarray". Else returns error message.
    """
    # checks whether the input "H" is a compatible type ("type: parity_check_matrix" or "type:np.ndarray")
    if isinstance(H, np.ndarray):
        H = ParityCheckMatrix(H)

    '''
    Input z: The code syndrome
    Type: "np.ndarray"
    '''
    if not isinstance(z, np.ndarray):
        raise Exception("Input 'z' should have type 'numpy.ndarray'")

    '''Code Parameters'''
    h = H.H  # parity check matrix
    n = H.n  # no. bits
    m = H.m  # no. checks
    Mn = H.bit_neighbourhoods  # Length: n. The checks that each bit is connected to.
    Nm = H.check_neighbourhoods  # Length:m. The bits that each check is connected to.

    '''
    Check whether syndrome is the correct length for the code
    '''
    if len(z) != m:
        raise Exception("Input syndrome 'z' is the incorrect length for parity check matrix 'H'")

    '''
    Belief propagation decoder: main algorithm
    Log-version
    '''

    '''
    Initialisation of the belief propagation decoder
    '''
    logP = np.log((1 - p) / p)

    logQ_0 = logP * h
    logQ = logQ_0

    recovery_operations = []

    for it in range(iterations):
        '''
        Horizontal-step
        '''

        dQ = np.tanh(logQ / 2)

        dR = np.zeros((m, n), dtype="float64")
        logR = np.zeros((m, n), dtype="float64")

        for i in range(m):

            for j in Nm[i]:
                dR[i, j] = (-1) ** z[i]
                for k in Nm[i]:
                    if k != j:
                        dR[i, j] *= dQ[i, k]

                nX = (1 + dR[i, j])
                dX = (1 - dR[i, j])
                if nX == 0:

                    logR[i, j] = -1000
                elif dX == 0:

                    logR[i, j] = 1000
                else:
                    logR[i, j] = np.log(nX / dX)

        ''' Vertical Step '''
        for j in range(n):
            for i in Mn[j]:
                Mji = np.array([k for k in Mn[j] if k != i]).astype(int)

                logQ[i, j] = logP + np.sum(logR[Mji, j])

        posterior_probability = np.zeros(n)
        log_posterior = np.zeros(n)

        for j in range(n):
            log_posterior[j] = logP + np.sum(logR[Mn[j], j])

        x = np.array(log_posterior <= 0).astype(int)

        pseudo_syndrome = (h @ x) % 2

        recovery_operations.append(np.nonzero(x)[0])

        '''
        Exit iteration if BP decoder has converged
        '''
        # print([it,pseudo_syndrome,z])
        if np.array_equal(pseudo_syndrome, z):
            # print(it)
            return [True, x, 1 / (1 + np.exp(log_posterior)), it]

    '''
    Return false flag if decoder does not converge
    '''
    return [False, x, 1 / (1 + np.exp(log_posterior)), it]



class PyBp():

    def __init__(self,H, error_rate, max_iter):

        self.H = ParityCheckMatrix(H)
        self.error_rate = error_rate
        self.max_iter = max_iter
        self.converge = False

    def decode(self,syndrome):

        decoding = bp_decoder(self.H,syndrome,self.error_rate,self.max_iter)
        if decoding[0]:
            self.converge = True
        else:
            self.converge = False

        return decoding[1]
