import numpy as np
from .mod2 import mod10_to_mod2


def hamming(rank):
    """ Outputs the Hamming code parity check matrix given its rank """
    rank = int(rank)
    num_rows = (2 ** rank) - 1

    pc_matrix = np.zeros((num_rows, rank), dtype=int)

    for i in range(0, num_rows):
        pc_matrix[i] = mod10_to_mod2(i + 1, rank)

    return pc_matrix.T


def rep_code(num_reps, standard_form=True):
    """ Outputs repetition code parity check matrix given number of repetitions (code distance) """

    pc_matrix = np.zeros((num_reps - 1, num_reps), dtype=int)

    if standard_form:

        for i in range(num_reps - 1):
            pc_matrix[i, i] = 1
            pc_matrix[i, num_reps - 1] = 1

    else:

        for i in range(num_reps - 1):
            pc_matrix[i, i] = 1
            pc_matrix[i, i + 1] = 1

    return pc_matrix


def ring_code(girth):
    """ Outputs the ring code parity check matrix for a given code distance (girth)"""

    pc_matrix = np.zeros((girth, girth), dtype=int)

    for i in range(girth - 1):
        pc_matrix[i, i] = 1
        pc_matrix[i, i + 1] = 1

    # close the loop
    i = girth - 1
    pc_matrix[i, 0] = 1
    pc_matrix[i, i] = 1

    return pc_matrix