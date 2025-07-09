from collections import defaultdict
import numpy as np


class G:
    """
    G is a group described in terms of its Cayley (multiplication) table.

    The group elements are represented by integers from 0 to n-1, where
    `n` is the order of the group. The identity element is assumed to be 0.
    """
    def __init__(self, table):
        """
        Initialize the group.

        :param table: A square NumPy array representing the Cayley table
                      of the group. Entry (i, j) gives the product of
                      group elements i and j. The identity element must
                      be at index 0.
        """
        self.table = table
        self.n = table.shape[0]
        self.pairs = self.get_pairs()

    def is_abelian(self):
        """
        Check if the group is Abelian (i.e., commutative).

        :return: True if the Cayley table is symmetric, False otherwise.
        """
        return np.array_equal(self.table, self.table.T)

    def inv(self, el):
        """
        Find the inverse of a given group element.

        :param el: Integer index of the group element.
        :return: Integer index of the inverse element, i.e., the unique element `x`
                such that table[el, x] == 0 (the identity).

        """
        row = self.table[el]
        return np.where(row == 0)[0][0]

    def op(self, i, j):

        """
        Perform the group operation on two elements.

        :param i: Integer index of the first element.
        :param j: Integer index of the second element.
        :return: Integer index of the product i * j.
        """
        return int(self.table[i, j])

    def ord(self, el):
        """
        :param el: Integer index of the element.
        :return: Order of the element (smallest positive k such that el^k = 0).
        """
        res = el
        order = 1
        while res != 0:
            res = self.op(el, res)
            order += 1
        return order

    def right_cayley(self, set_a):
        """
        :param set_a: Subset of group elements (by index) used as right generators.
        :return: Set of undirected edges in the right Cayley graph.
        """

        edges = set()
        # iterates all elements
        for g in range(self.n):
            for a in set_a:
                edges.add((g, self.op(g, a)))
        edges = {(min(ed), max(ed)) for ed in edges}
        return edges

    def left_cayley(self, set_a):
        """
        :param set_a: Subset of group elements (by index) used as left generators.
        :return: Set of undirected edges in the left Cayley graph.
        """
        edges = set()
        # iterates all elements
        for g in range(self.n):
            for a in set_a:
                edges.add((g, self.op(a, g)))
        edges = {(min(ed), max(ed)) for ed in edges}
        return edges

    def get_pairs(self):
        """
        :return: Dictionary mapping element order k to list of inverse pairs (i, j).
        """

        pairs = set()
        for i in range(self.n):
            pair = tuple(sorted({i, self.inv(i)}))
            pairs.add(pair)
        order_and_pairs = defaultdict(list)
        for pair in pairs:
            k = self.ord(pair[0])
            order_and_pairs[k].append(pair)
        return order_and_pairs


def cyclic_group(n):
    """
    :param n: Order of the cyclic group.
    :return: Cayley table of the cyclic group of order n (Z/nZ).
    """

    row = np.array([i for i in range(n)])
    table = []
    for i in range(n, 0, -1):
        table.append(np.roll(row, i))
    return np.vstack(table)


def direct_product(table_1, table_2):
    """
    :param table_1: Cayley table of the first group (NumPy array).
    :param table_2: Cayley table of the second group (NumPy array).
    :return: Cayley table of the direct product group.

    Table of Z3 x Z2 has [0, 1, 2, 3, 4, 5] --> [00, 01, 10, 11, 20, 21]
    Table of (Z3 x Z2) X Z2 --> [000, 001, 010, 011, 100, 101, 110, 111, 200, 201, 210, 211]
    [0 of first ]x i of second -- [1 of first ]x i of second -- etc.
    my_ord = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                my_ord.append((i, j, k))

    """

    n_1 = table_1.shape[0]
    n_2 = table_2.shape[1]

    def small_indices(el):
        i_1 = el // n_2
        i_2 = el % n_2
        return i_1, i_2

    def big_index(i_1, i_2):
        return i_1 * n_2 + i_2

    n = n_1 * n_2
    table = np.zeros([n, n]).astype(int)
    for g_i in range(n):
        for g_j in range(n):
            g_i_1, g_i_2 = small_indices(g_i)
            g_j_1, g_j_2 = small_indices(g_j)
            one = table_1[g_i_1, g_j_1]
            two = table_2[g_i_2, g_j_2]
            table[g_i, g_j] = big_index(one, two)

    return table


def from_tuple_to_direct_product_index(table_1, table_2, el):
    """
    :param table_1: Cayley table of the first group.
    :param table_2: Cayley table of the second group.
    :param el: Tuple (i, j) representing the element in the product group.
    :return: Flat index in the Cayley table of the direct product.
    """
    # n_1 = table_1.shape[0]
    n_2 = table_2.shape[0]
    return el[0] * n_2 + el[1]


if __name__ == '__main__':
    from group_tables import klein_table, quaternion
    # Create klein group
    g_klein = G(klein_table)
    # create edges of a right and a left cayley graph of g_klein
    klein_right = g_klein.right_cayley({1})
    klein_left = g_klein.left_cayley({2})

    # Create quaternion group and a left cayley
    q8 = G(quaternion)

    quaternion_left = q8.left_cayley({1, 5})
    print('Print edges of a left Cayley graph for the Quaternion group')
    print(quaternion_left)

    # this is the ordering that is used in direct product
    my_ord = []
    for i_ in range(4):
        for j_ in range(4):
            for k_ in range(2):
                my_ord.append((i_, j_, k_))

