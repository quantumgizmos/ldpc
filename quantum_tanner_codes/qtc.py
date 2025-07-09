import numpy as np
import networkx as nx
from collections import defaultdict
from auxiliary.groups import G, from_tuple_to_direct_product_index, cyclic_group, direct_product
from auxiliary.binary_utils import gaussian_reduction, get_support


def from_two_pc_to_x_and_z_gen(h_1, h_2, debug=False):
    """

    Constructs X and Z stabilizer generators for a product code from two parity-check matrices.

    :param h_1: Parity-check matrix of the first classical code.
    :param h_2: Parity-check matrix of the second classical code.
    :param debug: If True, prints intermediate matrices for debugging.
    :return: Tuple (x, z), where x is the X-generator matrix and z is the Z-generator matrix.

    gen for X code is g_1 x g_2
    gen for Z code is h_1 x h_2

    """
    g_1 = gaussian_reduction(h_1)['ker']
    g_2 = gaussian_reduction(h_2)['ker']
    x = np.kron(g_1, g_2)
    z = np.kron(h_1, h_2)
    if debug:
        print('g_1\n', g_1)
        print('g_2\n', g_2)
        print('h_1\n', h_1)
        print('h_2\n', h_2)
    return x, z


class Qtc:
    """
    A quantum tensor code (QTC) object constructed from a group G and two inverse-closed generating sets A and B.
    It builds a Cayley graph whose vertices are labeled by G and edges come from left and right Cayley graphs.

    The graph must be bipartite and connected to allow X and Z vertex assignment.

    The number of "squares" (g, ag, gb, agb) is G * Delta^2 / 4.
    """

    def __init__(self, group, a_set=None, b_set=None, default_x_and_z=True):
        """
        :param group: Instance of G class (group described by Cayley table), identity element is 0.
        :param a_set: Inverse-closed subset of G used for right Cayley graph edges.
        :param b_set: Inverse-closed subset of G used for left Cayley graph edges.
        :param default_x_and_z: If True and the graph is bipartite and connected, assign vertex sets of
        nx deafult bipartition to X and Z.
        """

        # a_set and b_set are ordered and inverse closed
        self.group = group
        self.a_set = a_set
        self.b_set = b_set

        # length of set_a and set_b gives the vertex degree of the graph, also the length of the inner code
        self.delta_a = len(a_set)
        self.delta_b = len(b_set)
        self.small_n = self.delta_a * self.delta_b

        # builds the right cayley graph with edges (g, ga); is a set of edges
        self.cay_a = group.right_cayley(a_set)
        self.cay_b = group.left_cayley(b_set)
        # as the left and right cayley graphs are defined on the same vertices, we build the total graph as
        # nx object
        graph = nx.Graph()
        graph.add_edges_from(self.cay_a.union(self.cay_b))
        self.graph = graph
        # CHECK
        assert self.group.n == graph.number_of_nodes()
        self.n_of_vertices = graph.number_of_nodes()
        # the graph needs to be bipartite, X and Z vertices are the vertices in the partition
        # also, we want it connected
        self.x_vertices = None
        self.z_vertices = None

        if default_x_and_z and nx.is_bipartite(graph) and nx.is_connected(graph):
            self.x_vertices, self.z_vertices = nx.bipartite.sets(graph)

        self.v_inner_ordering = None

        # squares are all the quadruple (g, ag, agb, gb)
        # self.squares is a list of quadruple of type tuple(sorted([v, va, bv, bva]))
        # self.four_corners is true iff all squares have 4 corners
        self.squares, self.four_corners = self.find_squares()
        # the number of squares should be G * Delta ^ 2 / 4
        self.n_of_squares = len(self.squares)
        # indexed_squares is a dictionary where
        # keys is a square (v, va, bv, bva) and value is a number 0, . . , n_of_squares
        self.square_to_index, self.vertex_to_incident_squares = self.index_squares_and_map_vertices()

    def __repr__(self):
        """
        :return: String summary of group size, number of squares, and generator set sizes.
        """
        return (f'size of G: {self.graph.number_of_nodes()}, number of squares {self.n_of_squares}\n'
                f'inner code: {self.small_n}, size A: {self.delta_a}, size B: {self.delta_b}')

    def find_squares(self):
        """
        Finds all squares of the form (v, va, bv, bva) and assigns them to each vertex v in self.v_inner_ordering.

        :return: A tuple (squares, four_corners):
                 - squares: list of tuples, each a square as sorted 4-tuple of vertex indices.
                 - four_corners: True if all squares have 4 distinct corners.
        """

        # for each vertex v, its small_n edges needs to be ordered so that we can put the small code constraints on it
        # so v_inner_ordering it's a dictionary with v, the vertex as a key, and value is a dictionary
        # {0: sq_i, 1: sq_j, . . . , small_n - 1: sq_k}
        v_inner_ordering = {v: {inner: None for inner in range(self.small_n)} for v in self.graph}

        four_corners = True

        set_of_squares = set()

        # iterate vertices
        # update the v_inner_ordering[v]: from 0 to small_n -1, location is a square expressed as tupla
        for v in self.graph:
            # we create a square, given by a tuple (a, b) with a in set_a and b in set_b

            for i_a, a in enumerate(self.a_set):
                for i_b, b in enumerate(self.b_set):
                    va = self.group.op(v, a)
                    bv = self.group.op(b, v)
                    bva = self.group.op(bv, a)
                    square = tuple(sorted([v, va, bv, bva]))
                    if len(set(square)) != 4:
                        four_corners = False
                    # print('square ', square)
                    # print(f'a {a}, b {b}, v {v}')
                    v_inner_ordering[v][self.delta_b * i_a + i_b] = square
                    # idx += 1
                    set_of_squares.add(square)
        # for k, c in v_inner_ordering.items():
        #     print(k)
        #     print(c)
        self.v_inner_ordering = v_inner_ordering

        return list(set_of_squares), four_corners

    def index_squares_and_map_vertices(self):

        """
        Indexes each square and maps vertices to their incident squares.

        Also modifies self.v_inner_ordering so that squares are represented by their index.

        :return: Tuple (square_to_index, vertex_to_incident_squares)
                 - square_to_index: dict mapping square (4-tuple) to unique integer index.
                 - vertex_to_incident_squares: dict mapping vertex to list of square indices it belongs to.

        """
        square_to_index = {sq: idx for idx, sq in enumerate(self.squares)}
        # now each square has a unique index assigned to it

        # adj vertex to square
        vertex_to_incident_squares = defaultdict(list)

        for sq, idx in square_to_index.items():
            for v in sq:
                vertex_to_incident_squares[v].append(idx)

        # for ech vertex v
        # temp[v] is a dictionary with k an integer from 0 . . . small_n-1
        temp = defaultdict()
        for v, local_view in self.v_inner_ordering.items():
            temp[v] = {k: square_to_index[c] for k, c in local_view.items()}

        # for k, c in temp.items():
        #     print(k, c)
        #     print(self.v_inner_ordering[k])
        #     print('\n')
        self.v_inner_ordering = temp

        return square_to_index, vertex_to_incident_squares

    def get_a_paritycheck(self, vertex_set, gen):
        """
        Constructs a parity-check matrix from a subset of vertices and a list of generator vectors.

        :param vertex_set: Set of vertices to assign generator constraints to.
        :param gen: List of generator vectors (each a binary array).
        :return: A NumPy array representing the parity-check matrix.
        """
        matrix = []
        for v in vertex_set:
            # for each generator
            for i, g in enumerate(gen):
                row = np.zeros(self.n_of_squares).astype(int)
                for inner_el in get_support(g):
                    row[self.v_inner_ordering[v][inner_el]] = 1
                matrix.append(row)
        return np.vstack(matrix)


if __name__ == '__main__':
    # Build the rotated toric code as a Quantum Tanner Code
    # n_ = 4 gives error as the logical i.e. line is counted as square as well
    n_ = 6
    c_6 = cyclic_group(n_)
    c_6_times_c_6 = direct_product(c_6, c_6)
    group_6_6 = G(c_6_times_c_6)

    # a_set is inverse close of {(0, 1)}
    # b_set is inverse close of {(1, 0)}

    print(f'The group index of the element (0,1) is: {from_tuple_to_direct_product_index(c_6, c_6, (0,1))}'
          f' and its inverse is: {group_6_6.inv(1)}')

    print(f'The group index of the element (1,0) is: {from_tuple_to_direct_product_index(c_6, c_6, (1,0))}'
          f' and its inverse is: {group_6_6.inv(6)}')

    # a = (0, 1) = 1 w inv 5 and b = (1, 0) = 6 w inv 30
    print('Quantum Tanner Code')
    qtc = Qtc(group_6_6, a_set={1, 5}, b_set={30, 6})
    print(qtc)
    print('We now build the pc matrices of the inner codes')
    # The pcm of the inner codes are A \otimes B and A dual \otimes B dual
    # For the Rotated Toric Code, A = B = [1, 1]
    code_a = np.array([[1, 1]])
    code_b = np.array([[1, 1]])
    x_code, z_code = from_two_pc_to_x_and_z_gen(code_a, code_b)
    print(f'Inner code for X stabilisers: {x_code}\nInner code for Z stabilisers: {z_code}')
    print('We now build the pc matrices for the Rotated Toric code as Quantum Tanner Code')
    hx_ = qtc.get_a_paritycheck(qtc.x_vertices, x_code)
    hz_ = qtc.get_a_paritycheck(qtc.z_vertices, z_code)
    print(f'hx:\n{hx_}\nhz:\n{hz_}')
