from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

"""
This class is not used right now, but may be used in the future.
"""


class GraphQtc:
    """
    A quantum tensor code (QTC) object built from a bipartite graph.

    The graph must be bipartite and connected to allow X and Z vertex assignment.

    The number of "squares" (4-cycles) is |G| * Δ² / 4, where |G| is the number of vertices and Δ is the degree.
    """

    def __init__(self, graph, default_x_z=True):
        """
        :param graph: networkx.Graph, assumed bipartite and undirected.
        :param default_x_z: If True and the graph is bipartite, assign nx default bipartition to X and Z.
        """
        self.graph = graph
        self.n_of_vertices = graph.number_of_nodes()

        self.x_vertices = None
        self.z_vertices = None

        if default_x_z:
            assert nx.is_bipartite(graph)
            self.x_vertices, self.z_vertices = nx.bipartite.sets(graph)

        self.squares = self.find_squares()
        self.n_of_squares = len(self.squares)

        self.indexed_squares, self.vertex_to_squares = self.index_squares_and_map_vertices()

    def __repr__(self):
        return f'size of G: {self.graph.number_of_nodes()}, number of squares {self.n_of_squares}'

    def find_squares(self):
        """
        :return: Set of 4-tuples representing all squares (4-cycles) in the graph.
        """
        set_of_squares = set()
        neighbors = {v: set(self.graph[v]) for v in self.graph}

        for v in self.graph:
            for v1, v2 in combinations(neighbors[v], 2):
                assert v2 not in neighbors[v1], 'bipartite problem'
                common = (neighbors[v1] & neighbors[v2]) - {v}
                for w in common:
                    square = tuple(sorted([v, v1, w, v2]))
                    set_of_squares.add(square)

        return set_of_squares

    def reset_squares(self, new_squares):
        """
        :param new_squares: Set of 4-tuples representing squares to replace the current square set.
        """
        self.squares = new_squares
        self.n_of_squares = len(self.squares)
        self.indexed_squares, self.vertex_to_squares = self.index_squares_and_map_vertices()

    def filter_squares(self, edge_set):
        """
        :param edge_set: Set of undirected edges (as tuples).
        :return: Subset of squares that contain exactly two edges in edge_set.
        """
        good_ones = set()
        for square in self.squares:
            flag = 0
            subgraph = self.graph.subgraph(square)
            for edge in list(subgraph.edges):
                if edge in edge_set or (edge[1], edge[0]) in edge_set:
                    flag += 1
            if flag == 2:
                good_ones.add(square)
        return good_ones

    def index_squares_and_map_vertices(self):
        """
        :return:
            - Dictionary mapping frozen sets of square vertices to indices.
            - Dictionary mapping each vertex to a list of square indices it appears in.
        """
        square_to_index = {frozenset(sq): idx for idx, sq in enumerate(self.squares)}
        vertex_to_squares_neigh = defaultdict(list)

        for sq_set, idx in square_to_index.items():
            for vertex in sq_set:
                vertex_to_squares_neigh[vertex].append(idx)

        return square_to_index, vertex_to_squares_neigh

    def get_a_paritycheck(self, vertex_set, gen):
        """
        :param vertex_set: Iterable of vertices (typically all X or all Z).
        :param gen: List of binary lists representing generator vectors.
        :return: Parity-check matrix as NumPy array with one row per (vertex, generator) pair.
        """
        matrix = []
        for v in vertex_set:
            current_support = self.vertex_to_squares[v]
            for g in gen:
                row = np.zeros(self.n_of_squares).astype(int)
                for index, label in enumerate(current_support):
                    row[label] = g[index]
                matrix.append(row)
        return np.vstack(matrix)
