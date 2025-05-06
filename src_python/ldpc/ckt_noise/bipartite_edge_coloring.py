# Copyright 2025 The Project Authors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Dict, Set
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np


@dataclass
class Node:
    # The neighboring edges of the current node u which have not been colored.
    # Each element of uncolored_edges is the index of the other node v in the edge (u, v).
    uncolored_edges: Set[int] = field(default_factory=set)
    # The neighboring edges of the current node u which have been colored, as a dictionary.
    # The keys are the colors (integers) and each value is the index of the other node.
    # i.e. `self.colored_edges[i] == j` means there is an edge (u, j) with color i.
    colored_edges: Dict[int, int] = field(default_factory=dict)
    # Which colors are not yet used at this node
    colors_available: Set[int] = field(default_factory=set)

    def add_color_to_uncolored_edge(self, dest: int, col: int) -> None:
        """Color edge (self, dest) as col, assuming edge is currently
        uncolored. Only changes edge color information at this node."""
        self.colored_edges[col] = dest
        self.colors_available.discard(col)
        self.uncolored_edges.discard(dest)

    def swap_edge_colors(self, color_1: int, color_2: int) -> None:
        """Swap the color of the edges with color_1 and color_2.
        Only changes edge color information at this node."""
        col1_dest = self.colored_edges[color_1]
        col2_dest = self.colored_edges[color_2]
        self.colored_edges[color_1] = col2_dest
        self.colored_edges[color_2] = col1_dest

    def change_edge_color(self, dest: int, from_col: int, to_col: int) -> None:
        """Remove color from_col and add color to_col for edge (self, dest).
        Only changes edge color information at this node."""
        assert self.colored_edges[from_col] == dest
        del self.colored_edges[from_col]
        self.colors_available.add(from_col)
        self.colored_edges[to_col] = dest
        self.colors_available.discard(to_col)
        self.uncolored_edges.discard(dest)


class BipartiteGraph:
    def __init__(self, num_a_nodes: int, num_b_nodes: int):
        self.a_nodes = [Node() for _ in range(num_a_nodes)]
        self.b_nodes = [Node() for _ in range(num_b_nodes)]
        self.degree = 0

    def add_edge(self, i: int, j: int):
        """Add node b_j as a neighbor of node a_i

        Parameters
        ----------
        i : int
            The index of node a_i
        j : int
            The index of node b_j
        """
        self.a_nodes[i].uncolored_edges.add(j)
        self.b_nodes[j].uncolored_edges.add(i)

    @staticmethod
    def from_biadjacency_matrix(biadj: csr_matrix) -> "BipartiteGraph":
        """Construct a bipartite graph with node set A and node set B,
        from a biadjacency matrix `biadj`.

        Row i of `biadj` is node a_i in the A set, and column j of `biadj` is
        node b_j in the B set. Node a_i is connected to node b_j if and
        only if `biadj[i,j]==1` (otherwise `biadj[i,j]==0`).

        Parameters
        ----------
        biadj : csr_matrix
            The biadjacency matrix used to define the bipartite graph.
            Should have binary elements (integers, either 0 or 1)

        Returns
        -------
        BipartiteGraph
            The BipartiteGraph corresonding to the biadjacency matrix biadj
        """
        biadj = csr_matrix(biadj)
        graph = BipartiteGraph(num_a_nodes=biadj.shape[0], num_b_nodes=biadj.shape[1])

        for i in range(biadj.shape[0]):
            for j in biadj.indices[biadj.indptr[i] : biadj.indptr[i + 1]]:
                graph.add_edge(i=i, j=j)

        max_deg_a = max(
            len(n.uncolored_edges) + len(n.colored_edges) for n in graph.a_nodes
        )
        max_deg_b = max(
            len(n.uncolored_edges) + len(n.colored_edges) for n in graph.b_nodes
        )
        # Chromatic index is just the degree of the graph
        num_colors = max(max_deg_a, max_deg_b)

        graph.degree = num_colors

        for a_node in graph.a_nodes:
            a_node.colors_available = set(range(num_colors))

        for b_node in graph.b_nodes:
            b_node.colors_available = set(range(num_colors))
        return graph

    def assert_has_edge_coloring(self):
        for i, a_node in enumerate(self.a_nodes):
            assert len(a_node.uncolored_edges) == 0
            num_edges = len(a_node.colored_edges)
            assert (
                num_edges
                == len(set(a_node.colored_edges.keys()))
                == len(set(a_node.colored_edges.items()))
            )
            for c, n in a_node.colored_edges.items():
                b = self.b_nodes[n]
                assert b.colored_edges[c] == i

        for i, b_node in enumerate(self.b_nodes):
            assert len(b_node.uncolored_edges) == 0
            num_edges = len(b_node.colored_edges)
            assert (
                num_edges
                == len(set(b_node.colored_edges.keys()))
                == len(set(b_node.colored_edges.items()))
            )
            for c, n in b_node.colored_edges.items():
                a = self.a_nodes[n]
                assert a.colored_edges[c] == i

    def bipartite_edge_coloring(self):
        """Find an edge coloring of the bipartite
        graph. The number of colors (chromatic index) is simply the degree
        of the graph. An edge coloring of a graph is an assignment of a color
        to each edge, such that no node has more than one edge of a given color.

        Uses a method based on augmenting paths (the second algorithm
        reviewed in https://dl.acm.org/doi/pdf/10.1145/800133.804346).
        For a bipartite graph with E edges and V nodes, the worst-case
        complexity is O(EV), although empirically it is closer
        to O(E) for many graphs (e.g. random bipartite graphs with
        constant maximum degree).

        Parameters
        ----------
        graph : BipartiteGraph
            The graph to edge-color
        """
        graph = self

        for i in range(len(graph.a_nodes)):
            a_i = graph.a_nodes[i]
            for j in sorted(a_i.uncolored_edges):
                b_j = graph.b_nodes[j]
                # Colors available for both nodes of this edge
                edge_cols_available = a_i.colors_available & b_j.colors_available

                if len(edge_cols_available) > 0:
                    c = min(edge_cols_available)
                    a_i.add_color_to_uncolored_edge(dest=j, col=c)
                    b_j.add_color_to_uncolored_edge(dest=i, col=c)
                else:
                    # No valid color found.
                    # We pick a color that is available on one side, let's say b_j.
                    c = min(b_j.colors_available)
                    assert c not in b_j.colored_edges
                    # This color c is not available for a_i, so we find an alternating
                    # path c->d->c->d where d is an available color for a_i and swap c<->d
                    d = min(a_i.colors_available)
                    assert d not in a_i.colored_edges

                    curr_a = a_i
                    curr_a_ind = i
                    curr_b = b_j
                    curr_b_ind = j
                    # Commit to color c for this edge (curr_a, curr_b)
                    # Update node curr_b with this color information
                    curr_b.add_color_to_uncolored_edge(dest=curr_a_ind, col=c)
                    # At curr_a we initially give the edge (curr_a, curr_b) the
                    # wrong color assignment of d, just as an initial condition
                    # for the while loop
                    curr_a.add_color_to_uncolored_edge(dest=curr_b_ind, col=d)

                    # Find an alternating path of edges colored c and d in the order
                    # (curr_b, curr_a, next_b, next_a, ...), inverting the colors as we go
                    while True:
                        if c in curr_a.colors_available:
                            curr_a.change_edge_color(
                                dest=curr_b_ind, from_col=d, to_col=c
                            )
                            break
                        next_b_ind = curr_a.colored_edges[c]
                        next_b = graph.b_nodes[next_b_ind]
                        # Swap coloring at curr_a
                        curr_a.swap_edge_colors(color_1=c, color_2=d)
                        if d in next_b.colors_available:
                            next_b.change_edge_color(
                                dest=curr_a_ind, from_col=c, to_col=d
                            )
                            break
                        next_a_ind = next_b.colored_edges[d]
                        next_a = graph.a_nodes[next_a_ind]
                        # Swap coloring at next_b
                        next_b.swap_edge_colors(color_1=c, color_2=d)
                        # Update current a and b
                        curr_a = next_a
                        curr_a_ind = next_a_ind
                        curr_b_ind = next_b_ind

    def to_biadjacency_matrix(self) -> csr_matrix:
        """Convert to a csr_matrix biadjacency matrix with each row an A node
        and each column a B node. The nonzero elements correspond to the edges
        in the bipartite graph, but rather than all being 1, their value
        specifies the color of the edge. The color c is in the range
        `1 <= c <= BipartiteGraph.degree`, where `BipartiteGraph.degree`
        is the chromatic index.
        If the edge is uncolored, the corresponding element in the matrix is
        given the value -1.

        Returns
        -------
        csr_matrix
            A biadjacency matrix M representing the bipartite graph where the
            element `M[i,j]` is nonzero if and only if there is an edge from
            `BipartiteGraph.a_nodes[i]` to `BipartiteGraph.b_nodes[j]`.
            The value of `M[i,j]` is its color in the BipartiteGraph + 1.
            In other words we have `1 <= M[i,j] <= BipartiteGraph.degree`,
            unless an edge is uncolored in which case `M[i,j] == -1`.
        """
        num_rows = len(self.a_nodes)
        num_cols = len(self.b_nodes)
        elements = []
        for i in range(len(self.a_nodes)):
            ai = self.a_nodes[i]
            for c, j in ai.colored_edges.items():
                elements.append((i, j, c + 1))
            for j in ai.uncolored_edges:
                elements.append((i, j, -1))

        row_ind, col_ind, data = zip(*elements)
        biadj = csr_matrix(
            (np.array(data, dtype=np.int64), (np.array(row_ind), np.array(col_ind))),
            shape=(num_rows, num_cols),
            dtype=np.int64,
        )
        return biadj


def is_valid_bipartite_edge_coloring(
    biadj_matrix: csr_matrix, colored_biadj_matrix: csr_matrix
) -> bool:
    """Checks whether `colored_biadj_matrix` is a valid minimum edge coloring of
    the bipartite graph defined by biadj_matrix.

    Parameters
    ----------
    biadj_matrix : csr_matrix[np.uint8]
        A biadjacency matrix defining a bipartite graph.
        For a bipartite graph with node sets A and B, each row of biadjacency_matrix
        corresponds to a node in A, and each column of biadjacency matrix corresponds
        to a node in B. Element `biadjacency_matrix[i,j]==1` if and only if node
        i in A is connected by an edge to node j in B, otherwise `biadjacency_matrix[i,j]==0`.
    colored_biadj_matrix : csr_matrix[np.int64]
        A biadjacency matrix representing the bipartite graph where the
        element `colored_biadj_matrix[i,j]` should be nonzero if and only if there
        is an edge from node i in the A set to node j in the B set.
        The value of `colored_biadj_matrix[i,j]` is the edge's color in the graph.
        In other words, if it is a valid minimum edge coloring, we should have
        `1 <= M[i,j] <= degree`, where degree is the maximum degree of the
        bipartite graph.

    Returns
    -------
    bool
        True if colored_biadj_matrix is a valid minimum edge coloring of the bipartite
        graph defined by biadj_matrix, otherwise False
    """
    m = csr_matrix(biadj_matrix)
    m.eliminate_zeros()
    m.sort_indices()
    m_csr = csr_matrix(colored_biadj_matrix)
    m_csr.eliminate_zeros()
    m_csr.sort_indices()

    if m.shape != m_csr.shape:
        return False

    # Check nonzero elements are the same
    if not np.array_equal(m.indices, m_csr.indices):
        return False
    if not np.array_equal(m.indptr, m_csr.indptr):
        return False

    m_csc = csc_matrix(m_csr)
    # All edges should be colored
    if np.any(m_csr.data < 0):
        return False
    row_weight = np.max(m_csr.indptr[1:] - m_csr.indptr[0:-1])
    col_weight = np.max(m_csc.indptr[1:] - m_csc.indptr[0:-1])
    degree = max(row_weight, col_weight)
    # Chromatic index is the degree
    if np.any(m_csr.data > degree):
        return False
    # Check nonzero elements in each row (using m_csr) and column
    # (using m_csc) have unique elements. Corresponds to checking
    # edge colors are unique for A and B node set, respectively.
    for m_sparse in (m_csr, m_csc):
        for i in range(m_sparse.indptr.shape[0] - 1):
            node_colors = m_sparse.data[m_sparse.indptr[i] : m_sparse.indptr[i + 1]]
            if np.unique(node_colors).shape[0] < node_colors.shape[0]:
                return False
    return True


def bipartite_edge_coloring(biadjacency_matrix: csr_matrix) -> csr_matrix:
    """Find an edge coloring of a bipartite graph defined by a
    biadjacency matrix. The chromatic index is given by the degree
    of the graph.

    An edge coloring of a graph is an assignment of a color to each edge,
    such that no node has more than one edge of a given color. For a
    bipartite graph, the number of colors needed (the chromatic index)
    is exactly equal to the maximum degree of the graph.

    The edge coloring is returned via a colored biadjacency matrix M (another
    scipy.sparse csr_matrix). The locations of the nonzero elements are identical to the
    locations of the nonzero elements in the input biadjacency matrix, but instead
    their integer value now corresponds to the color c of the edge.
    The color c of the edge is an integer satisfying `1 <= c <= degree` where
    degree is the maximum degree of the bipartite graph (its chromatic index).

    Uses a method based on augmenting paths (the second algorithm
    reviewed in https://dl.acm.org/doi/pdf/10.1145/800133.804346).
    For a bipartite graph with E edges and V nodes, the worst-case
    complexity is O(EV), although empirically it is closer
    to O(E) for many graphs (e.g. random bipartite graphs with
    constant maximum degree).

    Parameters
    ----------
    biadjacency_matrix : csr_matrix[np.uint8]
        A biadjacency matrix defining a bipartite graph.
        For a bipartite graph with node sets A and B, each row of biadjacency_matrix
        corresponds to a node in A, and each column of biadjacency matrix corresponds
        to a node in B. Element `biadjacency_matrix[i,j]==1` if and only if node
        i in A is connected by an edge to node j in B, otherwise `biadjacency_matrix[i,j]==0`.

    Returns
    -------
    csr_matrix[np.int64]
        A biadjacency matrix `M` representing the bipartite graph where the
        element `M[i,j]` is nonzero if and only if there is an edge from
        node i in the A set to node j in the B set (it has the same locations
        of nonzero elements as the input `biadjacency_matrix`).
        The value of `M[i,j]` is the edge's color in the graph.
        In other words we have `1 <= M[i,j] <= degree`, where degree
        is the maximum degree of the bipartite graph.
    """
    graph = BipartiteGraph.from_biadjacency_matrix(biadjacency_matrix)
    graph.bipartite_edge_coloring()
    col_biadj = graph.to_biadjacency_matrix()
    assert is_valid_bipartite_edge_coloring(
        biadj_matrix=biadjacency_matrix, colored_biadj_matrix=col_biadj
    )
    return col_biadj
