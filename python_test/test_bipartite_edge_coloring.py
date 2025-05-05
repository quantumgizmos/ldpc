from ldpc.ckt_noise.bipartite_edge_coloring import (
    BipartiteGraph,
    bipartite_edge_coloring,
    is_valid_bipartite_edge_coloring,
)
from scipy.sparse import csr_matrix
import numpy as np
import pytest


def test_bipartite_graph_from_adjacency_matrix():
    M = csr_matrix([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    G = BipartiteGraph.from_biadjacency_matrix(biadj=M)

    assert G.degree == 2

    a0 = G.a_nodes[0]
    a1 = G.a_nodes[1]
    assert a0.uncolored_edges == {0, 2}
    assert a0.colored_edges == {}
    assert a0.colors_available == {0, 1}
    assert a1.uncolored_edges == {0, 1}
    assert a1.colored_edges == {}
    assert a1.colors_available == {0, 1}

    b0 = G.b_nodes[0]
    b1 = G.b_nodes[1]
    b2 = G.b_nodes[2]
    assert b0.uncolored_edges == {0, 1}
    assert b0.colored_edges == {}
    assert b0.colors_available == {0, 1}
    assert b1.uncolored_edges == {1}
    assert b1.colored_edges == {}
    assert b1.colors_available == {0, 1}
    assert b2.uncolored_edges == {0}
    assert b2.colored_edges == {}
    assert b2.colors_available == {0, 1}


def test_bipartite_coloring_small_example():
    M = csr_matrix([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    C = bipartite_edge_coloring(biadjacency_matrix=M)
    C_expected = np.array([[3, 2, 0], [2, 1, 0], [1, 3, 2]], dtype=np.int64)
    assert np.array_equal(C.toarray(), C_expected)
    assert is_valid_bipartite_edge_coloring(M, C_expected)
    assert not is_valid_bipartite_edge_coloring(
        M, np.array([[3, 2, 0], [2, 1, 0], [0, 3, 2]], dtype=np.int64)
    )


def fixed_row_weight_rand_check_matrix(
    num_rows: int, num_cols: int, row_weight: int
) -> csr_matrix:
    A = np.zeros((num_rows, num_cols), dtype=np.uint8)
    for i in range(A.shape[0]):
        A[i, np.random.choice(A.shape[1], size=row_weight, replace=False)] = 1
    return csr_matrix(A)


@pytest.mark.parametrize(
    "num_rows,num_cols,row_weight",
    [(20, 50, 15), (5, 10, 5), (30, 60, 20), (45, 35, 20), (200, 1000, 20)],
)
def test_random_matrix_colorings(num_rows: int, num_cols: int, row_weight: int):
    for i in range(10):
        biadj = fixed_row_weight_rand_check_matrix(
            num_rows=num_rows, num_cols=num_cols, row_weight=row_weight
        )
        if i == 0:
            graph = BipartiteGraph.from_biadjacency_matrix(biadj=biadj)
            uncolored_biadj = graph.to_biadjacency_matrix()
            assert not np.any(uncolored_biadj.data != -1)
            assert np.array_equal(biadj.indices, uncolored_biadj.indices)
            assert np.array_equal(biadj.indptr, uncolored_biadj.indptr)
            graph.bipartite_edge_coloring()
            graph.assert_has_edge_coloring()
        colored_biadj = bipartite_edge_coloring(biadjacency_matrix=biadj)
        assert isinstance(biadj, csr_matrix)
        assert isinstance(colored_biadj, csr_matrix)
        assert colored_biadj.dtype == np.int64
        assert is_valid_bipartite_edge_coloring(
            biadj_matrix=biadj, colored_biadj_matrix=colored_biadj
        )
        assert not is_valid_bipartite_edge_coloring(
            biadj_matrix=biadj, colored_biadj_matrix=biadj
        )
