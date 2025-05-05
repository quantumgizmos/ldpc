import pytest
from scipy.sparse import csr_matrix, hstack, kron, eye, block_diag
import numpy as np
from typing import Tuple, Iterable
import itertools

from ldpc.ckt_noise.css_code_memory_circuit import make_css_code_memory_circuit


def repetition_code(distance: int) -> csr_matrix:
    row_ind, col_ind = zip(
        *((i, j) for i in range(distance) for j in (i, (i + 1) % distance))
    )
    data = np.ones(2 * distance, dtype=np.uint8)
    return csr_matrix((data, (row_ind, col_ind)))


def toric_code_matrices(
    distance: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """Check matrices of a toric code on an unrotated lattice"""
    H = repetition_code(distance=distance)
    assert H.shape[1] == H.shape[0] == distance
    e = eye(distance)
    Hx = csr_matrix(hstack([kron(H, e), kron(e, H.T)], dtype=np.uint8))
    Hz = csr_matrix(hstack([kron(e, H), kron(H.T, e)], dtype=np.uint8))
    L0 = csr_matrix(([1], ([0], [0])), shape=(1, distance), dtype=np.uint8)
    L1 = csr_matrix(np.ones((1, distance), dtype=np.uint8))
    Lx = csr_matrix(block_diag([kron(L0, L1), kron(L1, L0)]))
    Lz = csr_matrix(block_diag([kron(L1, L0), kron(L0, L1)]))

    for m in (Hx, Hz, Lx, Lz):
        m.data = m.data % 2
        m.sort_indices()
        m.eliminate_zeros()

    return Hx, Hz, Lx, Lz


@pytest.mark.parametrize("distance", range(2, 13))
def test_toric_code_matrices(distance: int):
    Hx, Hz, Lx, Lz = toric_code_matrices(distance=distance)

    assert not np.any((Hx @ Hz.T).data % 2)
    assert not np.any((Hx @ Lz.T).data % 2)
    assert not np.any((Hz @ Lx.T).data % 2)

    for m in (Hx, Hz):
        assert not np.any(m.sum(axis=0) != 2)
        assert not np.any(m.sum(axis=1) != 4)

    for m in (Lx, Lz):
        assert not np.any(m.sum(axis=1) != distance)
        assert not np.any(m.sum(axis=0) > 1)

    n = 2 * distance**2
    r = distance**2
    assert Hx.shape[1] == Hz.shape[1] == Lx.shape[1] == Lz.shape[1] == n
    assert Hx.shape[0] == Hz.shape[0] == r
    assert Lx.shape[0] == Lz.shape[0] == 2


@pytest.mark.parametrize(
    "distance,basis,other_basis_dets",
    itertools.product(range(2, 6), ("X", "Z"), (True, False)),
)
def test_toric_code_circuit(distance: int, basis: str, other_basis_dets: bool):
    basis = "X"
    Hx, Hz, Lx, Lz = toric_code_matrices(distance=distance)
    num_rounds = 5
    other_basis_dets = True
    p = 0.001
    circuit = make_css_code_memory_circuit(
        x_stabilizers=Hx,
        z_stabilizers=Hz,
        x_logicals=Lx,
        z_logicals=Lz,
        num_rounds=num_rounds,
        basis=basis,
        include_opposite_basis_detectors=other_basis_dets,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        idle_during_clifford_depolarization=p,
    )
    circuit.detector_error_model(decompose_errors=True)
    circuit.compile_detector_sampler().sample(shots=10)
    assert circuit.num_detectors == num_rounds * 2 * distance**2
    assert circuit.num_observables == 2
    assert (
        circuit.count_determined_measurements()
        == circuit.num_detectors + circuit.num_observables
    )

    assert circuit.num_qubits == 4 * distance**2

    # Normally we would need to be careful of hook errors when constucting a surface code
    # circuit, but since we are using a (less efficient) unrotated surface code, any
    # order of gates is guaranteed to preserve the full distance of the code.
    # This holds for any HGP code: https://arxiv.org/abs/2308.15520
    assert len(circuit.shortest_graphlike_error()) == distance


@pytest.mark.parametrize(
    "include_opposite_basis_detectors,basis,expected_det_coords",
    [
        [
            True,
            "X",
            [
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 1.0],
                [0.0, 5.0, 1.0, 1.0],
                [0.0, 4.0, 0.0, 1.0],
            ],
        ],
        [
            True,
            "Z",
            [
                [0.0, 5.0, 1.0, 0.0],
                [0.0, 4.0, 0.0, 1.0],
                [0.0, 5.0, 1.0, 1.0],
                [0.0, 5.0, 1.0, 1.0],
            ],
        ],
        [
            False,
            "X",
            [[0.0, 4.0, 0.0, 0.0], [0.0, 4.0, 0.0, 1.0], [0.0, 4.0, 0.0, 1.0]],
        ],
        [
            False,
            "Z",
            [[0.0, 5.0, 1.0, 0.0], [0.0, 5.0, 1.0, 1.0], [0.0, 5.0, 1.0, 1.0]],
        ],
    ],
)
def test_make_css_code_memory_circuit_coordinates_422_code(
    include_opposite_basis_detectors: bool,
    basis: str,
    expected_det_coords: Iterable[Iterable[int]],
):
    H = csr_matrix([[1, 1, 1, 1]])
    n = H.shape[1]
    rx = H.shape[0]
    # Put qubit index in the y (2nd) coordinate, basis in the
    # detector z (3rd) coordinate (0 for X, 1 for Z) and time
    # in the detector t (4th) coordinate.
    circuit = make_css_code_memory_circuit(
        x_stabilizers=H,
        z_stabilizers=H,
        x_logicals=[[1, 0, 1, 0], [0, 0, 1, 1]],
        z_logicals=[[1, 1, 0, 0], [0, 0, 1, 1]],
        num_rounds=2,
        basis=basis,
        include_opposite_basis_detectors=include_opposite_basis_detectors,
        qubit_coord_func=lambda q: [0, q],
        detector_coord_func=lambda q: [0, q, q >= n + rx, 0],
        shift_coords_per_round=[0, 0, 0, 1],
    )
    all_det_coords = []
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            args = inst.gate_args_copy()
            all_det_coords.append(args)
    assert all_det_coords == expected_det_coords


@pytest.mark.parametrize(
    "basis,expected_circuit",
    [
        (
            "X",
            """QUBIT_COORDS(0) 0
QUBIT_COORDS(1) 1
QUBIT_COORDS(2) 2
QUBIT_COORDS(3) 3
QUBIT_COORDS(4) 4
QUBIT_COORDS(5) 5
RX 0 1 2 3
Z_ERROR(0.004) 0 1 2 3
DEPOLARIZE1(0.002) 0 1 2 3
RX 4
Z_ERROR(0.004) 4
CX 4 0
DEPOLARIZE2(0.001) 4 0
DEPOLARIZE1(0.005) 1 2 3
TICK
CX 4 1
DEPOLARIZE2(0.001) 4 1
DEPOLARIZE1(0.005) 0 2 3
TICK
CX 4 2
DEPOLARIZE2(0.001) 4 2
DEPOLARIZE1(0.005) 0 1 3
TICK
CX 4 3
DEPOLARIZE2(0.001) 4 3
DEPOLARIZE1(0.005) 0 1 2
TICK
MX(0.003) 4
R 5
X_ERROR(0.004) 5
TICK
CX 0 5
DEPOLARIZE2(0.001) 0 5
DEPOLARIZE1(0.005) 1 2 3
TICK
CX 1 5
DEPOLARIZE2(0.001) 1 5
DEPOLARIZE1(0.005) 0 2 3
TICK
CX 2 5
DEPOLARIZE2(0.001) 2 5
DEPOLARIZE1(0.005) 0 1 3
TICK
CX 3 5
DEPOLARIZE2(0.001) 3 5
DEPOLARIZE1(0.005) 0 1 2
TICK
M(0.003) 5
DETECTOR(4, 0) rec[-2]
REPEAT 9 {
    DEPOLARIZE1(0.002) 0 1 2 3
    RX 4
    Z_ERROR(0.004) 4
    CX 4 0
    DEPOLARIZE2(0.001) 4 0
    DEPOLARIZE1(0.005) 1 2 3
    TICK
    CX 4 1
    DEPOLARIZE2(0.001) 4 1
    DEPOLARIZE1(0.005) 0 2 3
    TICK
    CX 4 2
    DEPOLARIZE2(0.001) 4 2
    DEPOLARIZE1(0.005) 0 1 3
    TICK
    CX 4 3
    DEPOLARIZE2(0.001) 4 3
    DEPOLARIZE1(0.005) 0 1 2
    TICK
    MX(0.003) 4
    R 5
    X_ERROR(0.004) 5
    TICK
    CX 0 5
    DEPOLARIZE2(0.001) 0 5
    DEPOLARIZE1(0.005) 1 2 3
    TICK
    CX 1 5
    DEPOLARIZE2(0.001) 1 5
    DEPOLARIZE1(0.005) 0 2 3
    TICK
    CX 2 5
    DEPOLARIZE2(0.001) 2 5
    DEPOLARIZE1(0.005) 0 1 3
    TICK
    CX 3 5
    DEPOLARIZE2(0.001) 3 5
    DEPOLARIZE1(0.005) 0 1 2
    TICK
    M(0.003) 5
    SHIFT_COORDS(0, 1)
    DETECTOR(4, 0) rec[-4] rec[-2]
    DETECTOR(5, 0) rec[-3] rec[-1]
}
MX(0.003) 0 1 2 3
DETECTOR(4, 0) rec[-6] rec[-4] rec[-3] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-4] rec[-2]
OBSERVABLE_INCLUDE(1) rec[-2] rec[-1]""",
        ),
        (
            "Z",
            """QUBIT_COORDS(0) 0
QUBIT_COORDS(1) 1
QUBIT_COORDS(2) 2
QUBIT_COORDS(3) 3
QUBIT_COORDS(4) 4
QUBIT_COORDS(5) 5
R 0 1 2 3
X_ERROR(0.004) 0 1 2 3
DEPOLARIZE1(0.002) 0 1 2 3
RX 4
Z_ERROR(0.004) 4
CX 4 0
DEPOLARIZE2(0.001) 4 0
DEPOLARIZE1(0.005) 1 2 3
TICK
CX 4 1
DEPOLARIZE2(0.001) 4 1
DEPOLARIZE1(0.005) 0 2 3
TICK
CX 4 2
DEPOLARIZE2(0.001) 4 2
DEPOLARIZE1(0.005) 0 1 3
TICK
CX 4 3
DEPOLARIZE2(0.001) 4 3
DEPOLARIZE1(0.005) 0 1 2
TICK
MX(0.003) 4
R 5
X_ERROR(0.004) 5
TICK
CX 0 5
DEPOLARIZE2(0.001) 0 5
DEPOLARIZE1(0.005) 1 2 3
TICK
CX 1 5
DEPOLARIZE2(0.001) 1 5
DEPOLARIZE1(0.005) 0 2 3
TICK
CX 2 5
DEPOLARIZE2(0.001) 2 5
DEPOLARIZE1(0.005) 0 1 3
TICK
CX 3 5
DEPOLARIZE2(0.001) 3 5
DEPOLARIZE1(0.005) 0 1 2
TICK
M(0.003) 5
DETECTOR(5, 0) rec[-1]
REPEAT 9 {
    DEPOLARIZE1(0.002) 0 1 2 3
    RX 4
    Z_ERROR(0.004) 4
    CX 4 0
    DEPOLARIZE2(0.001) 4 0
    DEPOLARIZE1(0.005) 1 2 3
    TICK
    CX 4 1
    DEPOLARIZE2(0.001) 4 1
    DEPOLARIZE1(0.005) 0 2 3
    TICK
    CX 4 2
    DEPOLARIZE2(0.001) 4 2
    DEPOLARIZE1(0.005) 0 1 3
    TICK
    CX 4 3
    DEPOLARIZE2(0.001) 4 3
    DEPOLARIZE1(0.005) 0 1 2
    TICK
    MX(0.003) 4
    R 5
    X_ERROR(0.004) 5
    TICK
    CX 0 5
    DEPOLARIZE2(0.001) 0 5
    DEPOLARIZE1(0.005) 1 2 3
    TICK
    CX 1 5
    DEPOLARIZE2(0.001) 1 5
    DEPOLARIZE1(0.005) 0 2 3
    TICK
    CX 2 5
    DEPOLARIZE2(0.001) 2 5
    DEPOLARIZE1(0.005) 0 1 3
    TICK
    CX 3 5
    DEPOLARIZE2(0.001) 3 5
    DEPOLARIZE1(0.005) 0 1 2
    TICK
    M(0.003) 5
    SHIFT_COORDS(0, 1)
    DETECTOR(4, 0) rec[-4] rec[-2]
    DETECTOR(5, 0) rec[-3] rec[-1]
}
M(0.003) 0 1 2 3
DETECTOR(5, 0) rec[-5] rec[-4] rec[-3] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-4] rec[-3]
OBSERVABLE_INCLUDE(1) rec[-2] rec[-1]""",
        ),
    ],
)
def test_css_code_memory_circuit_422_code_exact_match(
    basis: str, expected_circuit: str
):
    num_rounds = 10
    circuit = make_css_code_memory_circuit(
        x_stabilizers=[[1, 1, 1, 1]],
        z_stabilizers=[[1, 1, 1, 1]],
        x_logicals=[[1, 0, 1, 0], [0, 0, 1, 1]],
        z_logicals=[[1, 1, 0, 0], [0, 0, 1, 1]],
        num_rounds=num_rounds,
        basis=basis,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.002,
        before_measure_flip_probability=0.003,
        after_reset_flip_probability=0.004,
        idle_during_clifford_depolarization=0.005,
    )
    assert circuit.num_detectors == 2 * num_rounds
    assert circuit.num_observables == 2
    assert (
        circuit.count_determined_measurements()
        == circuit.num_detectors + circuit.num_observables
    )
    circuit.detector_error_model(decompose_errors=True)
    circuit.compile_detector_sampler().sample(shots=10)
    # Circuit is distance 1 (hook error from CX)
    assert len(circuit.shortest_graphlike_error()) in {1, 2}
    assert str(circuit) == expected_circuit
