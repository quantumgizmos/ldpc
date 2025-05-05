import stim
from scipy.sparse import csr_matrix, coo_matrix, spmatrix
from typing import Callable, Optional, Tuple, Iterable
import numpy as np
import numpy.typing as npt

from ldpc.ckt_noise.bipartite_edge_coloring import bipartite_edge_coloring


def append_cycle_cx_gates_from_steps(
    *,
    circuit: stim.Circuit,
    cx_steps: spmatrix,
    control_qubits: npt.NDArray,
    target_qubits: npt.NDArray,
    all_qubits: npt.NDArray,
    after_cx_depolarization: float,
    idle_during_cx_depolarization: float,
):
    """A cx gate controlled on control_qubits[i] and targeted on target_qubits[j]
    is implemented in timestep cx_steps[i,j], and is not implemented at all
    if cx_steps[i,j]==0.
    """
    cx_steps = coo_matrix(cx_steps)
    num_steps = np.max(cx_steps.data)

    for t in range(1, num_steps + 1):
        c_indices = cx_steps.row[cx_steps.data == t]
        x_indices = cx_steps.col[cx_steps.data == t]
        c_qubits = control_qubits[c_indices]
        x_qubits = target_qubits[x_indices]
        assert c_qubits.size == x_qubits.size
        cx_targets = np.empty((c_qubits.size * 2,), dtype=c_qubits.dtype)
        cx_targets[0::2] = c_qubits
        cx_targets[1::2] = x_qubits
        circuit.append(name="CX", targets=cx_targets)
        if after_cx_depolarization > 0:
            circuit.append(
                name="DEPOLARIZE2", targets=cx_targets, arg=after_cx_depolarization
            )
        if idle_during_cx_depolarization > 0:
            idle_qubits = np.setdiff1d(all_qubits, cx_targets)
            circuit.append(
                name="DEPOLARIZE1",
                targets=idle_qubits,
                arg=idle_during_cx_depolarization,
            )
        circuit.append("TICK")


def make_css_code_memory_circuit(
    *,
    x_stabilizers: csr_matrix,
    z_stabilizers: csr_matrix,
    x_logicals: csr_matrix,
    z_logicals: csr_matrix,
    num_rounds: int,
    basis: str,
    after_clifford_depolarization: float = 0,
    before_round_data_depolarization: float = 0,
    before_measure_flip_probability: float = 0,
    after_reset_flip_probability: float = 0,
    idle_during_clifford_depolarization: float = 0,
    include_opposite_basis_detectors: bool = True,
    qubit_coord_func: Optional[Callable[[int], Iterable[float]]] = None,
    detector_coord_func: Optional[Callable[[int], Iterable[float]]] = None,
    shift_coords_per_round: Optional[Iterable[int]] = None,
) -> stim.Circuit:
    """Generates a syndrome extraction circuit implementing a memory
    experiment for an arbitrary CSS code.

    This method uses the approach taken by Algorithm 1 in https://arxiv.org/abs/2109.14609
    to measure the X stabilizers and then the Z stabilizers in each round.
    In each round, the X stabilizers are measured using deg_X layers (time steps) of CNOT gates,
    and then the Z stabilizers are measured using deg_Z layers of CNOT gates.
    Here deg_X is the maximum degree of any node in the Tanner graph defining
    the X stabilizers (maximum weight of any row or column in the x_stabilizers matrix).
    Similarly deg_Z is the maximum weight of any row or column in the z_stabilizers matrix.

    Parameters
    ----------
    x_stabilizers : csr_matrix
        Check matrix defining the X stabilizers. `x_stabilizers[i,j] == 1` if and only
        if X stabilizer i is supported on qubit j, otherwise `x_stabilizers[i,j] == 0`.
    z_stabilizers : csr_matrix
        Check matrix defining the Z stabilizers. `z_stabilizers[i,j] == 1` if and only
        if Z stabilizer i is supported on qubit j, otherwise `z_stabilizers[i,j] == 0`.
    x_logicals : csr_matrix
        Check matrix defining the X logical operators. `x_logicals[i,j] == 1` if and only
        if X logical i is supported on qubit j, otherwise `x_logicals[i,j] == 0`.
    z_logicals : csr_matrix
        Check matrix defining the Z logical operators. `z_logicals[i,j] == 1` if and only
        if Z logical i is supported on qubit j, otherwise `z_logicals[i,j] == 0`.
    num_rounds : int
        The number of rounds in the memory experiment. The number of times each stabilizer
        is measured using its measure qubit.
    basis : str
        The basis of the memory experiment. This is the basis (X or Z) in which the memory
        experiment if implemented. If `basis=="X"`, the data qubits are initialized in
        |+> and measured at the end of the circuit in the X basis (the X logical operators
        are measured transversally). If `basis=="Z"`, the data qubits are initialized in
        |0> and measured at the end of the circuit in the Z basis (the Z logical operators
        are measured transversally).
    after_clifford_depolarization : float, optional
        The strength of a stim DEPOLARIZE2 error applied after each CNOT gate in
        the circuit, by default 0
    before_round_data_depolarization : float, optional
        The strength of a stim DEPOLARIZE1 error applied on each data qubit before each
        round, by default 0
    before_measure_flip_probability : float, optional
        The probability that the result of each measurement is incorrect, by default 0
    after_reset_flip_probability : float, optional
        The probability that the state of a qubit is flipped immediately after initialization.
        This is the strength of a stim X_ERROR immediately after each RZ gate, and also the
        strength of a stim Z_ERROR immediately after each RX gate. By default 0
    idle_during_clifford_depolarization : float, optional
        This is the strength of a DEPOLARIZE1 error applied to each idling qubit during a
        time step in which CNOT gates are being applied to other qubits. Specifically,
        this is the strength of a DEPOLARIZE1 error applied to each idling data qubit and
        X measure qubit during a time step in which CNOT gates are coupling other X measure
        qubits to data qubits; it is also the strength of a DEPOLARIZE1 error applied to
        each idling data qubit and Z measure qubit during a time step in which CNOT gates
        are coupling other Z measure qubits to data qubits. In other words, Z measure qubits
        are assumed not to be idle (not yet initialized) during X measurements, and X measure
        qubits are assumed not to be idle (not yet initialized) during Z measurements.
        By default 0
    include_opposite_basis_detectors : bool, optional
        Whether or not to include detectors of the opposite basis to that of the memory experiment.
        If set to True, then all detectors (both bases) are included. If set to False, then only
        X detectors are included in a `basis="X"` memory experiment, and only
        Z detectors are included in a `basis="Z"` memory experiment. Note that including both bases
        can degrade performance for some decoders (e.g. for those based on belief propagation, see
        Appendix A of https://arxiv.org/abs/2503.10988v1). Note that
        `include_opposite_basis_detectors=False` is sufficient for a decoder to be able to decode
        up to the full distance of the circuit (since an X memory experiment only needs to
        preserve the X logicals, and vice versa for Z), however
        `include_opposite_basis_detectors=True` provides full syndrome information that in principle
        can be used by a good decoder to exploit knowledge of Y errors, boosting performance.
        By default True
    qubit_coord_func : Optional[Optional[Callable[[int], Iterable[float]]]], optional
        A function that maps the index of a qubit to its coordinates, for use as an argument
        to the QUBIT_COORDS instruction for each qubit. By default None, in which case the
        QUBIT_COORDS for each qubit is just its index.
    detector_coord_func : Optional[Optional[Callable[[int], Iterable[float]]]], optional
        A function that maps the index of a measure qubit to the desired coordinates of
        any detector that uses it. Note that X stabilizer i is assigned a measure qubit
        with index `n + i`, where `n = x_stabilizers.shape[1]` is the number of data qubits,
        and Z stabilizer i is assigned a measure qubit with index `n + x_stabilizers.shape[0] + i`.
        For example, `detector_coord_func = lambda q: [q, q >= n + rx, 0]` (where
        `n = x_stabilizers.shape[1]` and `rx = x_stabilizers.shape[0]`) assigns [q, b, 0]
        coordinates for each detector, where q is the measure qubit index, b is its basis (0 for
        X and 1 for Z) and 0 is a placeholder for the round.
        By default None, in which case the coordinate for each detector is just [q, 0], where q is
        the coordinate of the measure qubit involved in the detector.
    shift_coords_per_round : Optional[Iterable[int]], optional
        The shift in the coordinates to apply to each detector at the end of each round. i.e. the
        argument to the SHIFT_COORDS instruction that is inserted after each round.
        By default None, in which case SHIFT_COORDS(0, 1) is inserted after each round.

    Returns
    -------
    stim.Circuit
        The stim circuit of the memory experiment
    """
    x_stabilizers = csr_matrix(x_stabilizers)
    z_stabilizers = csr_matrix(z_stabilizers)
    x_logicals = csr_matrix(x_logicals)
    z_logicals = csr_matrix(z_logicals)

    basis = str(basis).upper()
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be X or Z, not {basis}")

    n = x_stabilizers.shape[1]  # Number of data qubits
    rx = x_stabilizers.shape[0]  # Number of x stabilizers
    rz = z_stabilizers.shape[0]  # Number of z stabilizers

    if (
        z_stabilizers.shape[1] != n
        or x_logicals.shape[1] != n
        or z_logicals.shape[1] != n
    ):
        raise ValueError(
            "x_checks, z_checks, x_logicals and z_logicals must all have the same number of columns"
        )

    if shift_coords_per_round is None:
        shift_coords_per_round = [0, 1]

    data_qubits = np.arange(n, dtype=np.int64)
    x_measure_qubits = np.arange(n, n + rx, dtype=np.int64)
    z_measure_qubits = np.arange(n + rx, n + rx + rz, dtype=np.int64)

    basis_measure_qubit_offset = n if basis == "X" else n + rx

    x_steps = bipartite_edge_coloring(biadjacency_matrix=x_stabilizers)
    z_steps = bipartite_edge_coloring(biadjacency_matrix=z_stabilizers).T

    x_measure_and_data = np.concatenate([data_qubits, x_measure_qubits])
    z_measure_and_data = np.concatenate([data_qubits, z_measure_qubits])

    def append_cycle_gates_both_bases(circuit: stim.Circuit):
        if before_round_data_depolarization > 0:
            circuit.append(
                "DEPOLARIZE1", targets=data_qubits, arg=before_round_data_depolarization
            )
        circuit.append("RX", targets=list(x_measure_qubits))
        if after_reset_flip_probability > 0:
            circuit.append(
                "Z_ERROR",
                targets=list(x_measure_qubits),
                arg=after_reset_flip_probability,
            )
        # Measure X stabilizers
        append_cycle_cx_gates_from_steps(
            circuit=circuit,
            cx_steps=x_steps,
            control_qubits=x_measure_qubits,
            target_qubits=data_qubits,
            all_qubits=x_measure_and_data,
            after_cx_depolarization=after_clifford_depolarization,
            idle_during_cx_depolarization=idle_during_clifford_depolarization,
        )
        circuit.append(
            "MX", targets=list(x_measure_qubits), arg=before_measure_flip_probability
        )
        circuit.append("RZ", targets=list(z_measure_qubits))
        if after_reset_flip_probability > 0:
            circuit.append(
                "X_ERROR",
                targets=list(z_measure_qubits),
                arg=after_reset_flip_probability,
            )
        circuit.append("TICK")
        # Measure Z stabilizers
        append_cycle_cx_gates_from_steps(
            circuit=circuit,
            cx_steps=z_steps,
            control_qubits=data_qubits,
            target_qubits=z_measure_qubits,
            all_qubits=z_measure_and_data,
            after_cx_depolarization=after_clifford_depolarization,
            idle_during_cx_depolarization=idle_during_clifford_depolarization,
        )
        circuit.append(
            "MZ", targets=list(z_measure_qubits), arg=before_measure_flip_probability
        )

    opp_basis = "X" if basis == "Z" else "Z"

    # Create circuit head (first round of measurements)
    head = stim.Circuit()

    # Add qubit coords
    for i in range(n + rx + rz):
        head.append(
            "QUBIT_COORDS",
            targets=i,
            arg=qubit_coord_func(i) if qubit_coord_func is not None else i,
        )

    head.append(f"R{basis}", data_qubits)
    if after_reset_flip_probability > 0:
        head.append(
            f"{opp_basis}_ERROR", targets=data_qubits, arg=after_reset_flip_probability
        )

    append_cycle_gates_both_bases(circuit=head)

    # Add initial detectors
    offset = -rx - rz if basis == "X" else -rz
    r = rx if basis == "X" else rz

    for i in range(r):
        qubit_idx = basis_measure_qubit_offset + i
        head.append(
            "DETECTOR",
            targets=[stim.target_rec(offset + i)],
            arg=detector_coord_func(qubit_idx)
            if detector_coord_func is not None
            else [qubit_idx, 0],
        )

    # Create repeating part of the circuit
    body = stim.Circuit()
    append_cycle_gates_both_bases(circuit=body)
    body.append("SHIFT_COORDS", arg=shift_coords_per_round)
    # Add bulk detectors
    if basis == "X" or include_opposite_basis_detectors:
        for i in range(rx):
            qubit_idx = n + i
            body.append(
                "DETECTOR",
                targets=[
                    stim.target_rec(-2 * rx - 2 * rz + i),
                    stim.target_rec(-rx - rz + i),
                ],
                arg=detector_coord_func(qubit_idx)
                if detector_coord_func is not None
                else [qubit_idx, 0],
            )
    if basis == "Z" or include_opposite_basis_detectors:
        for i in range(rz):
            qubit_idx = n + rx + i
            body.append(
                "DETECTOR",
                targets=[stim.target_rec(-2 * rz - rx + i), stim.target_rec(-rz + i)],
                arg=detector_coord_func(qubit_idx)
                if detector_coord_func is not None
                else [qubit_idx, 0],
            )

    # Create final part of the circuit
    tail = stim.Circuit()
    tail.append(f"M{basis}", targets=data_qubits, arg=before_measure_flip_probability)
    # Add final detectors
    H = x_stabilizers if basis == "X" else z_stabilizers
    for i in range(H.shape[0]):
        qubit_idx = basis_measure_qubit_offset + i
        targets = [stim.target_rec(offset - n + i)]
        for j in H.indices[H.indptr[i] : H.indptr[i + 1]]:
            targets.append(stim.target_rec(-n + j))
        tail.append(
            "DETECTOR",
            targets=targets,
            arg=detector_coord_func(qubit_idx)
            if detector_coord_func is not None
            else [qubit_idx, 0],
        )

    # Add observables
    L = x_logicals if basis == "X" else z_logicals
    for i in range(L.shape[0]):
        targets = []
        for j in L.indices[L.indptr[i] : L.indptr[i + 1]]:
            targets.append(stim.target_rec(-n + j))
        tail.append("OBSERVABLE_INCLUDE", targets=targets, arg=i)

    return head + (num_rounds - 1) * body + tail
