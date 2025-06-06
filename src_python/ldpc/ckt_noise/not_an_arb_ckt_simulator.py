from typing import List, Union

from scipy.sparse import csr_matrix
import stim
import numpy as np
from pymatching import Matching

from ldpc.ckt_noise.bipartite_edge_coloring import bipartite_edge_coloring


def rep_code(d: int):
    """
    Generate a repetition code check matrix.

    Parameters:
    - d (int): The dimension of the repetition code.

    Returns:
    - csr_matrix: The check matrix of the repetition code.
    """
    h = np.zeros((d - 1, d), dtype=np.int8)
    for i in range(d - 1):
        h[i, i] = 1
        h[i, (i + 1)] = 1
    return csr_matrix(h), csr_matrix([[1] + [0] * (d - 1)])


def get_stabilizer_time_steps(pcm: csr_matrix):
    """
    .. warning::
       A bug was present in this function in ldpc version 2.3.5 and earlier,
       which, for some inputs, could lead to some nonzero elements of the check matrix
       being ignored (and not assigned to any time step). This bug was fixed in
       version 2.3.6, however this method is now deprecated.
       Users on affected versions are strongly advised to upgrade and use
       :func:`ldpc.ckt_noise.bipartite_edge_coloring.bipartite_edge_coloring` instead.
       See https://github.com/quantumgizmos/ldpc/issues/68 for more details.

    .. deprecated:: 2.3.6
       This function is deprecated and will be removed in a future version. Please use
       :func:`ldpc.ckt_noise.bipartite_edge_coloring.bipartite_edge_coloring` instead.


    Get the time steps and measured bits for a given parity check matrix.

    Args:
        pcm (csr_matrix): The parity check matrix.

    Returns:
        tuple: A tuple containing two lists:
            - time_steps: A 2D list where each row represents a time step and each column represents a stabilizer.
            - measured_bits: A 2D list where each row represents a stabilizer and each column represents a time step.
    """
    import warnings

    warnings.warn(
        "get_stabilizer_time_steps(pcm) is deprecated and will be removed in a future version. "
        "Use :func:`ldpc.ckt_noise.bipartite_edge_coloring.bipartite_edge_coloring` instead."
        "WARNING: in ldpc version 2.3.5 and earlier this method contained a bug which, for some "
        "inputs, could lead to some nonzero elements of the check matrix "
        "being ignored (see docstring).",
        DeprecationWarning,
        stacklevel=2,
    )

    col_mat = csr_matrix(bipartite_edge_coloring(pcm))
    max_time_steps = np.max(col_mat.data)
    num_stabs = pcm.shape[0]

    time_steps = [[None] * num_stabs for _ in range(max_time_steps)]
    measured_bits = [[None] * max_time_steps for _ in range(num_stabs)]

    for k in range(col_mat.shape[0]):
        for j in range(col_mat.indptr[k], col_mat.indptr[k + 1]):
            t = col_mat.data[j] - 1
            q = col_mat.indices[j]
            time_steps[t][k] = q
            measured_bits[k][t] = q

    return time_steps, measured_bits


def stim_circuit_from_time_steps(
    pcm: csr_matrix,
    logicals: csr_matrix,
    time_steps: List[List[Union[int, None]]],
    measured_bits: List[List[Union[int, None]]],
    after_clifford_depolarization: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    before_round_data_depolarization: float = 0.0,
    rounds: int = 3,
):
    """
    .. warning::
       A bug was present in this function in ldpc version 2.3.5 and earlier,
       which, for some inputs, could lead to some CNOT gates being omitted from the circuit.
       This bug was fixed in this method in version 2.3.6, however this method is now
       deprecated. Users on affected versions are strongly advised to upgrade and use
       :func:`ldpc.ckt_noise.css_code_memory_circuit.make_css_code_memory_circuit` instead.
       See https://github.com/quantumgizmos/ldpc/issues/68 for more details.

    .. deprecated:: 2.3.6
       This function is deprecated and will be removed in a future version. Please use
       :func:`ldpc.ckt_noise.css_code_memory_circuit.make_css_code_memory_circuit` instead.
       The new `make_css_code_memory_circuit()` method allows for measuring both X and Z
       stabilizers of a CSS code.


    Generates a stim circuit based on a parity check matrix and associated logcal operators.
    
    Note that this method only measures X stabilizers (defined by the input check matrix) and 
    is therefore only a reasonable circuit-level noise model for classical codes (e.g. repetition
    codes). To generate a stim circuit for a stim circuit, please use the new
    :func:`ldpc.ckt_noise.css_code_memory_circuit.make_css_code_memory_circuit` function instead.
    """
    import warnings

    warnings.warn(
        "stim_circuit_from_time_steps(pcm) is deprecated and will be removed in a future version. "
        "Use :func:`ldpc.ckt_noise.css_code_memory_circuit.make_css_code_memory_circuit` instead."
        "WARNING: in ldpc version 2.3.5 and earlier this method contained a bug where some"
        " CNOT gates could be silently omitted from the circuit for certain inputs (see docstring).",
        DeprecationWarning,
        stacklevel=2,
    )
    # use index convention 0..n-1 for qubits and n..n+m-1 for stabilizers.
    m, n = pcm.shape
    data = np.arange(n)
    checks = np.arange(n, n + m)

    init_circuit = stim.Circuit()
    circuit = stim.Circuit()
    final_circuit = stim.Circuit()

    init_circuit.append("R", np.arange(n + m))
    init_circuit.append("TICK")

    ######### INIT BLOCK #########
    if before_round_data_depolarization > 0:
        init_circuit.append("DEPOLARIZE1", data, before_round_data_depolarization)
        # add time ticks
        init_circuit.append("TICK", [])

    # stabilizer measurements circuits using "CX"
    for _, tick in enumerate(time_steps):
        for check, bit in enumerate(tick):
            if bit is None:
                continue
            init_circuit.append("CX", [bit, n + check])
            if after_clifford_depolarization > 0:
                init_circuit.append(
                    "DEPOLARIZE2", [bit, n + check], after_clifford_depolarization
                )
        init_circuit.append("TICK", [])

    # measure check qubits
    init_circuit.append("MR", checks, before_measure_flip_probability)
    init_circuit.append("X_ERROR", checks, after_reset_flip_probability)

    # define detectors
    for idx in range(m):
        init_circuit.append("DETECTOR", [stim.target_rec(-m + idx)], (idx + n, 0))
    ######### END INIT BLOCK #########

    ######### REPEAT BLOCK #########
    # add time ticks
    circuit.append("TICK", [])
    # before round data depolarization
    if before_round_data_depolarization > 0:
        circuit.append("DEPOLARIZE1", data, before_round_data_depolarization)
        # add time ticks
        circuit.append("TICK", [])

    # stabilizer measurements circuits using "CX"
    for _, tick in enumerate(time_steps):
        for check, bit in enumerate(tick):
            if bit is None:
                continue
            circuit.append("CX", [bit, n + check])
            if after_clifford_depolarization > 0:
                circuit.append(
                    "DEPOLARIZE2", [bit, n + check], after_clifford_depolarization
                )
        circuit.append("TICK", [])

    # measure check qubits
    circuit.append("MR", checks, before_measure_flip_probability)
    circuit.append("X_ERROR", checks, after_reset_flip_probability)

    circuit.append("SHIFT_COORDS", [], [0, 1])

    for idx in range(m):
        circuit.append(
            "DETECTOR",
            [stim.target_rec(-2 * m + idx), stim.target_rec(-m + idx)],
            (idx + n, 0),
        )
    circuit *= rounds
    ##### END REPEAT BLOCK ######

    #### BEGIN FINAL BLOCK ####
    if before_round_data_depolarization > 0:
        final_circuit.append("DEPOLARIZE1", data, before_round_data_depolarization)

    final_circuit.append("M", data)

    for k in range(m):
        bits = pcm[k].indices
        record_targets = [stim.target_rec(-m - n + k)]
        for bit in bits:
            record_targets.append(stim.target_rec(-n + bit))

        final_circuit.append("DETECTOR", record_targets, (k, 1))

    # iterate rows of logicals, add observable include
    for idx, logical in enumerate(logicals):
        final_circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-n + k) for k in logical.indices],
            idx,
        )
    #### END FINAL BLOCK ####

    complete_circuit = init_circuit + circuit + final_circuit

    return complete_circuit


# only for testing purposes, copied from the STIM readme.md
def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ps = np.linspace(0.03, 0.09, 11)
    for d in [5, 9, 13, 17]:
        rates = []
        pcm = rep_code(d)
        time_steps, measured_bits = get_stabilizer_time_steps(pcm)
        logicals = csr_matrix([[1] + [0] * (d - 1)])
        for p in ps:
            circuit = stim_circuit_from_time_steps(
                pcm,
                logicals,
                time_steps,
                measured_bits,
                before_round_data_depolarization=p,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                rounds=d,
            )

            num_shots = 50_000
            num_logical_errors = count_logical_errors(circuit, num_shots)

            rate = num_logical_errors / num_shots
            rates.append(rate)

        plt.plot(ps, rates, label=f"d={d}", marker="o")

    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
