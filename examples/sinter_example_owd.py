import numpy as np
import scipy.sparse
import sinter
import stim
from matplotlib import pyplot as plt

from ldpc.ckt_noise.sinter_overlapping_window_decoder import (
    SinterDecoder_BPOSD_OWD,
    SinterDecoder_LSD_OWD,
    SinterDecoder_PyMatching_OWD,
)

from src_python.ldpc.ckt_noise.not_an_arb_ckt_simulator import stim_circuit_from_time_steps, get_stabilizer_time_steps


def generate_decoders(ds: np.ndarray, decodings: np.ndarray):
    decoders = {}
    for d in ds:
        for r in decodings:
            decoders[f"bposd_owd_d{d}_r{r}"] = SinterDecoder_BPOSD_OWD(
                decodings=int(r),
                window=int(2 * d),
                commit=int(d),
                num_checks=int(d - 1),
            )
            decoders[f"lsd_owd_d{d}_r{r}"] = SinterDecoder_LSD_OWD(
                decodings=int(r),
                window=int(2 * d),
                commit=int(d),
                num_checks=int(d - 1),
            )
            decoders[f"pymatching_owd_d{d}_r{r}"] = SinterDecoder_PyMatching_OWD(
                decodings=int(r),
                window=int(2 * d),
                commit=int(d),
                num_checks=int((d**2 - 1)),
                decoder_config={},
            )

            # decoders[f"pymatching_owd_d{d}_r{r}"] = SinterDecoder_PyMatching_OWD(
            #     decodings=int(r),
            #     window=int(2 * d),
            #     commit=int(d),
            #     num_checks=int(d - 1),
            # )
    return decoders

def generate_example_tasks_surface(ps: np.ndarray, ds: np.ndarray, decodings: np.ndarray):
    for r in decodings:
        for p in ps:
            for d in ds:
                rounds = int((r + 1) * d)
                sc_circuit = stim.Circuit.generated(
                    rounds=rounds,
                    distance=int(d),
                    after_clifford_depolarization=p,
                    after_reset_flip_probability=p,
                    before_measure_flip_probability=p,
                    before_round_data_depolarization=p,
                    code_task=f"surface_code:rotated_memory_x",
                )
                yield sinter.Task(
                    circuit=sc_circuit,
                    decoder=f"pymatching_owd_d{d}_r{r}",
                    json_metadata={
                        "p": p,
                        "d": int(d),
                        # "decodings": int(rounds),
                        # "commit": d,
                        # "window": 2 * d,
                        "decodings": int(r),
                    },
                )
def generate_example_tasks_qera(ps: np.ndarray, ds: np.ndarray, decodings: np.ndarray):
    for r in decodings:
        for p in ps:
            for d in ds:
                pcm = scipy.sparse.load_npz(f"/home/luca/Documents/codeRepos/ldpc/python_test/pcms/hqcodes/HQ_{d}_hx.npz")
                logicals = scipy.sparse.load_npz(f"/home/luca/Documents/codeRepos/ldpc/python_test/pcms/hqcodes/HQ_{d}_lx.npz")
                time_steps, measured_bits = get_stabilizer_time_steps(pcm)

                sc_circuit= stim_circuit_from_time_steps(
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
                yield sinter.Task(
                    circuit=sc_circuit,
                    decoder=f"lsd_owd_d{d}_r{r}",
                    json_metadata={
                        "p": p,
                        "d": int(d),
                        # "decodings": int(rounds),
                        # "commit": d,
                        # "window": 2 * d,
                        "decodings": int(r),
                    },
                )


def generate_example_tasks(ps: np.ndarray, ds: np.ndarray, decodings: np.ndarray):
    for r in decodings:
        for p in ps:
            for d in ds:
                rounds = int((r + 1) * d)
                sc_circuit = stim.Circuit.generated(
                    rounds=rounds,
                    distance=int(d),
                    after_clifford_depolarization=p,
                    after_reset_flip_probability=p,
                    before_measure_flip_probability=p,
                    before_round_data_depolarization=p,
                    code_task=f"surface_code:rotated_memory_x",
                )
                yield sinter.Task(
                    circuit=sc_circuit,
                    decoder=f"pymatching_owd_d{d}_r{r}",
                    json_metadata={
                        "p": p,
                        "d": int(d),
                        # "decodings": int(rounds),
                        # "commit": d,
                        # "window": 2 * d,
                        "decodings": int(r),
                    },
                )


def main():
    decodings = np.array([2, 3])
    ps = np.geomspace(2e-3, 0.011, 9)
    ds = np.array([12])
    samples = sinter.collect(
        num_workers=6,
        max_shots=50_000,
        max_errors=500,
        tasks=generate_example_tasks_qera(ps, ds, decodings),
        custom_decoders=generate_decoders(ds, decodings),
        print_progress=True,
        save_resume_filepath=f"owd_sc.csv",
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, axis = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    sinter.plot_error_rate(
        ax=axis[0],
        stats=samples,
        group_func=lambda stat: f"d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.json_metadata["decodings"] == 1,
        x_func=lambda stat: stat.json_metadata["p"],
    )

    sinter.plot_error_rate(
        ax=axis[1],
        stats=samples,
        group_func=lambda stat: f"d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.json_metadata["decodings"] == 2,
        x_func=lambda stat: stat.json_metadata["p"],
    )

    sinter.plot_error_rate(
        ax=axis[2],
        stats=samples,
        group_func=lambda stat: f"d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.json_metadata["decodings"] == 3,
        x_func=lambda stat: stat.json_metadata["p"],
    )

    axis[0].set_ylabel("Logical Error Rate")
    axis[0].set_title("Decodings = 1")
    axis[1].set_title("Decodings = 2")
    axis[2].set_title("Decodings = 3")
    for ax in axis:
        ax.loglog()
        ax.grid()
        ax.set_xlabel("Physical Error Rate")
        ax.legend()

    # Save to file and also open in a window.
    fig.savefig("plot.png")
    plt.show()


if __name__ == "__main__":
    main()
    # pass
