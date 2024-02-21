import numpy as np
import sinter
import stim
from matplotlib import pyplot as plt
from ldpc.ckt_noise.sinter_overlapping_window_decoder import sinter_owd_decoders

from src_python.ldpc.ckt_noise.config import DEFAULT_DECODINGS, DEFAULT_WINDOW, DEFAULT_COMMIT, \
    DEFAULT_BPOSD_DECODER_ARGS, DEFAULT_LSD_DECODER_ARGS
from src_python.ldpc.ckt_noise.sinter_overlapping_window_decoder import SinterDecoder_BPOSD_OWD, SinterDecoder_LSD_OWD


def generate_example_tasks():
    for p in np.arange(0.001, 0.01, 0.002):
        for d in [5, 7, 9]:
            sc_circuit = stim.Circuit.generated(
                rounds=d,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
                code_task=f'surface_code:rotated_memory_z',
            )
            yield sinter.Task(
                circuit=sc_circuit,
                json_metadata={
                    'p': p,
                    'd': d,
                    'rounds': d,
                },
            )

def main():
    samples = sinter.collect(
        num_workers=10,
        max_shots=5000,
        max_errors=100,
        tasks=generate_example_tasks(),
        decoders=['bposd_owd', 'lsd_owd'],
        custom_decoders=sinter_owd_decoders(),
        print_progress=True,
        save_resume_filepath=f'owd_sc.csv',
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, axis = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    sinter.plot_error_rate(
        ax=axis[0],
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == 'bposd_owd',
        x_func=lambda stat: stat.json_metadata['p'],
    )

    sinter.plot_error_rate(
        ax=axis[1],
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == 'lsd_owd',
        x_func=lambda stat: stat.json_metadata['p'],
    )
    axis[0].set_ylabel('Logical Error Rate')
    axis[0].set_title('BPOSD_OWD')
    axis[1].set_title('LSD_OWD')
    for ax in axis:
        ax.loglog()
        ax.grid()
        ax.set_xlabel('Physical Error Rate')
        ax.legend()

    # Save to file and also open in a window.
    fig.savefig('plot.png')
    plt.show()


if __name__ == '__main__':
    main()
