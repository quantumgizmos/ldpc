import sinter
from ldpc.sinter_decoders import SinterBpOsdDecoder
import stim
from matplotlib import pyplot as plt
import numpy as np


def generate_example_tasks():
    for p in np.arange(0.001, 0.01, 0.001):
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
        max_shots=20_000,
        max_errors=100,
        tasks=generate_example_tasks(),
        decoders=['bposd'],
        custom_decoders={'bposd': SinterBpOsdDecoder(
            max_iter=5,
            bp_method="ms",
            ms_scaling_factor=0.625,
            schedule="parallel",
            osd_method="osd0")},
        print_progress=True,
        save_resume_filepath=f'bposd_surface_code.csv',
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']} dec={stat.decoder}",
        x_func=lambda stat: stat.json_metadata['p'],
    )
    ax.loglog()
    ax.grid()
    ax.set_title('Logical Error Rate vs Physical Error Rate')
    ax.set_ylabel('Logical Error Probability (per shot)')
    ax.set_xlabel('Physical Error Rate')
    ax.legend()

    # Save to file and also open in a window.
    fig.savefig('plot.png')
    plt.show()


if __name__ == '__main__':
    main()
