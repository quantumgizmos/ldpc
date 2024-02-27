import numpy as np
import sinter
import stim
from ldpc.sinter_decoders import SinterBeliefFindDecoder, SinterBpOsdDecoder
from matplotlib import pyplot as plt


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
    from src_python.ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder
    samples = sinter.collect(
        num_workers=10,
        max_shots=20_000,
        max_errors=100,
        tasks=generate_example_tasks(),
        decoders=['bposd', 'belief_find', 'bplsd'],
        custom_decoders={
            'bposd': SinterBpOsdDecoder(
                max_iter=10,
                bp_method="ms",
                ms_scaling_factor=0.625,
                schedule="parallel",
                osd_method="osd0"),
            "belief_find": SinterBeliefFindDecoder(max_iter=10,
                                                   bp_method="ms",
                                                   ms_scaling_factor=0.625,
                                                   schedule="parallel"),
            'bplsd': SinterLsdDecoder(
                max_iter=2,
                bp_method="ms",
                ms_scaling_factor=0.625,
                schedule="parallel",
                lsd_order=0),
        },

        print_progress=True,
        save_resume_filepath=f'bposd_surface_code.csv',
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, axis = plt.subplots(1, 3, sharey=True, figsize=(10, 5))
    sinter.plot_error_rate(
        ax=axis[0],
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == 'bposd',
        x_func=lambda stat: stat.json_metadata['p'],
    )

    sinter.plot_error_rate(
        ax=axis[1],
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == 'belief_find',
        x_func=lambda stat: stat.json_metadata['p'],
    )
    sinter.plot_error_rate(
        ax=axis[2],
        stats=samples,
        group_func=lambda stat: f"Rotated Surface Code d={stat.json_metadata['d']}",
        filter_func=lambda stat: stat.decoder == 'lsd',
        x_func=lambda stat: stat.json_metadata['p'],
    )
    axis[0].set_ylabel('Logical Error Rate')
    axis[0].set_title('BPOSD')
    axis[1].set_title('Belief Find')
    axis[2].set_title('LSD')
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
