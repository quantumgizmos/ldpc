import matplotlib.pyplot as plt
import numpy as np
import panqec.codes

from ldpc.monte_carlo_simulation.data_utils import BpParams
from ldpc.monte_carlo_simulation.quasi_single_shot_v2 import QSS_SimulatorV2

if __name__ == "__main__":
    ps = np.linspace(0.015, 0.035, 6)
    fig, axis = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    nr_samples = 1000
    decoding_rds = 1
    for dist in [3, 5, 7]:
        print(f"d={dist}")
        lsd_lers = []
        osd_lers = []
        codename = "2DTC"
        code = panqec.codes.Toric2DCode(dist)
        Hz = code.Hz.toarray().astype(np.int32)
        Lz = code.logicals_z[:, Hz.shape[1] :]
        for p in ps:
            print(f"p={p}")
            sim1 = QSS_SimulatorV2(
                H=Hz,
                L=Lz,
                per=p,
                ser=p,
                bias=[1.0, 0.0, 0.0],
                codename=codename,
                decoding_method="lsd",
                check_side="Z",
                analog_tg=False,
                rounds=(decoding_rds + 1)
                * dist,  # how often to decode, i.e., how often we slide window-1
                repetitions=2
                * dist,  # how many noisy syndromes, == window size == 2 * region_size
                experiment="test",
                bp_params=BpParams(max_bp_iter=5),
            )
            sim2 = QSS_SimulatorV2(
                H=Hz,
                L=Lz,
                per=p,
                ser=p,
                bias=[1.0, 0.0, 0.0],
                codename=codename,
                decoding_method="matching",
                check_side="Z",
                analog_tg=False,
                rounds=(decoding_rds + 1)
                * dist,  # how often to decode, i.e., how often we slide window-1
                repetitions=2
                * dist,  # how many noisy syndromes, == window size == 2 * region_size
                experiment="test",
                bp_params=BpParams(max_bp_iter=50),
            )
            out1 = sim1.run(samples=nr_samples)
            out2 = sim2.run(samples=nr_samples)
            lsd_lers.append(out1["x_ler"])
            osd_lers.append(out2["x_ler"])
        axis[0].plot(
            ps,
            lsd_lers,
            label=f"lsd d={dist}",
            marker="o",
            linestyle="dashed",
        )
        axis[1].plot(
            ps,
            osd_lers,
            label=f"bposd d={dist}",
            marker="x",
            linestyle="solid",
        )
    axis[0].legend()
    axis[1].legend()
    axis[0].set_xlabel("p")
    axis[0].set_ylabel("LER")
    axis[0].set_yscale("log")
    fig.savefig(f"code-{codename}")
    fig.show()
