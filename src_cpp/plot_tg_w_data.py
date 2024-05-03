# a python file for plotting bp and cluster data in a surface code tanner graph

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import private_lib as pl
import matplotlib.patheffects as pe


def check_marker(valid=False, figsize=None):
    if not valid:
        return {"marker": "s", "facecolor": "C0", "edgecolor": "black", "s": 50}
    else:
        return {"marker": "s", "facecolor": "C1", "edgecolor": "black", "s": 50}


def bit_marker(color, llr):
    return {
        "marker": "o",
        "facecolor": pl.lighten_color(color, llr),
        "edgecolor": "black",
        "s": 50,
    }


def main(L, syndrome, llrs, cluster_bits, figsize=None):
    check = {"marker": "s", "facecolor": "white", "edgecolor": "black", "s": 50}
    bit = {"marker": "o", "facecolor": "white", "edgecolor": "black", "s": 50}
    cluster_line = {
        "color": "C2",
        "zorder": 0,
        "linewidth": 1.5,
        "path_effects": [
            pe.withStroke(linewidth=1.5, foreground="C2"),
            pe.SimplePatchShadow(offset=(1, -1), shadow_rgbFace="C2", alpha=0.75),
            pe.Normal(),
        ],
    }
    grid_line = {
        "color": "black",
        "zorder": 0,
        "linewidth": 1,
        "path_effects": [
            pe.withStroke(linewidth=1, foreground="black"),
            pe.SimplePatchShadow(offset=(1, -1), shadow_rgbFace="black", alpha=0.75),
            pe.Normal(),
        ],
    }

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # place sector one bits at (2 * n, 2 * m)
    bit_positions = []
    idx = 1
    for m in range(L):
        for n in range(L):
            bit_positions.append((-2 * n, -2 * m))
            ax.scatter(-2 * n, -2 * m, **bit_marker("C0", llrs[idx - 1]))
            # write label on bit
            ax.text(
                -2 * n,
                -2 * m + 0.3,
                f"{idx}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
            )
            idx += 1

    # place sector two bits at (-2 * n - 1, -2 * m - 1)
    for m in range(L - 1):
        for n in range(L - 1):
            bit_positions.append((-2 * n - 1, -2 * m - 1))
            ax.scatter(-2 * n - 1, -2 * m - 1, **bit_marker("C0", llrs[idx - 1]))
            # write label on bit
            ax.text(
                -2 * n - 1 + 0.35,
                -2 * m - 1,
                f"{idx}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="C0",
            )
            idx += 1

    check_positions = []
    check_idx = 0

    # place checks at (-2 * n - 1, -2 * m )
    for m in range(L):
        for n in range(L - 1):
            check_positions.append((-2 * n - 1, -2 * m))
            if syndrome[check_idx] == 1:
                ax.scatter(-2 * n - 1, -2 * m, **check_marker(False))
            else:
                ax.scatter(-2 * n - 1, -2 * m, **check)
            check_idx += 1

    # draw horizontal lines
    for n in range(2 * L):
        for m in range(L):
            if n > 0:
                if n != 2 * L - 1:
                    ax.plot(
                        [-n, -n + 1],
                        [-2 * m, -2 * m],
                        **grid_line,
                    )

    # draw vertical lines
    for n in range(1, L):
        for m in range(2 * L):
            if m > 0:
                if m != 2 * L - 1:
                    ax.plot(
                        [-2 * n + 1, -2 * n + 1],
                        [-m, -m + 1],
                        **grid_line,
                    )

    # overwrite with cluster lines
    for idx, pos in enumerate(bit_positions):
        idx = idx + 1
        if idx in cluster_bits:
            if idx <= L**2:
                if idx % L != 1:
                    ax.plot(
                        [pos[0], pos[0] + 1],
                        [pos[1], pos[1]],
                        **cluster_line,
                    )
                if idx % L != 0:
                    ax.plot(
                        [pos[0], pos[0] - 1],
                        [pos[1], pos[1]],
                        **cluster_line,
                    )
            else:
                ax.plot(
                    [pos[0], pos[0]],
                    [pos[1], pos[1] - 1],
                    **cluster_line,
                )

                ax.plot(
                    [pos[0], pos[0]],
                    [pos[1], pos[1] + 1],
                    **cluster_line,
                )

    ax.set_xlim(1, -2 * L)
    ax.set_axis_off()


if __name__ == "__main__":
    L = 5
    syndrome = np.zeros(L * (L - 1))
    llrs = np.zeros(2 * L * (L - 1) + 1)
    llrs = np.random.rand(len(llrs))
    cluster_bits = [2, 5, 10, 11, 8]
    fig = main(L, syndrome, llrs, cluster_bits, figsize=(6, 4))

    plt.show()
