import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
    }
)


r = 30.0
R = 50.0
v = 2.0

t_max = 100.0
n_t = 2001
N_MC = 50000

t_int_max = 1000.0
n_int = 50000
seed = 20260608

colors = ["black", "red", "blue", "magenta"]

alpha_s = 0.055
alpha_c = 0.120

T0 = 10.0
gamma_T = 0.65


def f_T(t, R_x, v=v):
    t = np.asarray(t, dtype=float)
    f = np.zeros_like(t)

    eps = 1e-12
    idx = t > eps
    tv = t[idx]

    den = np.maximum(np.abs(R_x - v * tv), eps)
    log_term = np.log(np.abs((R_x + v * tv) / den))

    f[idx] = (
        1.0
        / (math.pi**2 * R_x**2 * tv)
        * (
            4.0 * R_x * v * tv
            + 2.0
            * (R_x - v * tv)
            * (R_x + v * tv)
            * log_term
        )
    )

    f[~np.isfinite(f)] = 0.0
    f[f < 0.0] = 0.0
    return f


def build_F_T(R_x):
    x = np.linspace(1e-4, t_int_max, n_int)
    f = f_T(x, R_x)

    area = np.trapz(f, x)
    if area <= 0.0:
        raise RuntimeError("Invalid distribution.")

    f /= area

    F = np.insert(
        np.cumsum(
            (f[:-1] + f[1:])
            * 0.5
            * np.diff(x)
        ),
        0,
        0.0,
    )

    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))

    F_unique, idx_unique = np.unique(F, return_index=True)

    return x[idx_unique], F_unique


T_s, F_s = build_F_T(r)
T_c, F_c = build_F_T(R)


def sample_T_min(rng, n, a, layer):
    if a <= 0.0:
        return np.full(n, np.inf)

    u = rng.random(n)
    F_single = 1.0 - np.power(1.0 - u, 1.0 / a)

    if layer == "s":
        x = np.interp(F_single, F_s, T_s)
    elif layer == "c":
        x = np.interp(F_single, F_c, T_c)
    else:
        raise ValueError("layer must be 's' or 'c'.")

    return T0 * np.power(x / T0, gamma_T)


def P_path(t, k12, k3, seed_i):
    rng = np.random.default_rng(seed_i)

    a_s = alpha_s * k12
    a_c = alpha_s * k12 + alpha_c * k3

    T_min_s = sample_T_min(
        rng,
        N_MC,
        a_s,
        "s",
    )

    T_min_c = sample_T_min(
        rng,
        N_MC,
        a_c,
        "c",
    )

    T_path = np.minimum(T_min_s, T_min_c)

    return np.mean(
        T_path[:, None] > t[None, :],
        axis=0,
    )


def y_formatter(y, _):
    if abs(y - round(y)) < 1e-9:
        return str(int(round(y)))
    return f"{y:.1f}"


def style_axis(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    ax.set_xlabel("t(Seconds)")
    ax.set_ylabel("Path stability probability")
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        width=0.8,
    )


def main():
    t = np.linspace(0.0, t_max, n_t)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.6),
    )

    seed_sequence = np.random.SeedSequence(seed)
    seeds = seed_sequence.spawn(8)

    k12 = 10
    k3_values = [1, 2, 3, 4]

    for i, (color, k3) in enumerate(zip(colors, k3_values)):
        y = P_path(
            t,
            k12=k12,
            k3=k3,
            seed_i=seeds[i],
        )

        axes[0].plot(
            t,
            y,
            color=color,
            linewidth=1.5,
            marker=".",
            markersize=2.0,
            markevery=18,
            label=rf"$k_3$={k3}",
        )

    style_axis(axes[0])

    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.63, 0.82),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
        borderpad=0.7,
    )

    axes[0].set_title(
        r"(a) Impact of the number of relay nodes $k_3$",
        y=-0.22,
        fontsize=14,
    )

    k3 = 2
    k12_values = [6, 7, 8, 9]

    for j, (color, k12) in enumerate(zip(colors, k12_values), start=4):
        y = P_path(
            t,
            k12=k12,
            k3=k3,
            seed_i=seeds[j],
        )

        axes[1].plot(
            t,
            y,
            color=color,
            linewidth=1.5,
            marker=".",
            markersize=2.0,
            markevery=18,
            label=rf"$k_1$+$k_2$={k12}",
        )

    style_axis(axes[1])

    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(0.60, 0.80),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
        borderpad=0.7,
    )

    axes[1].set_title(
        r"(b) Impact of the number of relay nodes $(k_1+k_2)$",
        y=-0.22,
        fontsize=14,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.97,
        bottom=0.19,
        wspace=0.15,
    )

    fig.savefig(
        "Fig7_path_stability_probability.svg",
        format="svg",
        bbox_inches="tight",
    )

    fig.savefig(
        "Fig7_path_stability_probability.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()