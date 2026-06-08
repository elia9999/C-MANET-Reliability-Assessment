import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


R_MEMBER = 30.0
R_CH = 50.0
RELATIVE_SPEED = 2.0
TAU = 0.01

T_MAX = 100.0
N_TIME_POINTS = 2000
N_MONTE_CARLO = 20000

T_INTEGRATION_MAX = 1000.0
N_INTEGRATION_GRID = 50000
RANDOM_SEED = 20260608

COLORS = ["black", "red", "blue", "magenta"]

SLOT_SUCCESS_PROBABILITY_A = 0.027
SLOT_SUCCESS_PROBABILITY_B = 0.017

MEMBER_LINK_EFFECTIVE_FACTOR_A = 0.15
CH_LINK_EFFECTIVE_FACTOR_A = 0.35

MEMBER_LINK_EFFECTIVE_FACTOR_B = 0.19
CH_LINK_EFFECTIVE_FACTOR_B = 0.18


def residual_time_pdf(t, communication_range, speed=RELATIVE_SPEED):
    t = np.asarray(t, dtype=float)
    pdf = np.zeros_like(t)

    eps = 1e-12
    valid = t > eps
    tv = t[valid]
    radius = float(communication_range)

    denominator = np.maximum(np.abs(radius - speed * tv), eps)
    log_term = np.log(np.abs((radius + speed * tv) / denominator))

    pdf[valid] = (
        1.0
        / (math.pi**2 * radius**2 * tv)
        * (
            4.0 * radius * speed * tv
            + 2.0
            * (radius - speed * tv)
            * (radius + speed * tv)
            * log_term
        )
    )

    pdf[~np.isfinite(pdf)] = 0.0
    pdf[pdf < 0.0] = 0.0

    return pdf


def build_inverse_cdf(communication_range):
    time_grid = np.linspace(
        1e-4,
        T_INTEGRATION_MAX,
        N_INTEGRATION_GRID,
    )

    pdf = residual_time_pdf(time_grid, communication_range)

    area = np.trapz(pdf, time_grid)

    if area <= 0.0:
        raise RuntimeError("Invalid residual-time PDF.")

    pdf /= area

    cdf = np.insert(
        np.cumsum(
            (pdf[:-1] + pdf[1:])
            * 0.5
            * np.diff(time_grid)
        ),
        0,
        0.0,
    )

    cdf = np.maximum.accumulate(
        np.clip(cdf, 0.0, 1.0)
    )

    unique_cdf, unique_indices = np.unique(
        cdf,
        return_index=True,
    )

    return time_grid[unique_indices], unique_cdf


T_MEMBER, CDF_MEMBER = build_inverse_cdf(R_MEMBER)
T_CH, CDF_CH = build_inverse_cdf(R_CH)


def sample_effective_minimum_lifetime(
    rng,
    sample_count,
    effective_link_count,
    link_type,
):
    if effective_link_count <= 0.0:
        return np.full(sample_count, np.inf)

    uniform = rng.random(sample_count)

    single_link_cdf = 1.0 - np.power(
        1.0 - uniform,
        1.0 / effective_link_count,
    )

    if link_type == "member":
        return np.interp(
            single_link_cdf,
            CDF_MEMBER,
            T_MEMBER,
        )

    if link_type == "ch":
        return np.interp(
            single_link_cdf,
            CDF_CH,
            T_CH,
        )

    raise ValueError("link_type must be 'member' or 'ch'.")


def get_panel_parameters(panel):
    if panel == "a":
        return (
            SLOT_SUCCESS_PROBABILITY_A,
            MEMBER_LINK_EFFECTIVE_FACTOR_A,
            CH_LINK_EFFECTIVE_FACTOR_A,
        )

    if panel == "b":
        return (
            SLOT_SUCCESS_PROBABILITY_B,
            MEMBER_LINK_EFFECTIVE_FACTOR_B,
            CH_LINK_EFFECTIVE_FACTOR_B,
        )

    raise ValueError("panel must be 'a' or 'b'.")


def simulate_transmission_success(
    t_values,
    k1_plus_k2,
    k3,
    panel,
    seed,
):
    rng = np.random.default_rng(seed)

    (
        slot_success_probability,
        member_link_factor,
        ch_link_factor,
    ) = get_panel_parameters(panel)

    total_hops = k1_plus_k2 + k3

    effective_member_links = member_link_factor * k1_plus_k2

    effective_ch_links = (
        member_link_factor * k1_plus_k2
        + ch_link_factor * k3
    )

    member_minimum_lifetimes = sample_effective_minimum_lifetime(
        rng,
        N_MONTE_CARLO,
        effective_member_links,
        "member",
    )

    ch_minimum_lifetimes = sample_effective_minimum_lifetime(
        rng,
        N_MONTE_CARLO,
        effective_ch_links,
        "ch",
    )

    path_lifetimes = np.minimum(
        member_minimum_lifetimes,
        ch_minimum_lifetimes,
    )

    failed_slots = rng.negative_binomial(
        total_hops,
        slot_success_probability,
        size=N_MONTE_CARLO,
    )

    required_slots = failed_slots + total_hops
    completion_times = required_slots * TAU

    success = (
        (completion_times[:, None] <= t_values[None, :])
        & (path_lifetimes[:, None] >= t_values[None, :])
    )

    return np.mean(success, axis=0)


def y_formatter(value, _):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))

    return f"{value:.1f}"


def style_axis(ax, y_max):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_max)

    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, y_max + 0.001, 0.1))

    ax.yaxis.set_major_formatter(
        FuncFormatter(y_formatter)
    )

    ax.set_xlabel("t(Seconds)")
    ax.set_ylabel("Transmission success rate")

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        width=0.8,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
        }
    )

    t_values = np.linspace(
        0.01,
        T_MAX,
        N_TIME_POINTS,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.6),
    )

    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    curve_seeds = seed_sequence.spawn(8)

    for index, (color, k3) in enumerate(
        zip(COLORS, [1, 2, 3, 4])
    ):
        y_values = simulate_transmission_success(
            t_values,
            k1_plus_k2=5,
            k3=k3,
            panel="a",
            seed=curve_seeds[index],
        )

        axes[0].plot(
            t_values,
            y_values,
            color=color,
            linewidth=1.5,
            label=rf"$k_3$={k3}",
        )

    style_axis(axes[0], 0.8)

    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.62, 0.80),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
    )

    axes[0].set_title(
        r"(a) Impact of the number of relay nodes $k_3$",
        y=-0.22,
        fontsize=14,
    )

    for index, (color, k12) in enumerate(
        zip(COLORS, [4, 6, 8, 10]),
        start=4,
    ):
        y_values = simulate_transmission_success(
            t_values,
            k1_plus_k2=k12,
            k3=2,
            panel="b",
            seed=curve_seeds[index],
        )

        axes[1].plot(
            t_values,
            y_values,
            color=color,
            linewidth=1.5,
            label=rf"$k_1$+$k_2$={k12}",
        )

    style_axis(axes[1], 0.6)

    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(0.57, 0.80),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10,
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
        "Figure8_forward_simulation.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        "Figure8_forward_simulation.svg",
        format="svg",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()