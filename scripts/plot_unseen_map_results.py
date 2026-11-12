from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "logs/evaluation/unseen_maps/"
    "20260628_005743_seed10_100permap"
)
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "combined_summary_by_map.csv"

METHOD_LABELS = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-RL",
    "bcbf_rl": "bCBF-RL",
}

METHOD_ORDER = ["no_cbf", "cbf_rl", "bcbf_rl"]


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    df["_order"] = df["method"].map(order)
    return df.sort_values("_order").reset_index(drop=True)


def annotate_bars(ax: plt.Axes, bars, suffix: str = "%") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.0f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_single_rate(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    labels = df["method_label"].tolist()
    values = 100.0 * df[column].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, bars)

    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_outcomes(df: pd.DataFrame) -> None:
    labels = df["method_label"].tolist()
    success = 100.0 * df["success_rate"].to_numpy()
    collision = 100.0 * df["collision_rate"].to_numpy()
    timeout = 100.0 * df["timeout_rate"].to_numpy()

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    bars_success = ax.bar(x - width, success, width, label="Success")
    bars_collision = ax.bar(x, collision, width, label="Collision")
    bars_timeout = ax.bar(x + width, timeout, width, label="Timeout")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Navigation Performance in a New Environment")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    annotate_bars(ax, bars_success)
    annotate_bars(ax, bars_collision)
    annotate_bars(ax, bars_timeout)

    fig.tight_layout()
    fig.savefig(OUT / "unseen_map_outcomes_grouped.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "unseen_map_outcomes_grouped.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_secondary_metrics(df: pd.DataFrame) -> None:
    labels = df["method_label"].tolist()
    clearance = df["mean_minimum_clearance_m"].to_numpy()
    path_length = df["mean_success_path_length_m"].to_numpy()
    goal_time = df["mean_success_goal_time_s"].to_numpy()

    plots = [
        (
            clearance,
            "Mean minimum clearance (m)",
            "Unseen-map minimum clearance",
            "unseen_map_mean_clearance",
        ),
        (
            path_length,
            "Successful path length (m)",
            "Successful path length on unseen map",
            "unseen_map_success_path_length",
        ),
        (
            goal_time,
            "Successful goal time (s)",
            "Successful goal time on unseen map",
            "unseen_map_success_goal_time",
        ),
    ]

    for values, ylabel, title, filename in plots:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        bars = ax.bar(labels, values)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        upper = max(values) * 1.25 if max(values) > 0 else 1.0
        ax.set_ylim(0, upper)

        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02 * upper,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_plot_data(df: pd.DataFrame) -> None:
    columns = [
        "map_name",
        "method",
        "method_label",
        "cases",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_success_path_length_m",
        "mean_success_goal_time_s",
        "mean_minimum_clearance_m",
        "mean_filter_active_fraction",
        "mean_command_correction_norm",
    ]
    df[columns].to_csv(OUT / "plot_data.csv", index=False)


def main() -> None:
    df = load_summary()

    plot_single_rate(
        df,
        "success_rate",
        "Success rate (%)",
        "Success rate on unseen map",
        "unseen_map_success_rate",
    )
    plot_single_rate(
        df,
        "collision_rate",
        "Collision rate (%)",
        "Collision rate on unseen map",
        "unseen_map_collision_rate",
    )
    plot_single_rate(
        df,
        "timeout_rate",
        "Timeout rate (%)",
        "Timeout rate on unseen map",
        "unseen_map_timeout_rate",
    )

    plot_grouped_outcomes(df)
    plot_secondary_metrics(df)
    write_plot_data(df)

    print(f"Figures written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
