#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["no_cbf", "cbf_rl", "bcbf_rl"]

METHOD_NAMES = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-RL",
    "bcbf_rl": "bCBF-RL",
}


def find_method_summary(result_dir: Path, method: str) -> Path:
    """Find a method summary in either a normal or unseen-map result tree."""
    candidates = [
        result_dir / method / f"{method}_summary.csv",
        result_dir / "shifted" / method / f"{method}_summary.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    recursive_matches = list(
        result_dir.glob(f"**/{method}/{method}_summary.csv")
    )

    if len(recursive_matches) == 1:
        return recursive_matches[0]

    if not recursive_matches:
        raise FileNotFoundError(
            f"Could not find {method}_summary.csv under {result_dir}"
        )

    raise RuntimeError(
        f"Multiple summaries found for {method} under {result_dir}:\n"
        + "\n".join(str(path) for path in recursive_matches)
    )


def load_results(result_dir: Path) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        summary_path = find_method_summary(result_dir, method)
        dataframe = pd.read_csv(summary_path)

        if len(dataframe) != 1:
            raise ValueError(
                f"Expected one row in {summary_path}, found {len(dataframe)}"
            )

        row = dataframe.iloc[0]

        collision_rate = (
            float(row.get("collision_rate", 0.0))
            + float(row.get("outside_rate", 0.0))
            + float(row.get("fall_rate", 0.0))
        )

        success_rate = float(row.get("success_rate", 0.0))
        timeout_rate = float(row.get("timeout_rate", 0.0))

        classified_total = success_rate + collision_rate + timeout_rate

        rows.append(
            {
                "method": method,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "timeout_rate": timeout_rate,
                "classified_total": classified_total,
                "summary_path": str(summary_path),
            }
        )

    return pd.DataFrame(rows)


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()

        if value <= 0:
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig, output_stem: Path, formats: list[str], dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    for extension in formats:
        output_path = output_stem.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")


def plot_environment(
    dataframe: pd.DataFrame,
    *,
    title: str,
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    method_labels = [
        METHOD_NAMES[method]
        for method in dataframe["method"]
    ]

    x = np.arange(len(method_labels))
    width = 0.23

    success = 100.0 * dataframe["success_rate"].to_numpy()
    collision = 100.0 * dataframe["collision_rate"].to_numpy()
    timeout = 100.0 * dataframe["timeout_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    success_bars = ax.bar(
        x - width,
        success,
        width,
        label="Success",
    )

    collision_bars = ax.bar(
        x,
        collision,
        width,
        label="Collision",
    )

    timeout_bars = ax.bar(
        x + width,
        timeout,
        width,
        label="Timeout",
    )

    annotate_bars(ax, success_bars)
    annotate_bars(ax, collision_bars)
    annotate_bars(ax, timeout_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Outcome rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )

    fig.tight_layout()
    save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)


def plot_success_comparison(
    original: pd.DataFrame,
    shifted: pd.DataFrame,
    *,
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    method_labels = [
        METHOD_NAMES[method]
        for method in METHOD_ORDER
    ]

    x = np.arange(len(method_labels))
    width = 0.34

    original_success = (
        100.0
        * original.set_index("method")
        .loc[METHOD_ORDER, "success_rate"]
        .to_numpy()
    )

    shifted_success = (
        100.0
        * shifted.set_index("method")
        .loc[METHOD_ORDER, "success_rate"]
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    original_bars = ax.bar(
        x - width / 2,
        original_success,
        width,
        label="Original environment",
    )

    shifted_bars = ax.bar(
        x + width / 2,
        shifted_success,
        width,
        label="Shifted environment",
    )

    annotate_bars(ax, original_bars)
    annotate_bars(ax, shifted_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Navigation Success Across Environments")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot original and shifted evaluation statistics for "
            "the server-trained policies."
        )
    )

    parser.add_argument(
        "--original-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shifted-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/evaluation/server_policyseed42"),
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original = load_results(args.original_dir)
    shifted = load_results(args.shifted_dir)

    print("\nOriginal environment:")
    print(
        original[
            [
                "method",
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "classified_total",
            ]
        ].to_string(index=False)
    )

    print("\nShifted environment:")
    print(
        shifted[
            [
                "method",
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "classified_total",
            ]
        ].to_string(index=False)
    )

    plot_environment(
        original,
        title="Navigation Performance in Original Environment",
        output_stem=args.output_dir / "original_environment",
        formats=args.formats,
        dpi=args.dpi,
    )

    plot_environment(
        shifted,
        title="Navigation Performance in Shifted Environment",
        output_stem=args.output_dir / "shifted_environment",
        formats=args.formats,
        dpi=args.dpi,
    )

    plot_success_comparison(
        original,
        shifted,
        output_stem=args.output_dir / "success_comparison",
        formats=args.formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
