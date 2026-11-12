#!/usr/bin/env python3
"""Plot dynamic-obstacle position-only evaluation results.

Expected input:
  RUN_DIR/combined_summary_by_map.csv

Generated figures:
  dynamic_outcomes_by_map
  dynamic_success_rate_by_map
  dynamic_collision_rate_by_map
  dynamic_min_clearance_by_map
  dynamic_filter_activity_by_map
  dynamic_command_correction_by_map
  dynamic_goal_time_by_map
  dynamic_path_length_by_map
  dynamic_aggregate_outcomes
  dynamic_aggregate_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["no_cbf", "cbf_rl", "bcbf_rl"]
METHOD_LABELS = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-CF",
    "bcbf_rl": "bCBF-CF",
}
MAP_ORDER = ["bottleneck", "clustered", "corridor", "shifted", "slalom"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def save(fig: plt.Figure, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["map_rank"] = out["map_name"].map({m: i for i, m in enumerate(MAP_ORDER)})
    out["method_rank"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    out["method_label"] = out["method"].map(METHOD_LABELS)
    return out.sort_values(["map_rank", "method_rank"])


def grouped_by_map(
    df: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output: Path,
    formats: list[str],
    dpi: int,
    percent: bool = False,
) -> None:
    pivot = (
        df.pivot(index="map_name", columns="method", values=metric)
        .reindex([m for m in MAP_ORDER if m in set(df["map_name"])])
    )

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, method in enumerate(METHOD_ORDER):
        if method not in pivot.columns:
            continue
        values = pivot[method].to_numpy(dtype=float)
        if percent:
            values = 100.0 * values
        ax.bar(x + (i - 1) * width, values, width, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([str(m).capitalize() for m in pivot.index])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, output, formats, dpi)


def outcomes_panel(df: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    """One compact 3-row figure for success/collision/timeout across maps."""
    maps = [m for m in MAP_ORDER if m in set(df["map_name"])]
    metrics = [
        ("success_rate", "Success rate (%)"),
        ("collision_rate", "Collision rate (%)"),
        ("timeout_rate", "Timeout rate (%)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    x = np.arange(len(maps))
    width = 0.24

    for ax, (metric, ylabel) in zip(axes, metrics):
        pivot = df.pivot(index="map_name", columns="method", values=metric).reindex(maps)
        for i, method in enumerate(METHOD_ORDER):
            if method not in pivot.columns:
                continue
            ax.bar(
                x + (i - 1) * width,
                100.0 * pivot[method].to_numpy(dtype=float),
                width,
                label=METHOD_LABELS[method],
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0.0)

    axes[0].set_title("Dynamic-obstacle position-only outcomes by map")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([m.capitalize() for m in maps])
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    save(fig, out / "dynamic_outcomes_by_map", formats, dpi)


def aggregate(df: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    agg = (
        df.groupby("method", as_index=False)
        .agg(
            cases=("cases", "sum"),
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            timeout_rate=("timeout_rate", "mean"),
            fall_rate=("fall_rate", "mean"),
            outside_rate=("outside_rate", "mean"),
            mean_success_path_length_m=("mean_success_path_length_m", "mean"),
            mean_success_goal_time_s=("mean_success_goal_time_s", "mean"),
            mean_minimum_dynamic_clearance_m=("mean_minimum_dynamic_clearance_m", "mean"),
            minimum_dynamic_clearance_overall_m=("minimum_dynamic_clearance_overall_m", "min"),
            mean_filter_active_fraction=("mean_filter_active_fraction", "mean"),
            mean_command_correction_norm=("mean_command_correction_norm", "mean"),
        )
    )
    agg["method_rank"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    agg["method_label"] = agg["method"].map(METHOD_LABELS)
    agg = agg.sort_values("method_rank")
    agg.to_csv(out / "dynamic_aggregate_summary.csv", index=False)

    x = np.arange(len(agg))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (metric, label) in enumerate(
        [
            ("success_rate", "Success"),
            ("collision_rate", "Collision"),
            ("timeout_rate", "Timeout"),
        ]
    ):
        ax.bar(x + (i - 1) * width, 100.0 * agg[metric], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["method_label"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Aggregate dynamic-obstacle outcomes")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, out / "dynamic_aggregate_outcomes", formats, dpi)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    summary = run_dir / "combined_summary_by_map.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)

    out = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df = ordered(df)
    df.to_csv(out / "dynamic_summary_copy.csv", index=False)

    outcomes_panel(df, out, args.formats, args.dpi)

    grouped_by_map(
        df, "success_rate",
        title="Dynamic-obstacle success rate by map",
        ylabel="Success rate (%)",
        output=out / "dynamic_success_rate_by_map",
        formats=args.formats,
        dpi=args.dpi,
        percent=True,
    )
    grouped_by_map(
        df, "collision_rate",
        title="Dynamic-obstacle collision rate by map",
        ylabel="Collision rate (%)",
        output=out / "dynamic_collision_rate_by_map",
        formats=args.formats,
        dpi=args.dpi,
        percent=True,
    )
    grouped_by_map(
        df, "timeout_rate",
        title="Dynamic-obstacle timeout rate by map",
        ylabel="Timeout rate (%)",
        output=out / "dynamic_timeout_rate_by_map",
        formats=args.formats,
        dpi=args.dpi,
        percent=True,
    )

    grouped_by_map(
        df, "mean_minimum_dynamic_clearance_m",
        title="Mean minimum dynamic clearance by map",
        ylabel="Clearance (m)",
        output=out / "dynamic_mean_minimum_clearance_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )
    grouped_by_map(
        df, "minimum_dynamic_clearance_overall_m",
        title="Worst-case dynamic clearance by map",
        ylabel="Clearance (m)",
        output=out / "dynamic_worst_clearance_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )
    grouped_by_map(
        df, "mean_filter_active_fraction",
        title="Safety-filter activity by map",
        ylabel="Active fraction",
        output=out / "dynamic_filter_activity_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )
    grouped_by_map(
        df, "mean_command_correction_norm",
        title="Mean command correction by map",
        ylabel="Command correction norm",
        output=out / "dynamic_command_correction_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )
    grouped_by_map(
        df, "mean_success_goal_time_s",
        title="Successful goal time by map",
        ylabel="Time (s)",
        output=out / "dynamic_goal_time_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )
    grouped_by_map(
        df, "mean_success_path_length_m",
        title="Successful path length by map",
        ylabel="Path length (m)",
        output=out / "dynamic_path_length_by_map",
        formats=args.formats,
        dpi=args.dpi,
    )

    aggregate(df, out, args.formats, args.dpi)

    print(f"Saved dynamic evaluation figures to: {out}")


if __name__ == "__main__":
    main()
