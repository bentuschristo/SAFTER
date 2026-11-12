#!/usr/bin/env python3
"""
Plot local-sensing evaluation summaries for Go2 safe navigation.

Expected input:
  combined_summary_by_map.csv

Example:
  python scripts/plot_local_sensing_eval_summary_v2.py \
    --csv "$HOME/Downloads/go2_local_sensing_eval_20260707/go2_local_sensing_eval_20260707/20260707_145922_seed42_20permap/combined_summary_by_map.csv" \
    --out-dir "$HOME/Downloads/go2_local_sensing_eval_20260707/figures_local_sensing_v2"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "no_cbf": "No-CBF",
    "cbf_rl": "CBF-CF",
    "bcbf_rl": "bCBF-CF",
}

METHOD_ORDER = ["no_cbf", "cbf_rl", "bcbf_rl"]
RADIUS_ORDER = ["R2", "R3", "R4", "R5", "full"]
RADIUS_LABELS = {
    "R2": "2 m",
    "R3": "3 m",
    "R4": "4 m",
    "R5": "5 m",
    "full": "Full",
}


REQUIRED_COLUMNS = {
    "method",
    "map_name",
    "sensing_radius_label",
    "success_rate",
    "collision_rate",
    "timeout_rate",
}


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, REQUIRED_COLUMNS)
    df = df.copy()
    df = df[df["method"].isin(METHOD_ORDER)]
    df = df[df["sensing_radius_label"].isin(RADIUS_ORDER)]

    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["radius_plot_label"] = df["sensing_radius_label"].map(RADIUS_LABELS)
    df["sensing_radius_label"] = pd.Categorical(
        df["sensing_radius_label"], categories=RADIUS_ORDER, ordered=True
    )
    return df


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_over_sensing(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    percent: bool = False,
) -> None:
    if metric not in df.columns:
        print(f"[skip] {metric}: column not found")
        return

    agg = (
        df.groupby(["sensing_radius_label", "method"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(RADIUS_ORDER))

    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].set_index("sensing_radius_label")
        y = [sub.loc[r, metric] if r in sub.index else np.nan for r in RADIUS_ORDER]
        if percent:
            y = [100.0 * v for v in y]
        ax.plot(x, y, marker="o", linewidth=2, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([RADIUS_LABELS[r] for r in RADIUS_ORDER])
    ax.set_xlabel("Sensing radius")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    save_figure(fig, out_path)


def plot_metric_bar_by_method(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    percent: bool = False,
) -> None:
    if metric not in df.columns:
        print(f"[skip] {metric}: column not found")
        return

    agg = df.groupby("method", observed=True)[metric].mean().reindex(METHOD_ORDER)
    values = agg.values.astype(float)
    if percent:
        values = 100.0 * values

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)

    for bar, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{v:.1f}" if percent else f"{v:.3f}",
            ha="center",
            va="bottom",
        )

    save_figure(fig, out_path)


def plot_per_map_success(df: pd.DataFrame, out_path: Path) -> None:
    maps = sorted(df["map_name"].dropna().unique())
    if not maps:
        print("[skip] per-map success: no map_name values")
        return

    fig, axes = plt.subplots(len(maps), 1, figsize=(7.2, 2.3 * len(maps)), sharex=True)
    if len(maps) == 1:
        axes = [axes]

    x = np.arange(len(RADIUS_ORDER))
    for ax, map_name in zip(axes, maps):
        mdf = df[df["map_name"] == map_name]
        for method in METHOD_ORDER:
            sub = mdf[mdf["method"] == method].set_index("sensing_radius_label")
            y = [
                100.0 * sub.loc[r, "success_rate"] if r in sub.index else np.nan
                for r in RADIUS_ORDER
            ]
            ax.plot(x, y, marker="o", linewidth=2, label=METHOD_LABELS[method])
        ax.set_ylabel("Success (%)")
        ax.set_title(str(map_name))
        ax.grid(True, alpha=0.35)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([RADIUS_LABELS[r] for r in RADIUS_ORDER])
    axes[-1].set_xlabel("Sensing radius")
    axes[0].legend(frameon=True, ncol=3)
    save_figure(fig, out_path)


def plot_failure_breakdown_by_method(df: pd.DataFrame, out_path: Path) -> None:
    """Stacked outcome rates averaged over all maps and sensing radii."""
    agg = (
        df.groupby("method", observed=True)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            timeout_rate=("timeout_rate", "mean"),
        )
        .reindex(METHOD_ORDER)
    )

    agg["other_rate"] = 1.0 - (
        agg["success_rate"] + agg["collision_rate"] + agg["timeout_rate"]
    )
    agg["other_rate"] = agg["other_rate"].clip(lower=0.0)

    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    success = 100.0 * agg["success_rate"].values
    collision = 100.0 * agg["collision_rate"].values
    timeout = 100.0 * agg["timeout_rate"].values
    other = 100.0 * agg["other_rate"].values

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(labels, success, label="Success")
    ax.bar(labels, collision, bottom=success, label="Collision")
    ax.bar(labels, timeout, bottom=success + collision, label="Timeout")

    if np.nanmax(other) > 1e-6:
        ax.bar(labels, other, bottom=success + collision + timeout, label="Other")

    ax.set_ylabel("Episode outcome rate (%)")
    ax.set_title("Aggregate episode outcome breakdown")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True, ncol=2)
    save_figure(fig, out_path)


def plot_outcome_breakdown_vs_sensing(df: pd.DataFrame, out_path: Path) -> None:
    """One panel per method, showing success/collision/timeout over sensing radius."""
    fig, axes = plt.subplots(len(METHOD_ORDER), 1, figsize=(7.2, 2.6 * len(METHOD_ORDER)), sharex=True)
    if len(METHOD_ORDER) == 1:
        axes = [axes]

    x = np.arange(len(RADIUS_ORDER))
    outcome_specs = [
        ("success_rate", "Success"),
        ("collision_rate", "Collision"),
        ("timeout_rate", "Timeout"),
    ]

    for ax, method in zip(axes, METHOD_ORDER):
        mdf = df[df["method"] == method]
        agg = mdf.groupby("sensing_radius_label", observed=True).mean(numeric_only=True)
        for col, label in outcome_specs:
            y = [100.0 * agg.loc[r, col] if r in agg.index else np.nan for r in RADIUS_ORDER]
            ax.plot(x, y, marker="o", linewidth=2, label=label)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.35)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([RADIUS_LABELS[r] for r in RADIUS_ORDER])
    axes[-1].set_xlabel("Sensing radius")
    axes[0].legend(frameon=True, ncol=3)
    save_figure(fig, out_path)


def save_aggregate_table(df: pd.DataFrame, out_path: Path) -> None:
    agg_spec = {
        "success_rate": ("success_rate", "mean"),
        "collision_rate": ("collision_rate", "mean"),
        "timeout_rate": ("timeout_rate", "mean"),
    }

    optional_cols = [
        "mean_minimum_clearance_m",
        "mean_success_goal_time_s",
        "mean_filter_active_fraction",
        "mean_command_correction_norm",
        "mean_visible_obstacles",
    ]
    for col in optional_cols:
        if col in df.columns:
            agg_spec[col] = (col, "mean")

    summary = df.groupby("method", observed=True).agg(**agg_spec).reindex(METHOD_ORDER)
    summary.index = [METHOD_LABELS[m] for m in METHOD_ORDER]
    summary.to_csv(out_path)
    print("\nAggregate summary:")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = prepare(df)

    # Primary performance and safety trends.
    plot_metric_over_sensing(
        df,
        metric="success_rate",
        ylabel="Success rate (%)",
        title="Success rate vs sensing radius",
        out_path=args.out_dir / "success_rate_vs_sensing_radius",
        percent=True,
    )
    plot_metric_over_sensing(
        df,
        metric="collision_rate",
        ylabel="Collision rate (%)",
        title="Collision rate vs sensing radius",
        out_path=args.out_dir / "collision_rate_vs_sensing_radius",
        percent=True,
    )
    plot_metric_over_sensing(
        df,
        metric="timeout_rate",
        ylabel="Timeout rate (%)",
        title="Timeout rate vs sensing radius",
        out_path=args.out_dir / "timeout_rate_vs_sensing_radius",
        percent=True,
    )

    # Optional diagnostics if available.
    plot_metric_over_sensing(
        df,
        metric="mean_minimum_clearance_m",
        ylabel="Mean minimum clearance (m)",
        title="Minimum clearance vs sensing radius",
        out_path=args.out_dir / "minimum_clearance_vs_sensing_radius",
    )
    plot_metric_over_sensing(
        df,
        metric="mean_success_goal_time_s",
        ylabel="Goal time for successful trials (s)",
        title="Goal time vs sensing radius",
        out_path=args.out_dir / "goal_time_vs_sensing_radius",
    )
    plot_metric_over_sensing(
        df,
        metric="mean_filter_active_fraction",
        ylabel="Filter active fraction",
        title="Filter intervention vs sensing radius",
        out_path=args.out_dir / "filter_active_fraction_vs_sensing_radius",
    )
    plot_metric_over_sensing(
        df,
        metric="mean_visible_obstacles",
        ylabel="Mean visible obstacles",
        title="Visible obstacles vs sensing radius",
        out_path=args.out_dir / "visible_obstacles_vs_sensing_radius",
    )

    # Aggregate method summaries.
    plot_metric_bar_by_method(
        df,
        metric="success_rate",
        ylabel="Mean success rate (%)",
        title="Aggregate success rate",
        out_path=args.out_dir / "aggregate_success_rate_by_method",
        percent=True,
    )
    plot_metric_bar_by_method(
        df,
        metric="collision_rate",
        ylabel="Mean collision rate (%)",
        title="Aggregate collision rate",
        out_path=args.out_dir / "aggregate_collision_rate_by_method",
        percent=True,
    )
    plot_metric_bar_by_method(
        df,
        metric="timeout_rate",
        ylabel="Mean timeout rate (%)",
        title="Aggregate timeout rate",
        out_path=args.out_dir / "aggregate_timeout_rate_by_method",
        percent=True,
    )

    # Failure-mode explanation plots.
    plot_failure_breakdown_by_method(
        df,
        out_path=args.out_dir / "aggregate_episode_outcome_breakdown",
    )
    plot_outcome_breakdown_vs_sensing(
        df,
        out_path=args.out_dir / "outcome_breakdown_vs_sensing_radius",
    )

    # Per-map success panel.
    plot_per_map_success(
        df,
        out_path=args.out_dir / "per_map_success_rate_vs_sensing_radius",
    )

    save_aggregate_table(df, args.out_dir / "aggregate_summary_by_method.csv")
    print("\nSaved plots to:", args.out_dir)


if __name__ == "__main__":
    main()
