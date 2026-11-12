#!/usr/bin/env python3
"""Plot unseen-map layouts and optional trajectories with safe/backup-set shading.

Shading convention:
  - Blue: backup set C_B
  - Green: safe-only band C_S \ C_B
  - Red: robot-clearance / unsafe region
  - Gray: physical obstacle

Examples
--------
All five layouts:
    python scripts/plot_unseen_maps_with_sets.py \
      --run-dir logs/evaluation/unseen_maps/20260703_022535_seed42_20permap \
      --output-dir logs/evaluation/figures_v4/maps_with_sets

One trajectory case:
    python scripts/plot_unseen_maps_with_sets.py \
      --run-dir logs/evaluation/unseen_maps/20260703_105125_seed42_20permap \
      --maps corridor \
      --case-id 3 \
      --output-dir logs/evaluation/figures_v4/trajectory_sets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd


MAP_ORDER = ["shifted", "corridor", "clustered", "slalom", "bottleneck"]
METHODS = ["no_cbf", "cbf_rl", "bcbf_rl"]
METHOD_LABELS = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-CF",
    "bcbf_rl": "bCBF-CF",
}
METHOD_COLORS = {
    "no_cbf": "#555555",
    "cbf_rl": "#1f77b4",
    "bcbf_rl": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--maps", nargs="+", choices=MAP_ORDER, default=MAP_ORDER)
    p.add_argument("--case-id", type=int)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--robot-radius", type=float, default=0.30)
    p.add_argument("--backup-margin", type=float, default=0.15)
    p.add_argument("--arena-limit", type=float, default=7.5)
    p.add_argument("--grid-points", type=int, default=500)
    return p.parse_args()


def compute_set_masks(
    map_df: pd.DataFrame,
    *,
    arena_limit: float,
    robot_radius: float,
    backup_margin: float,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(-arena_limit, arena_limit, grid_points)
    y_values = np.linspace(-arena_limit, arena_limit, grid_points)
    xx, yy = np.meshgrid(x_values, y_values)

    safe_clearance = np.full_like(xx, np.inf, dtype=float)
    backup_clearance = np.full_like(xx, np.inf, dtype=float)

    for obstacle in map_df.itertuples(index=False):
        distance = np.hypot(
            xx - float(obstacle.x),
            yy - float(obstacle.y),
        )
        safe_radius = float(obstacle.radius) + robot_radius
        backup_radius = safe_radius + backup_margin

        safe_clearance = np.minimum(safe_clearance, distance - safe_radius)
        backup_clearance = np.minimum(backup_clearance, distance - backup_radius)

    safe_mask = safe_clearance >= 0.0
    backup_mask = backup_clearance >= 0.0
    return xx, yy, safe_mask, backup_mask


def draw_sets(
    ax: plt.Axes,
    map_df: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    safe_mask: np.ndarray,
    backup_mask: np.ndarray,
    *,
    robot_radius: float,
    backup_margin: float,
) -> None:
    safe_only_mask = safe_mask & (~backup_mask)

    # Backup set: free space satisfying the more conservative margin.
    ax.contourf(
        xx, yy, backup_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#7aa6e8"],
        alpha=0.16,
        zorder=0,
    )

    # Safe-only band between safe and backup boundaries.
    ax.contourf(
        xx, yy, safe_only_mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#39ff88"],
        alpha=0.90,
        zorder=1,
    )

    for obstacle in map_df.itertuples(index=False):
        center = (float(obstacle.x), float(obstacle.y))
        obstacle_radius = float(obstacle.radius)
        safe_radius = obstacle_radius + robot_radius
        backup_radius = safe_radius + backup_margin

        # Unsafe robot-clearance region.
        ax.add_patch(
            Circle(
                center,
                safe_radius,
                facecolor="#e85d68",
                edgecolor="none",
                alpha=0.50,
                zorder=4,
            )
        )

        # Backup boundary.
        ax.add_patch(
            Circle(
                center,
                backup_radius,
                facecolor="none",
                edgecolor="#2855a6",
                linewidth=1.3,
                linestyle="-.",
                zorder=2,
            )
        )

        # Safe boundary.
        ax.add_patch(
            Circle(
                center,
                safe_radius,
                facecolor="none",
                edgecolor="#27864b",
                linewidth=1.5,
                zorder=3,
            )
        )

        # Physical obstacle.
        ax.add_patch(
            Circle(
                center,
                obstacle_radius,
                facecolor="#d3d3d3",
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
            )
        )

        if hasattr(obstacle, "obstacle_id"):
            label = str(int(obstacle.obstacle_id))
            ax.text(
                center[0], center[1], label,
                ha="center", va="center", fontsize=7, zorder=6,
            )


def build_xy_path(case_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = case_rows.sort_values("step")
    x = np.concatenate((
        np.array([rows["pre_x_m"].iloc[0]]),
        rows["post_x_m"].to_numpy(),
    ))
    y = np.concatenate((
        np.array([rows["pre_y_m"].iloc[0]]),
        rows["post_y_m"].to_numpy(),
    ))
    return x, y


def draw_trajectory_case(
    ax: plt.Axes,
    map_dir: Path,
    case_id: int,
) -> None:
    start_goal_drawn = False

    for method in METHODS:
        traj_csv = map_dir / method / f"{method}_trajectory.csv"
        if not traj_csv.is_file():
            continue

        traj = pd.read_csv(traj_csv)
        rows = traj[traj["case_id"].astype(int) == case_id].copy()
        if rows.empty:
            continue

        x, y = build_xy_path(rows)
        ax.plot(
            x, y,
            linewidth=2.4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=10,
        )

        if not start_goal_drawn:
            start_x = float(rows["start_x_m"].iloc[0])
            start_y = float(rows["start_y_m"].iloc[0])
            goal_x = float(rows["goal_x_m"].iloc[0])
            goal_y = float(rows["goal_y_m"].iloc[0])

            ax.scatter(
                start_x, start_y,
                marker="o", s=85,
                edgecolors="black", linewidths=1.0,
                color="#1f77b4",
                label="Start", zorder=12,
            )
            ax.scatter(
                goal_x, goal_y,
                marker="*", s=190,
                edgecolors="black", linewidths=1.0,
                color="#ffb000",
                label="Goal", zorder=12,
            )
            ax.plot(
                [start_x, goal_x],
                [start_y, goal_y],
                linestyle="--",
                linewidth=1.1,
                color="#5f9ed1",
                alpha=0.75,
                label="Straight-line reference",
                zorder=7,
            )
            start_goal_drawn = True


def configure_axis(ax: plt.Axes, *, arena_limit: float) -> None:
    ax.set_xlim(-arena_limit, arena_limit)
    ax.set_ylim(-arena_limit, arena_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(True, alpha=0.23, zorder=20)


def legend_handles(include_trajectories: bool) -> list:
    handles = [
        Patch(facecolor="#7aa6e8", alpha=0.20, label=r"Backup set $\mathcal{C}_B$"),
        Patch(facecolor="#39ff88", alpha=0.90, label=r"Safe set $\mathcal{C}_S$"),
        Patch(facecolor="#e85d68", alpha=0.65, label="Robot-clearance region"),
        Patch(facecolor="#d3d3d3", edgecolor="black", label="Physical obstacle"),
    ]
    if include_trajectories:
        handles += [
            Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.4, label=METHOD_LABELS[m])
            for m in METHODS
        ]
        handles += [
            Line2D([0], [0], marker="o", linestyle="none", markeredgecolor="black", label="Start"),
            Line2D([0], [0], marker="*", linestyle="none", markeredgecolor="black", markersize=12, label="Goal"),
            Line2D([0], [0], linestyle="--", color="#5f9ed1", label="Straight-line reference"),
        ]
    return handles


def plot_map(
    run_dir: Path,
    map_name: str,
    *,
    case_id: int | None,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    robot_radius: float,
    backup_margin: float,
    arena_limit: float,
    grid_points: int,
) -> None:
    map_dir = run_dir / map_name
    map_csv = map_dir / "map.csv"
    if not map_csv.is_file():
        raise FileNotFoundError(map_csv)

    map_df = pd.read_csv(map_csv)
    required = {"x", "y", "radius"}
    missing = required - set(map_df.columns)
    if missing:
        raise KeyError(f"{map_csv} is missing columns: {sorted(missing)}")

    xx, yy, safe_mask, backup_mask = compute_set_masks(
        map_df,
        arena_limit=arena_limit,
        robot_radius=robot_radius,
        backup_margin=backup_margin,
        grid_points=grid_points,
    )

    fig, ax = plt.subplots(figsize=(7.3, 7.0))
    draw_sets(
        ax,
        map_df,
        xx,
        yy,
        safe_mask,
        backup_mask,
        robot_radius=robot_radius,
        backup_margin=backup_margin,
    )

    include_trajectories = case_id is not None
    if case_id is not None:
        draw_trajectory_case(ax, map_dir, case_id)

    configure_axis(ax, arena_limit=arena_limit)
    title = f"{map_name.capitalize()} map"
    if case_id is not None:
        title += f" — case {case_id}"
    ax.set_title(title)

    ax.legend(
        handles=legend_handles(include_trajectories),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2 if not include_trajectories else 3,
        frameon=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"{map_name}_case_{case_id}_sets"
        if case_id is not None
        else f"{map_name}_layout_sets"
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    for map_name in args.maps:
        plot_map(
            run_dir,
            map_name,
            case_id=args.case_id,
            output_dir=output_dir,
            formats=args.formats,
            dpi=args.dpi,
            robot_radius=args.robot_radius,
            backup_margin=args.backup_margin,
            arena_limit=args.arena_limit,
            grid_points=args.grid_points,
        )


if __name__ == "__main__":
    main()
