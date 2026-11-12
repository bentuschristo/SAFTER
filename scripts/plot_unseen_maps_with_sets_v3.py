#!/usr/bin/env python3
r"""Plot unseen-map layouts and optional trajectories with safe/backup-set shading.

Version 3 adds explicit transparent obstacle/wall bands on top of the
set masks, making robot-clearance, safe-only, and backup boundaries visible
in presentation figures.

Shading convention:
  - Blue: backup set C_B
  - Green: safe-only band C_S \ C_B
  - Red: robot-clearance / unsafe region
  - Gray: physical obstacle

Examples
--------
All five layouts:
    python scripts/plot_unseen_maps_with_sets_v3.py \
      --run-dir logs/evaluation/unseen_maps/20260703_022535_seed42_20permap \
      --output-dir logs/evaluation/figures_v4/maps_with_sets_v3

One trajectory case:
    python scripts/plot_unseen_maps_with_sets_v3.py \
      --run-dir logs/evaluation/unseen_maps/20260703_105125_seed42_20permap \
      --maps corridor \
      --case-id 3 \
      --output-dir logs/evaluation/figures_v4/trajectory_sets_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle, Wedge
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

# Same palette as the v1 unseen-map set plot and the original-training-map plot.
BACKUP_COLOR = "#7aa6e8"       # blue
SAFE_ONLY_COLOR = "#39ff88"    # green
UNSAFE_COLOR = "#e85d68"       # red
OBSTACLE_COLOR = "#d3d3d3"     # gray
BACKUP_EDGE = "#2855a6"
SAFE_EDGE = "#27864b"


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
    p.add_argument("--no-explicit-bands", action="store_true", help="Disable explicit annulus/wall bands and use only contour masks.")
    return p.parse_args()


def compute_set_masks(
    map_df: pd.DataFrame,
    *,
    arena_limit: float,
    robot_radius: float,
    backup_margin: float,
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return grid and masks for C_S and C_B.

    The v1 script only considered obstacle clearances. This v2 version also
    treats the square arena boundary as a wall. Therefore:
      C_S requires clearance >= robot_radius from walls and obstacles.
      C_B requires clearance >= robot_radius + backup_margin from walls and obstacles.
    """
    x_values = np.linspace(-arena_limit, arena_limit, grid_points)
    y_values = np.linspace(-arena_limit, arena_limit, grid_points)
    xx, yy = np.meshgrid(x_values, y_values)

    # Wall clearances. Positive means the robot center is inside the set.
    wall_safe_clearance = np.minimum.reduce(
        [
            xx - (-arena_limit + robot_radius),
            (arena_limit - robot_radius) - xx,
            yy - (-arena_limit + robot_radius),
            (arena_limit - robot_radius) - yy,
        ]
    )
    wall_backup_clearance = np.minimum.reduce(
        [
            xx - (-arena_limit + robot_radius + backup_margin),
            (arena_limit - robot_radius - backup_margin) - xx,
            yy - (-arena_limit + robot_radius + backup_margin),
            (arena_limit - robot_radius - backup_margin) - yy,
        ]
    )

    obstacle_safe_clearance = np.full_like(xx, np.inf, dtype=float)
    obstacle_backup_clearance = np.full_like(xx, np.inf, dtype=float)

    for obstacle in map_df.itertuples(index=False):
        distance = np.hypot(
            xx - float(obstacle.x),
            yy - float(obstacle.y),
        )
        safe_radius = float(obstacle.radius) + robot_radius
        backup_radius = safe_radius + backup_margin

        obstacle_safe_clearance = np.minimum(obstacle_safe_clearance, distance - safe_radius)
        obstacle_backup_clearance = np.minimum(obstacle_backup_clearance, distance - backup_radius)

    safe_mask = (wall_safe_clearance >= 0.0) & (obstacle_safe_clearance >= 0.0)
    backup_mask = (wall_backup_clearance >= 0.0) & (obstacle_backup_clearance >= 0.0)
    return xx, yy, safe_mask, backup_mask


def _add_wall_bands(
    ax: plt.Axes,
    *,
    arena_limit: float,
    robot_radius: float,
    backup_margin: float,
) -> None:
    """Draw explicit wall bands so boundary margins are visible in slides."""
    L = float(arena_limit)
    r = float(robot_radius)
    m = float(backup_margin)

    # Red robot-clearance band: outside C_S near the true wall.
    red_rects = [
        (-L, -L, 2 * L, r),          # bottom
        (-L, L - r, 2 * L, r),       # top
        (-L, -L, r, 2 * L),          # left
        (L - r, -L, r, 2 * L),       # right
    ]
    for x, y, w, h in red_rects:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=UNSAFE_COLOR, edgecolor="none", alpha=0.45, zorder=3)
        )

    # Green safe-only band: C_S \ C_B near the wall.
    if m > 0.0:
        g0 = L - r - m
        g1 = L - r
        green_rects = [
            (-g1, -g1, 2 * g1, m),       # bottom
            (-g1, g0, 2 * g1, m),        # top
            (-g1, -g1, m, 2 * g1),       # left
            (g0, -g1, m, 2 * g1),        # right
        ]
        for x, y, w, h in green_rects:
            ax.add_patch(
                Rectangle((x, y), w, h, facecolor=SAFE_ONLY_COLOR, edgecolor="none", alpha=0.55, zorder=3.2)
            )


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
    explicit_bands: bool = True,
) -> None:
    safe_only_mask = safe_mask & (~backup_mask)
    unsafe_mask = ~safe_mask
    arena_limit = float(max(np.max(np.abs(xx)), np.max(np.abs(yy))))

    # Backup set: free space satisfying the more conservative margin.
    ax.contourf(
        xx, yy, backup_mask.astype(float),
        levels=[0.5, 1.5],
        colors=[BACKUP_COLOR],
        alpha=0.14,
        zorder=0,
    )

    # Coarse masks remain useful for exact obstacle/wall unions.
    ax.contourf(
        xx, yy, safe_only_mask.astype(float),
        levels=[0.5, 1.5],
        colors=[SAFE_ONLY_COLOR],
        alpha=0.35,
        zorder=1,
    )
    ax.contourf(
        xx, yy, unsafe_mask.astype(float),
        levels=[0.5, 1.5],
        colors=[UNSAFE_COLOR],
        alpha=0.30,
        zorder=2,
    )

    if explicit_bands:
        _add_wall_bands(
            ax,
            arena_limit=arena_limit,
            robot_radius=robot_radius,
            backup_margin=backup_margin,
        )

    # True arena boundary.
    ax.add_patch(
        Rectangle(
            (-arena_limit, -arena_limit),
            2.0 * arena_limit,
            2.0 * arena_limit,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            zorder=8,
        )
    )

    # Wall safe and backup boundaries.
    ax.add_patch(
        Rectangle(
            (-arena_limit + robot_radius, -arena_limit + robot_radius),
            2.0 * (arena_limit - robot_radius),
            2.0 * (arena_limit - robot_radius),
            fill=False,
            edgecolor=SAFE_EDGE,
            linewidth=1.7,
            zorder=6,
        )
    )
    ax.add_patch(
        Rectangle(
            (
                -arena_limit + robot_radius + backup_margin,
                -arena_limit + robot_radius + backup_margin,
            ),
            2.0 * (arena_limit - robot_radius - backup_margin),
            2.0 * (arena_limit - robot_radius - backup_margin),
            fill=False,
            edgecolor=BACKUP_EDGE,
            linewidth=1.4,
            linestyle="-.",
            zorder=5,
        )
    )

    for obstacle in map_df.itertuples(index=False):
        center = (float(obstacle.x), float(obstacle.y))
        obstacle_radius = float(obstacle.radius)
        safe_radius = obstacle_radius + robot_radius
        backup_radius = safe_radius + backup_margin

        if explicit_bands:
            # Red robot-clearance disk: outside C_S.
            ax.add_patch(
                Circle(
                    center,
                    safe_radius,
                    facecolor=UNSAFE_COLOR,
                    edgecolor="none",
                    alpha=0.45,
                    zorder=3,
                )
            )
            # Green annulus: C_S \ C_B.
            if backup_margin > 0.0:
                ax.add_patch(
                    Wedge(
                        center,
                        backup_radius,
                        0.0,
                        360.0,
                        width=backup_margin,
                        facecolor=SAFE_ONLY_COLOR,
                        edgecolor="none",
                        alpha=0.55,
                        zorder=3.2,
                    )
                )

        # Backup boundary.
        ax.add_patch(
            Circle(
                center,
                backup_radius,
                facecolor="none",
                edgecolor=BACKUP_EDGE,
                linewidth=1.4,
                linestyle="-.",
                zorder=5,
            )
        )

        # Safe boundary.
        ax.add_patch(
            Circle(
                center,
                safe_radius,
                facecolor="none",
                edgecolor=SAFE_EDGE,
                linewidth=1.7,
                zorder=6,
            )
        )

        # Physical obstacle.
        ax.add_patch(
            Circle(
                center,
                obstacle_radius,
                facecolor=OBSTACLE_COLOR,
                edgecolor="black",
                linewidth=1.2,
                zorder=7,
            )
        )

        if hasattr(obstacle, "obstacle_id"):
            label = str(int(obstacle.obstacle_id))
            ax.text(
                center[0], center[1], label,
                ha="center", va="center", fontsize=7, zorder=9,
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
        Patch(facecolor=BACKUP_COLOR, alpha=0.20, label=r"Backup set $\mathcal{C}_B$"),
        Patch(facecolor=SAFE_ONLY_COLOR, alpha=0.90, label=r"Safe set $\mathcal{C}_S$"),
        Patch(facecolor=UNSAFE_COLOR, alpha=0.65, label="Robot-clearance region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Physical obstacle"),
        Line2D([0], [0], color=SAFE_EDGE, linewidth=1.5, label=r"$\partial\mathcal{C}_S$"),
        Line2D([0], [0], color=BACKUP_EDGE, linewidth=1.3, linestyle="-.", label=r"$\partial\mathcal{C}_B$"),
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
        f"{map_name}_case_{case_id}_sets_v3"
        if case_id is not None
        else f"{map_name}_layout_sets_v3"
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

    if args.robot_radius < 0.0:
        raise ValueError("--robot-radius must be nonnegative.")
    if args.backup_margin < 0.0:
        raise ValueError("--backup-margin must be nonnegative.")
    if args.arena_limit <= args.robot_radius + args.backup_margin:
        raise ValueError("--arena-limit must be larger than robot_radius + backup_margin.")

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
