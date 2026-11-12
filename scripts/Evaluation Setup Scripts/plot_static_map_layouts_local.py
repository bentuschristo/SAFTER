#!/usr/bin/env python3
"""Standalone paper plotter for training/unseen static map layouts.

This script has no dependency on the Unitree/MuJoCo repo and does not require
map.csv files. It hardcodes the original training map, the five structured
unseen maps, and the structured start/goal sampling boxes.

Default laser margins:
    d_safe = 0.15 m
    d_B    = 0.30 m
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


BACKUP_COLOR = "#7aa6e8"
SAFE_ONLY_COLOR = "#39ff88"
UNSAFE_COLOR = "#e85d68"
OBSTACLE_COLOR = "#d3d3d3"
BACKUP_EDGE = "#2855a6"
SAFE_EDGE = "#27864b"
START_REGION_COLOR = "#e100ff"   # initial/start region shade
GOAL_REGION_COLOR = "#ff9d00"      # goal region shade
REGION_ALPHA = 0.30                # larger=darker, smaller=lighter


TRAINING_MAP = (
    Obstacle(-3.000, -2.000, 0.550),
    Obstacle(-1.500, +2.200, 0.450),
    Obstacle(+1.200, -1.000, 0.600),
    Obstacle(+3.000, +2.300, 0.500),
    Obstacle(+0.300, +3.800, 0.400),
    Obstacle(-5.000, +4.500, 0.500),
    Obstacle(-5.000, -4.500, 0.450),
    Obstacle(+5.000, +4.500, 0.550),
    Obstacle(+5.000, -4.500, 0.500),
)


UNSEEN_MAPS = {
    "shifted": (
        Obstacle(-4.2, -1.0, 0.55),
        Obstacle(-2.0,  3.1, 0.45),
        Obstacle( 0.4, -2.8, 0.60),
        Obstacle( 2.8,  3.7, 0.50),
        Obstacle( 4.4,  0.2, 0.40),
        Obstacle(-5.4,  5.2, 0.50),
        Obstacle(-5.2, -4.8, 0.45),
        Obstacle( 5.3,  4.7, 0.55),
        Obstacle( 4.8, -5.1, 0.50),
    ),
    "corridor": (
        Obstacle(-3.8, -2.2, 0.50),
        Obstacle(-1.9, -2.2, 0.50),
        Obstacle( 0.0, -2.2, 0.50),
        Obstacle( 1.9, -2.2, 0.50),
        Obstacle( 3.8, -2.2, 0.50),
        Obstacle(-3.8,  2.2, 0.50),
        Obstacle(-1.9,  2.2, 0.50),
        Obstacle( 1.9,  2.2, 0.50),
        Obstacle( 3.8,  2.2, 0.50),
    ),
    "clustered": (
        Obstacle(-2.6,  1.7, 0.55),
        Obstacle(-1.3,  2.7, 0.45),
        Obstacle(-0.2,  1.5, 0.60),
        Obstacle(-2.1,  3.8, 0.50),
        Obstacle( 0.1,  3.6, 0.40),
        Obstacle( 3.8, -3.6, 0.50),
        Obstacle( 5.0, -2.4, 0.45),
        Obstacle( 3.1, -1.9, 0.55),
        Obstacle( 5.2, -4.8, 0.50),
    ),
    "slalom": (
        Obstacle(-5.2, -2.3, 0.50),
        Obstacle(-3.9,  2.1, 0.50),
        Obstacle(-2.6, -2.2, 0.50),
        Obstacle(-1.3,  2.2, 0.50),
        Obstacle( 0.0, -2.2, 0.50),
        Obstacle( 1.3,  2.2, 0.50),
        Obstacle( 2.6, -2.2, 0.50),
        Obstacle( 3.9,  2.1, 0.50),
        Obstacle( 5.2, -2.3, 0.50),
    ),
    "bottleneck": (
        Obstacle(-4.8,  4.5, 0.50),
        Obstacle(-4.8, -4.5, 0.50),
        Obstacle( 4.8,  4.5, 0.50),
        Obstacle( 4.8, -4.5, 0.50),
        Obstacle(-1.8,  1.15, 0.55),
        Obstacle( 0.0,  1.15, 0.55),
        Obstacle( 1.8,  1.15, 0.55),
        Obstacle(-0.9, -1.15, 0.55),
        Obstacle( 0.9, -1.15, 0.55),
    ),
}


START_GOAL_REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "bottleneck": {"start": (-7.5, -4.0, -2.0, 2.0), "goal": (4.0, 7.5, -2.0, 2.0)},
    "clustered": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (6.0, 7.5, -7.5, -6.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "shifted": {"start": (-7.5, -4.0, -6.0, 6.0), "goal": (4.0, 7.5, -6.0, 6.0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_static_layouts_laser_margin"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--d-safe", type=float, default=0.15)
    parser.add_argument("--backup-margin", type=float, default=0.30)
    parser.add_argument("--arena-limit", type=float, default=7.5)
    parser.add_argument("--grid-points", type=int, default=600)
    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--hide-labels", action="store_false", dest="show_labels")
    return parser.parse_args()


def compute_masks(obstacles, *, arena_limit, d_safe, backup_margin, grid_points):
    x_values = np.linspace(-arena_limit, arena_limit, grid_points)
    y_values = np.linspace(-arena_limit, arena_limit, grid_points)
    xx, yy = np.meshgrid(x_values, y_values)

    wall_safe_clearance = np.minimum.reduce([
        xx - (-arena_limit + d_safe),
        (arena_limit - d_safe) - xx,
        yy - (-arena_limit + d_safe),
        (arena_limit - d_safe) - yy,
    ])
    wall_backup_clearance = np.minimum.reduce([
        xx - (-arena_limit + d_safe + backup_margin),
        (arena_limit - d_safe - backup_margin) - xx,
        yy - (-arena_limit + d_safe + backup_margin),
        (arena_limit - d_safe - backup_margin) - yy,
    ])

    obstacle_safe_clearance = np.full_like(xx, np.inf, dtype=float)
    obstacle_backup_clearance = np.full_like(xx, np.inf, dtype=float)
    for obs in obstacles:
        dist = np.hypot(xx - obs.x, yy - obs.y)
        safe_radius = obs.radius + d_safe
        backup_radius = safe_radius + backup_margin
        obstacle_safe_clearance = np.minimum(obstacle_safe_clearance, dist - safe_radius)
        obstacle_backup_clearance = np.minimum(obstacle_backup_clearance, dist - backup_radius)

    safe_mask = (wall_safe_clearance >= 0.0) & (obstacle_safe_clearance >= 0.0)
    backup_mask = (wall_backup_clearance >= 0.0) & (obstacle_backup_clearance >= 0.0)
    return xx, yy, safe_mask, backup_mask


def draw_start_goal_regions(ax, map_name):
    if map_name not in START_GOAL_REGIONS:
        return
    for key, color in (("start", START_REGION_COLOR), ("goal", GOAL_REGION_COLOR)):
        x_min, x_max, y_min, y_max = START_GOAL_REGIONS[map_name][key]
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               facecolor=color, edgecolor="none", alpha=REGION_ALPHA, zorder=1.6))


def draw_sets(ax, obstacles, xx, yy, safe_mask, backup_mask, *, d_safe, backup_margin, arena_limit, show_labels):
    safe_only_mask = safe_mask & (~backup_mask)
    unsafe_mask = ~safe_mask

    ax.contourf(xx, yy, backup_mask.astype(float), levels=[0.5, 1.5], colors=[BACKUP_COLOR], alpha=0.16, zorder=0)
    ax.contourf(xx, yy, safe_only_mask.astype(float), levels=[0.5, 1.5], colors=[SAFE_ONLY_COLOR], alpha=0.90, zorder=1)
    ax.contourf(xx, yy, unsafe_mask.astype(float), levels=[0.5, 1.5], colors=[UNSAFE_COLOR], alpha=0.50, zorder=2)

    ax.add_patch(Rectangle((-arena_limit, -arena_limit), 2 * arena_limit, 2 * arena_limit,
                           fill=False, edgecolor="black", linewidth=1.5, zorder=8))
    ax.add_patch(Rectangle((-arena_limit + d_safe, -arena_limit + d_safe),
                           2 * (arena_limit - d_safe), 2 * (arena_limit - d_safe),
                           fill=False, edgecolor=SAFE_EDGE, linewidth=1.5, zorder=6))
    ax.add_patch(Rectangle((-arena_limit + d_safe + backup_margin, -arena_limit + d_safe + backup_margin),
                           2 * (arena_limit - d_safe - backup_margin),
                           2 * (arena_limit - d_safe - backup_margin),
                           fill=False, edgecolor=BACKUP_EDGE, linewidth=1.3, linestyle="-.", zorder=5))

    for i, obs in enumerate(obstacles):
        center = (obs.x, obs.y)
        safe_radius = obs.radius + d_safe
        backup_radius = safe_radius + backup_margin
        ax.add_patch(Circle(center, backup_radius, facecolor="none", edgecolor=BACKUP_EDGE,
                            linewidth=1.3, linestyle="-.", zorder=3))
        ax.add_patch(Circle(center, safe_radius, facecolor="none", edgecolor=SAFE_EDGE,
                            linewidth=1.5, zorder=4))
        ax.add_patch(Circle(center, obs.radius, facecolor=OBSTACLE_COLOR, edgecolor="black",
                            linewidth=1.2, zorder=7))
        if show_labels:
            ax.text(obs.x, obs.y, str(i), ha="center", va="center", fontsize=7, zorder=9)


def configure_axis(ax, *, arena_limit):
    ax.set_xlim(-arena_limit, arena_limit)
    ax.set_ylim(-arena_limit, arena_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (m)", labelpad=8)
    ax.set_ylabel("Y position (m)")
    ax.grid(True, alpha=0.23, zorder=20)


def legend_handles():
    return [
        Patch(facecolor=START_REGION_COLOR, alpha=REGION_ALPHA, label="Initial region"),
        Patch(facecolor=GOAL_REGION_COLOR, alpha=REGION_ALPHA, label="Goal region"),
        Patch(facecolor=BACKUP_COLOR, alpha=0.20, label=r"Backup set $\mathcal{C}_B$"),
        Patch(facecolor=SAFE_ONLY_COLOR, alpha=0.90, label=r"Safe set $\mathcal{C}_S$"),
        Patch(facecolor=UNSAFE_COLOR, alpha=0.65, label="Laser safety region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Physical obstacle"),
        Line2D([0], [0], color=SAFE_EDGE, linewidth=1.5, label=r"$\partial\mathcal{C}_S$"),
        Line2D([0], [0], color=BACKUP_EDGE, linewidth=1.3, linestyle="-.", label=r"$\partial\mathcal{C}_B$"),
    ]


def plot_one(name, obstacles, outdir, *, d_safe, backup_margin, arena_limit, grid_points, formats, dpi, show_labels):
    xx, yy, safe_mask, backup_mask = compute_masks(
        obstacles, arena_limit=arena_limit, d_safe=d_safe, backup_margin=backup_margin, grid_points=grid_points
    )
    if name == "original_training":
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
    else:
        fig, ax = plt.subplots(figsize=(4.2, 4.1))
    draw_sets(ax, obstacles, xx, yy, safe_mask, backup_mask,
              d_safe=d_safe, backup_margin=backup_margin, arena_limit=arena_limit, show_labels=show_labels)
    if name != "original_training":
        draw_start_goal_regions(ax, name)
    configure_axis(ax, arena_limit=arena_limit)

    if name == "original_training":
        title = "Original training map"
        stem_name = f"original_training_layout_sets_backup_{backup_margin:.2f}".replace(".", "p")
    else:
        title = f"{name.capitalize()} map"
        stem_name = f"{name}_layout_sets"

    ax.set_title(title)

    if name == "original_training":
        ax.legend(
            handles=legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            frameon=True,
            fontsize=8,
            handlelength=1.7,
            columnspacing=1.0,
            labelspacing=0.45,
            borderpad=0.45,
        )
        fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.30)
    else:
        fig.tight_layout(rect=(0, 0, 1, 1))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / stem_name
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}")


def save_legend_only(outdir, *, formats, dpi):
    fig, ax = plt.subplots(figsize=(8.2, 1.15))
    ax.axis("off")
    ax.legend(handles=legend_handles(), loc="center", ncol=4, frameon=True)
    stem = outdir / "map_layout_legend_only"
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}")


def main():
    args = parse_args()
    outdir = args.output_dir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    plot_one("original_training", TRAINING_MAP, outdir, d_safe=args.d_safe, backup_margin=args.backup_margin,
             arena_limit=args.arena_limit, grid_points=args.grid_points, formats=args.formats,
             dpi=args.dpi, show_labels=args.show_labels)
    for name in ["slalom", "bottleneck", "clustered", "corridor", "shifted"]:
        plot_one(name, UNSEEN_MAPS[name], outdir, d_safe=args.d_safe, backup_margin=args.backup_margin,
                 arena_limit=args.arena_limit, grid_points=args.grid_points, formats=args.formats,
                 dpi=args.dpi, show_labels=args.show_labels)
    # Legend is included in the original training map figure.
    # save_legend_only(outdir, formats=args.formats, dpi=args.dpi)


if __name__ == "__main__":
    main()
