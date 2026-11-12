#!/usr/bin/env python3
"""Collision-sample trajectories for SEA-Nav and bcbf_cf on the v2
corridor-derived High-Shift map, overlaid on the obstacle layout -- for
visually inspecting the failure mode each method actually exhibits (e.g.
SEA-Nav's total collapse clustering near the remaining top-row circle).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5
# Both methods use the same 0.30m collision-circle radius (SAC-laser's
# --collision-circle-radius-m 0.30, SEA-Nav's SEANAV_COLLISION_CIRCLE_RADIUS_M=0.30).
ROBOT_COLLISION_RADIUS_M = 0.30
FOOTPRINT_FILL_ALPHA = 0.45
FOOTPRINT_EDGE_WIDTH = 1.6
N_SAMPLES = 8
SEED = 0

# v2 layout (evaluate_laser_unseen_maps_structured.py's
# UNSEEN_MAPS["corridor_derived_highshift_v2"]).
OBSTACLES = [
    (-4.00, 2.2, 0.65, "circle"), (-2.00, 2.2, 0.65, "square"),
    (2.00, 2.2, 0.65, "square"), (4.00, 2.2, 0.65, "circle"),
    (-4.50, -2.2, 0.50, "circle"), (-2.45, -2.2, 0.65, "circle"),
    (0.00, -2.2, 0.90, "square"), (2.45, -2.2, 0.65, "circle"),
    (4.50, -2.2, 0.50, "circle"),
]
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

SEANAV_ROOT = Path(
    "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/"
    "2048env_1000iter_vxm1_saccorrected_corridor_derived_highshift_v2rev2_seed0_25seed_trajectories"
)
BCBF_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/20260924_103529_corridor_derived_highshift_v2rev2_bcbf_cf_seeds1000-1024_20perseed_500cases_90s_local/_pooled_batches/corridor_derived_highshift_v2/bcbf_cf/bcbf_cf_trajectory.csv"
)

FS_TITLE = 22
FS_LABEL = 20
FS_TICK = 16
FS_LEGEND = 16


def load_seanav_collisions(n: int, seed: int) -> list[tuple[list[float], list[float], float, float]]:
    rng = random.Random(seed)
    seed_dirs = sorted(SEANAV_ROOT.glob("seed_*"))
    collisions = []
    for seed_dir in seed_dirs:
        ep_path = seed_dir / "episodes.csv"
        traj_path = seed_dir / "trajectory.csv"
        if not ep_path.is_file() or not traj_path.is_file():
            continue
        with ep_path.open() as f:
            episodes = [r for r in csv.DictReader(f) if r["difficulty"] == "hard" and r["outcome"] == "collision"]
        with traj_path.open() as f:
            traj = [r for r in csv.DictReader(f) if r["difficulty"] == "hard"]
        by_ep = defaultdict(list)
        for r in traj:
            by_ep[(r["env_id"], r["episode_idx"])].append(r)
        for e in episodes:
            key = (e["env_id"], e["episode_idx"])
            rows = sorted(by_ep.get(key, []), key=lambda r: int(r["step"]))
            if rows:
                xs = [float(r["x"]) for r in rows]
                ys = [float(r["y"]) for r in rows]
                gx, gy = float(e["goal_x"]), float(e["goal_y"])
                collisions.append((xs, ys, gx, gy))
        if len(collisions) >= n * 5:
            break
    rng.shuffle(collisions)
    return collisions[:n]


def load_bcbf_cf_collisions(n: int, seed: int) -> list[tuple[list[float], list[float], float, float]]:
    rng = random.Random(seed)
    by_case = defaultdict(list)
    with BCBF_CF_TRAJ.open() as f:
        for r in csv.DictReader(f):
            by_case[(r["evaluation_seed"], r["case_id"])].append(r)
    collisions = []
    for key, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        if rows and rows[-1]["outcome"] == "collision":
            xs = [float(r["post_x_m"]) for r in rows]
            ys = [float(r["post_y_m"]) for r in rows]
            gx, gy = float(rows[0]["goal_x_m"]), float(rows[0]["goal_y_m"])
            collisions.append((xs, ys, gx, gy))
    rng.shuffle(collisions)
    return collisions[:n]


def draw_panel(ax, title: str, trajectories: list[tuple[list[float], list[float], float, float]]) -> None:
    x_min, x_max, y_min, y_max = START_REGION
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=START_COLOR, edgecolor=START_COLOR, linewidth=1.0, alpha=0.28, zorder=1,
    ))
    x_min, x_max, y_min, y_max = GOAL_REGION
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, linewidth=1.0, alpha=0.28, zorder=1,
    ))
    for x, y, radius, shape in OBSTACLES:
        if shape == "square":
            ax.add_patch(Rectangle(
                (x - radius, y - radius), 2 * radius, 2 * radius,
                facecolor=OBSTACLE_COLOR, edgecolor="black", linewidth=0.8, zorder=2, alpha=0.9,
            ))
        else:
            ax.add_patch(Circle(
                (x, y), radius, facecolor=OBSTACLE_COLOR, edgecolor="black",
                linewidth=0.8, zorder=2, alpha=0.9,
            ))
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=1.6, zorder=4,
    ))

    cmap = plt.get_cmap("tab10")
    for i, (xs, ys, gx, gy) in enumerate(trajectories):
        color = cmap(i % 10)
        ax.plot([xs[0], gx], [ys[0], gy], color=color, linewidth=1.2, linestyle="--", zorder=3, alpha=0.6)
        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=5, alpha=0.85)
        ax.scatter([xs[0]], [ys[0]], marker="o", s=40, color=color, zorder=6, edgecolor="black", linewidth=0.6)
        ax.scatter([gx], [gy], marker="*", s=140, color=color, zorder=6, edgecolor="black", linewidth=0.7)
        ax.add_patch(Circle(
            (xs[-1], ys[-1]), ROBOT_COLLISION_RADIUS_M,
            facecolor=to_rgba(color, FOOTPRINT_FILL_ALPHA), edgecolor=color,
            linewidth=FOOTPRINT_EDGE_WIDTH, zorder=6,
        ))
        ax.scatter([xs[-1]], [ys[-1]], marker="X", s=90, color=color, zorder=7, edgecolor="black", linewidth=0.8)

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=FS_TICK, length=5)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=12)


def main() -> None:
    seanav_traj = load_seanav_collisions(N_SAMPLES, SEED)
    bcbf_traj = load_bcbf_cf_collisions(N_SAMPLES, SEED)
    print(f"SEA-Nav collision samples: {len(seanav_traj)}")
    print(f"bcbf_cf collision samples: {len(bcbf_traj)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    draw_panel(axes[0], "SEA-Nav collisions", seanav_traj)
    draw_panel(axes[1], "bcbf_cf collisions", bcbf_traj)
    axes[0].set_ylabel("y [m]", fontsize=FS_LABEL)
    axes[1].tick_params(labelleft=False)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        plt.Line2D([0], [0], marker="o", color="gray", lw=0, label="Start", markersize=8, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="*", color="gray", lw=0, label="Goal", markersize=13, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="X", color="gray", lw=0, label="Collision point", markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="Direct start-goal line"),
        plt.Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
                    markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint (0.30m)", markersize=14),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.5, bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle("Corridor-derived High-Shift v2 -- collision samples", fontsize=24, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.26, wspace=0.08)

    output_base = Path("logs/evaluation/figures/corridor_derived_highshift_v3_collision_samples")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
