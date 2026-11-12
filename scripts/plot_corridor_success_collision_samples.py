#!/usr/bin/env python3
"""Success and collision sample trajectories for SEA-Nav and bcbf_cf on the
corridor (Medium-Shift) map -- baseline comparison for the corridor-derived
High-Shift design work: corridor's own top row already omits the center
obstacle (4 circles, not 5 -- a wide-open gap baked in from the original
design), so this checks whether SEA-Nav uses it here either, and where
each method's real failures concentrate on the map that everything else
was benchmarked against.
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
ROBOT_COLLISION_RADIUS_M = 0.30
FOOTPRINT_FILL_ALPHA = 0.45
FOOTPRINT_EDGE_WIDTH = 1.6
N_SAMPLES = 6
SEED = 0

# corridor (evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS["corridor"]) --
# bottom row has 5 obstacles, top row only 4 (center omitted since the
# original design).
OBSTACLES = [
    (-3.8, -2.2, 0.50, "circle"), (-1.9, -2.2, 0.50, "circle"), (0.0, -2.2, 0.50, "circle"),
    (1.9, -2.2, 0.50, "circle"), (3.8, -2.2, 0.50, "circle"),
    (-3.8, 2.2, 0.50, "circle"), (-1.9, 2.2, 0.50, "circle"),
    (1.9, 2.2, 0.50, "circle"), (3.8, 2.2, 0.50, "circle"),
]
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

SEANAV_ROOT = Path(
    "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/"
    "2048env_1000iter_vxm1_saccorrected_seed0_25seed_trajectories"
)
BCBF_CF_TRAJ_RUNROOT_FILE = Path("/tmp/bcbf_cf_corridor_traj_runroot.txt")

FS_TITLE = 20
FS_SUPTITLE = 24
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14


def load_seanav(n: int, seed: int, outcome: str) -> list[tuple[list[float], list[float], float, float]]:
    rng = random.Random(seed)
    seed_dirs = sorted(SEANAV_ROOT.glob("seed_*"))
    results = []
    for seed_dir in seed_dirs:
        ep_path = seed_dir / "episodes.csv"
        traj_path = seed_dir / "trajectory.csv"
        if not ep_path.is_file() or not traj_path.is_file():
            continue
        with ep_path.open() as f:
            episodes = [r for r in csv.DictReader(f) if r["difficulty"] == "medium" and r["outcome"] == outcome]
        with traj_path.open() as f:
            traj = [r for r in csv.DictReader(f) if r["difficulty"] == "medium"]
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
                results.append((xs, ys, gx, gy))
        if len(results) >= n * 5:
            break
    rng.shuffle(results)
    return results[:n]


def load_bcbf_cf(n: int, seed: int, outcome: str) -> list[tuple[list[float], list[float], float, float]]:
    run_root = BCBF_CF_TRAJ_RUNROOT_FILE.read_text().strip()
    traj_path = Path(run_root) / "_pooled_batches" / "corridor" / "bcbf_cf" / "bcbf_cf_trajectory.csv"
    rng = random.Random(seed)
    by_case = defaultdict(list)
    with traj_path.open() as f:
        for r in csv.DictReader(f):
            by_case[(r["evaluation_seed"], r["case_id"])].append(r)
    results = []
    for key, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        if rows and rows[-1]["outcome"] == outcome:
            xs = [float(r["post_x_m"]) for r in rows]
            ys = [float(r["post_y_m"]) for r in rows]
            gx, gy = float(rows[0]["goal_x_m"]), float(rows[0]["goal_y_m"])
            results.append((xs, ys, gx, gy))
    rng.shuffle(results)
    return results[:n]


def draw_panel(ax, title: str, trajectories: list[tuple[list[float], list[float], float, float]], outcome: str) -> None:
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
        ax.plot([xs[0], gx], [ys[0], gy], color=color, linewidth=1.0, linestyle="--", zorder=3, alpha=0.55)
        ax.plot(xs, ys, color=color, linewidth=1.6, zorder=5, alpha=0.85)
        ax.scatter([xs[0]], [ys[0]], marker="o", s=32, color=color, zorder=6, edgecolor="black", linewidth=0.5)
        ax.scatter([gx], [gy], marker="*", s=110, color=color, zorder=6, edgecolor="black", linewidth=0.6)
        end_marker = "X" if outcome == "collision" else "*"
        ax.add_patch(Circle(
            (xs[-1], ys[-1]), ROBOT_COLLISION_RADIUS_M,
            facecolor=to_rgba(color, FOOTPRINT_FILL_ALPHA), edgecolor=color,
            linewidth=FOOTPRINT_EDGE_WIDTH, zorder=6,
        ))
        if outcome == "collision":
            ax.scatter([xs[-1]], [ys[-1]], marker="X", s=75, color=color, zorder=7, edgecolor="black", linewidth=0.7)

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=FS_TICK, length=4)
    ax.grid(False)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)


def main() -> None:
    seanav_success = load_seanav(N_SAMPLES, SEED, "success")
    seanav_collision = load_seanav(N_SAMPLES, SEED, "collision")
    bcbf_success = load_bcbf_cf(N_SAMPLES, SEED, "success")
    bcbf_collision = load_bcbf_cf(N_SAMPLES, SEED, "collision")
    print(f"SEA-Nav: {len(seanav_success)} success, {len(seanav_collision)} collision samples")
    print(f"bcbf_cf: {len(bcbf_success)} success, {len(bcbf_collision)} collision samples")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 14.0))
    draw_panel(axes[0, 0], "SEA-Nav -- success", seanav_success, "success")
    draw_panel(axes[0, 1], "SEA-Nav -- collision", seanav_collision, "collision")
    draw_panel(axes[1, 0], "bcbf_cf -- success", bcbf_success, "success")
    draw_panel(axes[1, 1], "bcbf_cf -- collision", bcbf_collision, "collision")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]", fontsize=FS_LABEL)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for ax in axes[1, :]:
        ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        plt.Line2D([0], [0], marker="o", color="gray", lw=0, label="Start", markersize=7, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="*", color="gray", lw=0, label="Goal", markersize=12, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="X", color="gray", lw=0, label="Collision point", markersize=9, markeredgecolor="black"),
        plt.Line2D([0], [0], color="gray", lw=1.0, linestyle="--", label="Direct start-goal line"),
        plt.Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
                    markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint (0.30m)", markersize=13),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.4, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Corridor (Medium-Shift) -- success / collision samples", fontsize=FS_SUPTITLE, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10, wspace=0.06, hspace=0.12)

    output_base = Path("logs/evaluation/figures/corridor_success_collision_samples")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
