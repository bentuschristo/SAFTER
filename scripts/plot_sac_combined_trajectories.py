#!/usr/bin/env python3
"""Combined trajectory figure: 1 row x 3 maps (Easy/Medium/Hard), each panel
overlaying SEA-Nav (purple), Vanilla (blue), CBF-CF (orange), bCBF-CF
(green) on the IDENTICAL start/goal scenario used in
plot_fail_fail_fail_succeed_trajectories.py.

SEA-Nav's trace here is from the wall-fix verification run (still the old,
41-ray checkpoint) -- unaffected by the later ray-count/FOV/max-range
config change (evaluated separately, not re-plotted here), so this data
remains valid to show.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
METHOD_COLORS = {
    "seanav": "#9467bd", "vanilla": "#1f77b4", "cbf_cd_cf": "#ff7f0e", "bcbf_cf": "#2ca02c",
}
METHOD_LABELS = {
    "seanav": "SEA-Nav", "vanilla": "Vanilla", "cbf_cd_cf": "CBF-CF", "bcbf_cf": "bCBF-CF (Ours)",
}
ARENA_MIN, ARENA_MAX = -7.5, 7.5

MAPS = {
    "slalom": (
        (-5.2, -2.3, 0.50), (-3.9, 2.1, 0.50), (-2.6, -2.2, 0.50),
        (-1.3, 2.2, 0.50), (0.0, -2.2, 0.50), (1.3, 2.2, 0.50),
        (2.6, -2.2, 0.50), (3.9, 2.1, 0.50), (5.2, -2.3, 0.50),
    ),
    "corridor": (
        (-3.8, -2.2, 0.50), (-1.9, -2.2, 0.50), (0.0, -2.2, 0.50),
        (1.9, -2.2, 0.50), (3.8, -2.2, 0.50), (-3.8, 2.2, 0.50),
        (-1.9, 2.2, 0.50), (1.9, 2.2, 0.50), (3.8, 2.2, 0.50),
    ),
    "clustered_v2": (
        (-3.0, 1.4, 0.55), (-1.3, 2.7, 0.45), (-0.08, 0.55, 0.60),
        (-2.4, 4.1, 0.50), (0.5, 3.9, 0.40), (1.98, -2.71, 0.50),
        (3.35, -1.44, 0.45), (-0.42, -1.55, 0.55), (3.47, -3.86, 0.50),
    ),
}
REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "clustered_v2": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (5.5, 7.5, -7.5, -5.5)},
}
MAP_ORDER = ("slalom", "corridor", "clustered_v2")
COLUMN_TITLES = ("Easy", "Medium", "Hard")
METHOD_ORDER = ("seanav", "vanilla", "cbf_cd_cf", "bcbf_cf")

LASER_TRAJ_ROOT = Path("logs/evaluation/trajectory_demo_laser")
LASER_CASE_IDS = {"slalom": 39, "corridor": 3, "clustered_v2": 0}
SEANAV_TRAJ_CSV = Path("logs/evaluation/seanav_native_eval/trajectory_demo/seanav_trajectory.csv")
SEANAV_ENV_ID = {"slalom": 3, "corridor": 10, "clustered_v2": 2}
SEANAV_DIFFICULTY = {"slalom": "easy", "corridor": "medium", "clustered_v2": "hard"}


def load_laser_trajectory(map_name: str, method: str) -> tuple[list[float], list[float], str]:
    case_id = LASER_CASE_IDS[map_name]
    path = LASER_TRAJ_ROOT / map_name / "seed_1000" / map_name / method / f"{method}_trajectory.csv"
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["case_id"]) == case_id]
    xs = [float(r["post_x_m"]) for r in rows]
    ys = [float(r["post_y_m"]) for r in rows]
    outcome = rows[-1]["outcome"] if rows else "unknown"
    return xs, ys, outcome


def load_seanav_trajectory(map_name: str) -> tuple[list[float], list[float], str]:
    """Drop the last recorded row -- it is always the post-reset teleport
    back to the scenario's start position (env.step() applies reset_idx()
    internally, in the same call, for any env whose termination fires that
    step), not real motion. See evaluate_single_scenario_trajectory.py."""
    difficulty = SEANAV_DIFFICULTY[map_name]
    env_id = SEANAV_ENV_ID[map_name]
    with SEANAV_TRAJ_CSV.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["difficulty"] == difficulty and int(r["env_id"]) == env_id
        ]
    rows.sort(key=lambda r: int(r["step"]))
    rows = rows[:-1]
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    outcome = rows[-1]["outcome"] if rows else "unknown"
    return xs, ys, outcome


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.28, zorder=1,
    ))


def draw_panel(ax, map_name: str, title: str) -> None:
    add_region(ax, REGIONS[map_name]["start"], START_COLOR)
    add_region(ax, REGIONS[map_name]["goal"], GOAL_COLOR)
    for x, y, radius in MAPS[map_name]:
        ax.add_patch(Circle(
            (x, y), radius, facecolor=OBSTACLE_COLOR, edgecolor="black",
            linewidth=0.8, zorder=2, alpha=0.9,
        ))
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=1.6, zorder=4,
    ))

    for method in METHOD_ORDER:
        if method == "seanav":
            xs, ys, outcome = load_seanav_trajectory(map_name)
        else:
            xs, ys, outcome = load_laser_trajectory(map_name, method)
        color = METHOD_COLORS[method]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=5, solid_capstyle="round",
                 label=METHOD_LABELS[method])
        end_x, end_y = xs[-1], ys[-1]
        if outcome == "success":
            ax.plot(end_x, end_y, marker="*", markersize=16, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.8, zorder=6)
        else:
            ax.plot(end_x, end_y, marker="x", markersize=11, markeredgewidth=3.0,
                     color=color, zorder=6)

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=9, length=3)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_title(title, fontsize=13, pad=8)


def main() -> None:
    fig, axes = plt.subplots(1, len(MAP_ORDER), figsize=(4.0 * len(MAP_ORDER), 4.4))
    for idx, (ax, map_name) in enumerate(zip(axes, MAP_ORDER)):
        draw_panel(ax, map_name, COLUMN_TITLES[idx])
        if idx == 0:
            ax.set_ylabel("y [m]", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ] + [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.0, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ] + [
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10,
                   markeredgewidth=3.0, label="Collision / timeout"),
        plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=14,
                   label="Goal reached"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.12), fontsize=10, columnspacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.24, wspace=0.08)

    output_base = Path("logs/evaluation/figures/combined_trajectories")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
