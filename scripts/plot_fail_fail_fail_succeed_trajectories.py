#!/usr/bin/env python3
"""4 (methods) x 3 (maps) trajectory grid: SEA-Nav, Vanilla, CBF-CF all fail
(collision/timeout/other) and bCBF-CF succeeds, on the IDENTICAL start/goal
scenario per map (slalom=Easy, corridor=Medium, clustered_v2[latest]=Hard).

Trajectory line: red. Collision/timeout end: black X. Goal reached: gold
star. Obstacles/start-goal regions drawn the same way as the
easy/medium/hard preview scripts.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
TRAJ_COLOR = "#d62728"
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
METHOD_LABELS = {
    "seanav": "SEA-Nav", "vanilla": "Vanilla",
    "cbf_cd_cf": "CBF-CF", "bcbf_cf": "bCBF-CF (Ours)",
}

LASER_TRAJ_ROOT = Path("logs/evaluation/trajectory_demo_laser")
LASER_CASE_IDS = {"slalom": 39, "corridor": 3, "clustered_v2": 0}
SEANAV_TRAJ_CSV = Path("logs/evaluation/seanav_native_eval/trajectory_demo/seanav_trajectory.csv")
SEANAV_ENV_ID = {"slalom": 3, "corridor": 10, "clustered_v2": 2}
SEANAV_DIFFICULTY = {"slalom": "easy", "corridor": "medium", "clustered_v2": "hard"}

FAIL_OUTCOMES = {"collision", "timeout", "other_failure", "fallen", "outside"}


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
    """NOTE: the recording script (as it stood when this CSV was generated)
    read env.root_states AFTER env.step(), which already applies
    reset_idx() internally for any env whose termination fired that same
    step -- so the LAST recorded row for a finished env is always the
    post-reset teleport back to the scenario's start position, not its
    true collision/timeout point. Drop it here. (Fixed at the source in
    evaluate_single_scenario_trajectory.py for any future re-recording.)
    """
    difficulty = SEANAV_DIFFICULTY[map_name]
    env_id = SEANAV_ENV_ID[map_name]
    with SEANAV_TRAJ_CSV.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["difficulty"] == difficulty and int(r["env_id"]) == env_id
        ]
    rows.sort(key=lambda r: int(r["step"]))
    rows = rows[:-1]  # drop the corrupted post-reset final frame
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


def draw_panel(ax, map_name: str, xs: list[float], ys: list[float], outcome: str,
                title: str | None, show_ylabel: bool, row_label: str | None) -> None:
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

    if xs:
        ax.plot(xs, ys, color=TRAJ_COLOR, linewidth=1.8, zorder=5, solid_capstyle="round")
        end_x, end_y = xs[-1], ys[-1]
        if outcome == "success":
            ax.plot(end_x, end_y, marker="*", markersize=18, markerfacecolor="#ffd60a",
                     markeredgecolor="black", markeredgewidth=0.8, zorder=6)
        else:
            ax.plot(end_x, end_y, marker="x", markersize=12, markeredgewidth=3.0,
                     color="black", zorder=6)

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=8, length=3)
    ax.grid(False)
    if title is not None:
        ax.set_title(title, fontsize=13, pad=6)
    if show_ylabel:
        ax.set_ylabel(row_label, fontsize=12, fontweight="bold")
    else:
        ax.tick_params(labelleft=False)


def main() -> None:
    trajectories: dict[str, dict[str, tuple[list[float], list[float], str]]] = {}
    for map_name in MAP_ORDER:
        trajectories[map_name] = {}
        trajectories[map_name]["seanav"] = load_seanav_trajectory(map_name)
        for method in ("vanilla", "cbf_cd_cf", "bcbf_cf"):
            trajectories[map_name][method] = load_laser_trajectory(map_name, method)

    for map_name in MAP_ORDER:
        print(f"{map_name}:")
        for method in METHOD_ORDER:
            xs, ys, outcome = trajectories[map_name][method]
            print(f"  {method:10s} outcome={outcome:10s} n_steps={len(xs)}")

    n_rows, n_cols = len(METHOD_ORDER), len(MAP_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows))

    for row_idx, method in enumerate(METHOD_ORDER):
        for col_idx, map_name in enumerate(MAP_ORDER):
            ax = axes[row_idx, col_idx]
            xs, ys, outcome = trajectories[map_name][method]
            draw_panel(
                ax, map_name, xs, ys, outcome,
                title=COLUMN_TITLES[col_idx] if row_idx == 0 else None,
                show_ylabel=(col_idx == 0),
                row_label=METHOD_LABELS[method],
            )

    fig.legend(
        handles=(
            Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
            Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
            Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
            plt.Line2D([0], [0], color=TRAJ_COLOR, linewidth=1.8, label="Trajectory"),
            plt.Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10,
                       markeredgewidth=3.0, label="Collision / timeout"),
            plt.Line2D([0], [0], marker="*", color="#ffd60a", linestyle="None", markersize=14,
                       markeredgecolor="black", label="Goal reached"),
        ),
        loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.08, wspace=0.08, hspace=0.18)

    output_base = Path("logs/evaluation/figures/fail_fail_fail_succeed_trajectories")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
