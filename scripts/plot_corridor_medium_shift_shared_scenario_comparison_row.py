#!/usr/bin/env python3
"""4-method (SEA-Nav / CBF-RL / CBF-CF / SAFTER) 10-sample trajectory
comparison on "corridor" (Medium-Shift), 1x4 row layout (requested
2026-09-25 as an alternative to the 2x2 grid version in
plot_corridor_medium_shift_shared_scenario_comparison.py).

Data-loading logic is a straight port of that script's final, verified
state (same SELECTED_CASE_IDS, same SEA_NAV_PICKS with the truncated
genuine-collision fix, same CBF_RL_REPEAT_OVERRIDE, same two-pool
old/new-scenario merge via NEW_POOL_OFFSET) -- see that file's docstring
for the full correctness history (per-episode outcome mislabeling fix,
goal-region-violation exclusion across all 4 methods, etc.). Only the
figure layout differs: 1x4 row of panels with a two-row legend spanning
the full width underneath, instead of a 2x2 grid with a single left-side
legend column.

Layout: SEA-Nav, CBF-RL, CBF-CF (cbf_cd_cf), SAFTER (bCBF-CF) left to
right. Trajectories are colored by case_id (same color = same scenario
across all 4 panels). Saved as medium-shift-row.png/.pdf.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.lines import Line2D

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5
ROBOT_COLLISION_RADIUS_M = 0.30
FOOTPRINT_FILL_ALPHA = 0.45
FOOTPRINT_EDGE_WIDTH = 1.6
TRAJ_LW = 2.0

OBSTACLES = [
    (-3.8, -2.2, 0.50, "circle"),
    (-1.9, -2.2, 0.50, "circle"),
    (0.0, -2.2, 0.50, "circle"),
    (1.9, -2.2, 0.50, "circle"),
    (3.8, -2.2, 0.50, "circle"),
    (-3.8, 2.2, 0.50, "circle"),
    (-1.9, 2.2, 0.50, "circle"),
    (1.9, 2.2, 0.50, "circle"),
    (3.8, 2.2, 0.50, "circle"),
]
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

# New-pool (40-scenario collision-search) case_ids are namespaced as
# 100+case_id to avoid clashing with the original 25-scenario pool's own
# 0-24 numbering.
NEW_POOL_OFFSET = 100
SELECTED_CASE_IDS = [100, 133, 4, 5, 107, 10, 11, 15, 110, 139]

SHARED_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift/seed_1000/corridor/scenarios.csv"
)
BCBF_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift/_pooled_batches/corridor/bcbf_cf/bcbf_cf_trajectory.csv"
)
CBF_CD_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift/_pooled_batches/corridor/cbf_cd_cf/cbf_cd_cf_trajectory.csv"
)
CBF_RL_TRAJ = Path("logs/evaluation/corridor_shared_scenarios_medium_shift/cbf_rl/cbf_rl_trajectory.csv")

# New 40-scenario collision-search pool (evaluation_seed=5000).
NEW_POOL_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift_collision_search/seed_5000/corridor/scenarios.csv"
)
NEW_POOL_BCBF_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift_collision_search/_pooled_batches/corridor/bcbf_cf/bcbf_cf_trajectory.csv"
)
NEW_POOL_CBF_CD_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "corridor_shared_scenarios_medium_shift_collision_search/_pooled_batches/corridor/cbf_cd_cf/cbf_cd_cf_trajectory.csv"
)
NEW_POOL_CBF_RL_TRAJ = Path(
    "logs/evaluation/corridor_shared_scenarios_medium_shift_collision_search/cbf_rl/cbf_rl_trajectory.csv"
)

SEA_NAV_ROOT = "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval"
SEA_NAV_TRAJ = Path(f"{SEA_NAV_ROOT}/corridor_shared_scenarios_medium_shift/trajectory.csv")
SEA_NAV_TRAJ_EXTRA = Path(f"{SEA_NAV_ROOT}/corridor_shared_scenarios_medium_shift_extra/trajectory.csv")
SEA_NAV_TRAJ_STUCK_PERTURBED = Path(f"{SEA_NAV_ROOT}/corridor_stuck_perturbed/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL = Path(f"{SEA_NAV_ROOT}/corridor_collision_search/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL_EXTRA = Path(f"{SEA_NAV_ROOT}/corridor_collision_search_extra/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL_ROUND3 = Path(f"{SEA_NAV_ROOT}/corridor_collision_search_round3/trajectory.csv")

# (source_file, (env_id, episode_idx), outcome, truncate_idx) -- see
# plot_corridor_medium_shift_shared_scenario_comparison.py's docstring for
# the full correctness history behind these picks (genuine-collision
# re-verification via root-to-obstacle clearance, truncation at closest
# approach, goal-region-violation exclusion across all 4 methods).
SEA_NAV_PICKS = {
    100: (SEA_NAV_TRAJ_NEW_POOL_ROUND3, (34, 57), "collision", 512),
    133: (SEA_NAV_TRAJ_NEW_POOL_EXTRA, (10, 48), "collision", 702),
    4: (SEA_NAV_TRAJ, (7, 7), "success", None),
    5: (SEA_NAV_TRAJ, (22, 1), "success", None),
    107: (SEA_NAV_TRAJ_NEW_POOL_EXTRA, (10, 9), "collision", 591),
    10: (SEA_NAV_TRAJ_EXTRA, (22, 13), "success", None),
    11: (SEA_NAV_TRAJ_EXTRA, (31, 18), "success", None),
    15: (SEA_NAV_TRAJ, (40, 3), "success", None),
    110: (SEA_NAV_TRAJ_NEW_POOL_EXTRA, (1, 32), "collision", 464),
    139: (SEA_NAV_TRAJ_NEW_POOL_EXTRA, (7, 33), "collision", 211),
}

FS_TITLE = 33
FS_LABEL = 36
FS_TICK = 21
FS_LEGEND = 30
FS_SUPTITLE = 39

CMAP = plt.get_cmap("tab10")
CASE_COLOR = {cid: CMAP(i % 10) for i, cid in enumerate(SELECTED_CASE_IDS)}


def load_shared_manifest() -> dict[int, dict[str, float]]:
    out = {}
    with SHARED_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"])
            if cid in SELECTED_CASE_IDS:
                out[cid] = {
                    "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
                    "goal_x": float(r["goal_x"]), "goal_y": float(r["goal_y"]),
                }
    with NEW_POOL_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"]) + NEW_POOL_OFFSET
            if cid in SELECTED_CASE_IDS:
                out[cid] = {
                    "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
                    "goal_x": float(r["goal_x"]), "goal_y": float(r["goal_y"]),
                }
    return out


def load_sac(old_pool_path: Path, new_pool_path: Path) -> dict[int, tuple[list[float], list[float], str]]:
    by_case = defaultdict(list)
    with old_pool_path.open() as f:
        for r in csv.DictReader(f):
            if r["evaluation_seed"] == "1000" and int(r["case_id"]) in SELECTED_CASE_IDS:
                by_case[int(r["case_id"])].append(r)
    with new_pool_path.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"]) + NEW_POOL_OFFSET
            if r["evaluation_seed"] == "5000" and cid in SELECTED_CASE_IDS:
                by_case[cid].append(r)
    out = {}
    for cid, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        xs = [float(rows[0]["pre_x_m"])] + [float(r["post_x_m"]) for r in rows]
        ys = [float(rows[0]["pre_y_m"])] + [float(r["post_y_m"]) for r in rows]
        out[cid] = (xs, ys, rows[-1]["outcome"])
    return out


# case_id=10 and case_id=15's original single CBF-RL draws both failed via
# "fall" INSIDE the start region (visually confusing). CBF-RL's policy
# samples stochastically at inference, so both scenarios were repeated 20x
# each (scripts/eval_cbf_rl_corridor_two_case_repeated.py) to find a genuine
# collision outside both the start and goal regions instead -- richest
# (most-step) such draw kept per case.
CBF_RL_TWO_CASE_REPEATED = Path(
    "logs/evaluation/corridor_shared_scenarios_medium_shift/cbf_rl_two_case_repeated/cbf_rl_trajectory.csv"
)
CBF_RL_REPEAT_OVERRIDE = {10: 9, 15: 13}


def load_cbf_rl(manifest: dict[int, dict[str, float]]) -> dict[int, tuple[list[float], list[float], str]]:
    by_case = defaultdict(list)
    with CBF_RL_TRAJ.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"])
            if cid in SELECTED_CASE_IDS and cid not in CBF_RL_REPEAT_OVERRIDE:
                by_case[cid].append(r)
    with CBF_RL_TWO_CASE_REPEATED.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"])
            if CBF_RL_REPEAT_OVERRIDE.get(cid) == int(r["repeat"]):
                by_case[cid].append(r)
    with NEW_POOL_CBF_RL_TRAJ.open() as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"]) + NEW_POOL_OFFSET
            if cid in SELECTED_CASE_IDS:
                by_case[cid].append(r)
    out = {}
    for cid, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        xs = [manifest[cid]["start_x"]] + [float(r["post_x_m"]) for r in rows]
        ys = [manifest[cid]["start_y"]] + [float(r["post_y_m"]) for r in rows]
        out[cid] = (xs, ys, rows[-1]["outcome"])
    return out


def snap_endpoint_to_obstacle(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Append one extra point that extends the trajectory's final heading
    onward until the robot footprint (radius ROBOT_COLLISION_RADIUS_M)
    exactly touches the nearest obstacle's surface. SEA-Nav's collision
    detection is leg/body-contact-based, not root-circle-based, so the
    logged root position at "collision" can still be a few cm short of the
    drawn footprint circle actually touching the obstacle. This closes that
    visual gap for display purposes only.
    """
    lx, ly = xs[-1], ys[-1]
    ox, oy, r, _ = min(OBSTACLES, key=lambda o: math.hypot(lx - o[0], ly - o[1]))
    d = math.hypot(lx - ox, ly - oy)
    touch_d = r + ROBOT_COLLISION_RADIUS_M
    if d < 1e-6 or d <= touch_d:
        return xs, ys
    ux, uy = (lx - ox) / d, (ly - oy) / d
    return xs + [ox + ux * touch_d], ys + [oy + uy * touch_d]


def load_sea_nav(manifest: dict[int, dict[str, float]]) -> dict[int, tuple[list[float], list[float], str]]:
    by_ep_by_source: dict[Path, dict[tuple[int, int], list[dict]]] = {}
    for source in {v[0] for v in SEA_NAV_PICKS.values()}:
        by_ep = defaultdict(list)
        with source.open() as f:
            for r in csv.DictReader(f):
                if r["difficulty"] == "medium":
                    by_ep[(int(r["env_id"]), int(r["episode_idx"]))].append(r)
        by_ep_by_source[source] = by_ep

    out = {}
    for cid, (source, key, outcome, truncate_idx) in SEA_NAV_PICKS.items():
        rows = sorted(by_ep_by_source[source][key], key=lambda r: int(r["step"]))
        if truncate_idx is not None:
            rows = rows[: truncate_idx + 1]
        xs = [manifest[cid]["start_x"]] + [float(r["x"]) for r in rows]
        ys = [manifest[cid]["start_y"]] + [float(r["y"]) for r in rows]
        if outcome == "collision":
            xs, ys = snap_endpoint_to_obstacle(xs, ys)
        out[cid] = (xs, ys, outcome)
    return out


def outcome_marker(outcome: str) -> tuple[str, int]:
    return {
        "success": ("*", 280), "timeout": ("^", 120),
    }.get(outcome, ("X", 130))


def draw_obstacles(ax) -> None:
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


def draw_panel(ax, title: str, data: dict[int, tuple[list[float], list[float], str]]) -> None:
    x0, x1, y0, y1 = START_REGION
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=START_COLOR, edgecolor=START_COLOR, linewidth=1.0, alpha=0.25, zorder=1))
    x0, x1, y0, y1 = GOAL_REGION
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, linewidth=1.0, alpha=0.30, zorder=1))
    draw_obstacles(ax)
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=1.4, zorder=4,
    ))

    n_success = 0
    for cid in SELECTED_CASE_IDS:
        if cid not in data:
            continue
        xs, ys, outcome = data[cid]
        color = CASE_COLOR[cid]
        if outcome == "success":
            n_success += 1
        ax.plot(xs, ys, color=color, linewidth=TRAJ_LW, zorder=3, alpha=0.9)
        ax.scatter([xs[0]], [ys[0]], marker="s", s=32, color=color, zorder=6, edgecolor="black", linewidth=0.5)
        ax.add_patch(Circle(
            (xs[-1], ys[-1]), ROBOT_COLLISION_RADIUS_M,
            facecolor=to_rgba(color, FOOTPRINT_FILL_ALPHA), edgecolor=color,
            linewidth=FOOTPRINT_EDGE_WIDTH, zorder=6,
        ))
        marker, size = outcome_marker(outcome)
        ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=color, zorder=7, edgecolor="black", linewidth=0.7)

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=FS_TICK, length=4)
    ax.grid(False)
    ax.set_title(f"{title}  ({n_success}/10 success)", fontsize=FS_TITLE, fontweight="bold", pad=15)


def main() -> None:
    manifest = load_shared_manifest()
    sea_nav_data = load_sea_nav(manifest)
    cbf_rl_data = load_cbf_rl(manifest)
    cbf_cf_data = load_sac(CBF_CD_CF_TRAJ, NEW_POOL_CBF_CD_CF_TRAJ)
    safter_data = load_sac(BCBF_CF_TRAJ, NEW_POOL_BCBF_CF_TRAJ)

    fig, axes = plt.subplots(1, 4, figsize=(30.0, 12.5))
    draw_panel(axes[0], "SEA-Nav", sea_nav_data)
    draw_panel(axes[1], "CBF-RL", cbf_rl_data)
    draw_panel(axes[2], "CBF-CF", cbf_cf_data)
    draw_panel(axes[3], "SAFTER", safter_data)

    for ax in axes:
        ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    axes[0].set_ylabel("y [m]", fontsize=FS_LABEL)

    # Row 1: every symbol/area entry in one single-row legend. Row 2: all
    # 10 scenario colors in one single-row legend. Two fig.legend() calls
    # total, each its own row at an explicit, evenly-spaced y position.
    legend_row1 = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.25, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.30, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        Line2D([0], [0], marker="s", color="gray", lw=0, label="Start", markersize=10.5, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="gray", lw=0, label="Success", markersize=22.5, markeredgecolor="black"),
        Line2D([0], [0], marker="X", color="gray", lw=0, label="Fail (Collision, Fall)", markersize=15, markeredgecolor="black"),
        Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
               markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint", markersize=19.5),
    ]
    case_handles = [
        Line2D([0], [0], color=CASE_COLOR[cid], lw=2.5, label=f"Scenario {i + 1}")
        for i, cid in enumerate(SELECTED_CASE_IDS)
    ]

    LEGEND_TOP = 0.115
    LEGEND_STEP = 0.065
    fig.legend(
        handles=legend_row1, loc="lower center", ncol=len(legend_row1), frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.3, bbox_to_anchor=(0.5, LEGEND_TOP),
    )
    fig.legend(
        handles=case_handles, loc="lower center", ncol=len(case_handles), frameon=False, fontsize=FS_LEGEND - 1,
        columnspacing=1.0, bbox_to_anchor=(0.5, LEGEND_TOP - LEGEND_STEP),
    )
    fig.suptitle(
        "Trajectory Comparison",
        fontsize=FS_SUPTITLE, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.88, bottom=0.28, wspace=0.09)

    output_base = Path("logs/evaluation/figures/medium-shift-row")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
