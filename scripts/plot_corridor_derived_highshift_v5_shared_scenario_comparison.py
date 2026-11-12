#!/usr/bin/env python3
"""4-method (SEA-Nav / CBF-RL / CBF-CF / SAFTER) 8-sample trajectory
comparison on corridor_derived_highshift_v5, requested 2026-09-24: each
method evaluated on the IDENTICAL 8 (start, goal, yaw) scenario pairs
(case_id in SELECTED_CASE_IDS, drawn from the 20 scenarios shared across
all 4 methods -- see scripts/generate_corridor_derived_highshift_v5_
fixed_scenario_manifest.py's sibling shared-scenario infra:
scripts/eval_cbf_rl_v5_shared_scenarios.py,
scripts/generate_seanav_v5_shared_scenarios_bank.py, and bcbf_cf/cbf_cd_cf
reused directly from the existing full-500-case structured run since it
already shares evaluation_seed=1000's manifest across both methods).

The 8 case_ids were chosen (brute-force search over all C(20,8) subsets of
the 20 shared scenarios) to jointly best match each method's REAL measured
success/outcome rate (SAFTER: server 61.4%, CBF-CF: server 33.8%, CBF-RL:
local 31.0%/15.4% timeout, SEA-Nav: local 0.0%) -- see the 2026-09-24
session notes. Achieved: SAFTER 5/8 success, CBF-CF 3/8, CBF-RL 2/8 success
+1 timeout+1 fall+4 collision, SEA-Nav 0/8 success (5 collision + 3
other_failure) -- an exact or near-exact match on every method. case_id=3
(originally part of the best-scoring subset) was later swapped for
case_id=19 after discovering every one of its available SEA-Nav failure
draws terminated within 7 steps of spawn, always inside the initial
region -- case_id=19 gives the same near-optimal fit with a >1000-step
SEA-Nav failure draw instead.

Layout: 2x2 grid -- top-left SEA-Nav, top-right CBF-RL, bottom-left CBF-CF
(cbf_cd_cf), bottom-right SAFTER (bCBF-CF). Trajectories are colored by
case_id (same color = same scenario across all 4 panels), consistent with
the "initial condition and goal must be the same across methods"
requirement.
"""
from __future__ import annotations

import csv
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
    (-4.000, 2.2, 0.65, "circle"),
    (-2.000, 2.2, 0.65, "circle"),
    (2.000, 2.2, 0.65, "circle"),
    (4.000, 2.2, 0.65, "circle"),
    (-4.080, -0.4, 0.50, "circle"),
    (-2.350, -1.3, 0.65, "circle"),
    (0.000, -2.2, 0.90, "square"),
    (2.350, -1.3, 0.65, "circle"),
    (4.080, -0.4, 0.50, "circle"),
]
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

SELECTED_CASE_IDS = [0, 2, 19, 4, 5, 7, 13, 16]

SHARED_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/"
    "seed_1000/corridor_derived_highshift_v5/scenarios.csv"
)
BCBF_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/"
    "_pooled_batches/corridor_derived_highshift_v5/bcbf_cf/bcbf_cf_trajectory.csv"
)
CBF_CD_CF_TRAJ = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/"
    "_pooled_batches/corridor_derived_highshift_v5/cbf_cd_cf/cbf_cd_cf_trajectory.csv"
)
CBF_RL_TRAJ = Path("logs/evaluation/corridor_derived_highshift_v5_shared_scenarios/cbf_rl/cbf_rl_trajectory.csv")
SEA_NAV_ROOT = "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval"
SEA_NAV_TRAJ = Path(f"{SEA_NAV_ROOT}/corridor_derived_highshift_v5_shared_scenarios/trajectory.csv")
# A second, larger batch (700 episodes/map instead of 200, different eval
# seed) run 2026-09-24 specifically to find richer non-degenerate failure
# draws for case_id 5 -- its only two matches in the original batch were
# 1-2 step near-instant collisions that never left the initial region.
SEA_NAV_TRAJ_EXTRA = Path(f"{SEA_NAV_ROOT}/corridor_derived_highshift_v5_shared_scenarios_extra/trajectory.csv")

# (env_id, episode_idx) chosen per case_id -- matched by exact start-position
# proximity (<0.05m) directly against trajectory.csv, NOT by cross-
# referencing episodes.csv's episode_idx (that numbering does not align
# with trajectory.csv's -- confirmed 2026-09-24: episodes.csv only counts
# episodes up to each map's target count while overshoot episodes still get
# logged into trajectory.csv, shifting the two files' episode_idx out of
# sync for envs that had overshoot; matching by goal proximity + that stale
# episode_idx silently pulled entirely wrong episodes/trajectories for
# several cases). Outcome is each pick's own real recorded outcome. Among
# picks with the exact same case_id, the richest (most-step) available
# draw within the SAME outcome class is preferred for visual clarity --
# case_id=3 was dropped entirely (swapped for case_id=19) because every
# one of its 9 available failure draws across two eval batches terminated
# in <=7 steps, always right next to the start; case_id=5's original pick
# (2 steps) was replaced with a 789-step draw found in the extra batch.
SEA_NAV_PICKS = {
    0: (SEA_NAV_TRAJ, (5, 1), "other_failure"),
    2: (SEA_NAV_TRAJ, (8, 1), "collision"),
    19: (SEA_NAV_TRAJ, (56, 1), "other_failure"),
    4: (SEA_NAV_TRAJ, (8, 6), "collision"),
    5: (SEA_NAV_TRAJ_EXTRA, (50, 20), "other_failure"),
    7: (SEA_NAV_TRAJ, (11, 0), "collision"),
    13: (SEA_NAV_TRAJ, (32, 0), "collision"),
    16: (SEA_NAV_TRAJ, (29, 0), "collision"),
}

FS_TITLE = 22
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 20
FS_SUPTITLE = 26

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
    return out


def load_sac(path: Path) -> dict[int, tuple[list[float], list[float], str]]:
    by_case = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["evaluation_seed"] == "1000" and int(r["case_id"]) in SELECTED_CASE_IDS:
                by_case[int(r["case_id"])].append(r)
    out = {}
    for cid, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        xs = [float(rows[0]["pre_x_m"])] + [float(r["post_x_m"]) for r in rows]
        ys = [float(rows[0]["pre_y_m"])] + [float(r["post_y_m"]) for r in rows]
        out[cid] = (xs, ys, rows[-1]["outcome"])
    return out


def load_cbf_rl(manifest: dict[int, dict[str, float]]) -> dict[int, tuple[list[float], list[float], str]]:
    by_case = defaultdict(list)
    with CBF_RL_TRAJ.open() as f:
        for r in csv.DictReader(f):
            if int(r["case_id"]) in SELECTED_CASE_IDS:
                by_case[int(r["case_id"])].append(r)
    out = {}
    for cid, rows in by_case.items():
        rows.sort(key=lambda r: int(r["step"]))
        xs = [manifest[cid]["start_x"]] + [float(r["post_x_m"]) for r in rows]
        ys = [manifest[cid]["start_y"]] + [float(r["post_y_m"]) for r in rows]
        out[cid] = (xs, ys, rows[-1]["outcome"])
    return out


def load_sea_nav(manifest: dict[int, dict[str, float]]) -> dict[int, tuple[list[float], list[float], str]]:
    by_ep_by_source: dict[Path, dict[tuple[int, int], list[dict]]] = {}
    for source in {v[0] for v in SEA_NAV_PICKS.values()}:
        by_ep = defaultdict(list)
        with source.open() as f:
            for r in csv.DictReader(f):
                if r["difficulty"] == "hard":
                    by_ep[(int(r["env_id"]), int(r["episode_idx"]))].append(r)
        by_ep_by_source[source] = by_ep

    out = {}
    for cid, (source, key, outcome) in SEA_NAV_PICKS.items():
        rows = sorted(by_ep_by_source[source][key], key=lambda r: int(r["step"]))
        # Snap the plotted start to the exact shared manifest coordinate
        # (matching load_cbf_rl) -- the sim's own first logged sample is
        # already within ~0.05m of it (that's how the pick was validated),
        # so this just removes the residual sub-tick offset for a clean
        # visual match across all 4 panels.
        xs = [manifest[cid]["start_x"]] + [float(r["x"]) for r in rows]
        ys = [manifest[cid]["start_y"]] + [float(r["y"]) for r in rows]
        out[cid] = (xs, ys, outcome)
    return out


def outcome_marker(outcome: str) -> tuple[str, int]:
    # Collision / fall / other_failure are all real navigation failures with
    # different low-level causes but the same high-level meaning here, so
    # they're drawn identically ("Fail"); success and timeout stay distinct.
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
    ax.set_title(f"{title}  ({n_success}/8 success)", fontsize=FS_TITLE, fontweight="bold", pad=10)


def main() -> None:
    manifest = load_shared_manifest()
    sea_nav_data = load_sea_nav(manifest)
    cbf_rl_data = load_cbf_rl(manifest)
    cbf_cf_data = load_sac(CBF_CD_CF_TRAJ)
    safter_data = load_sac(BCBF_CF_TRAJ)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 14.0))
    draw_panel(axes[0, 0], "SEA-Nav", sea_nav_data)
    draw_panel(axes[0, 1], "CBF-RL", cbf_rl_data)
    draw_panel(axes[1, 0], "CBF-CF", cbf_cf_data)
    draw_panel(axes[1, 1], "SAFTER", safter_data)

    for ax in axes[1, :]:
        ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]", fontsize=FS_LABEL)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.25, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.30, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        Line2D([0], [0], marker="s", color="gray", lw=0, label="Start", markersize=10.5, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="gray", lw=0, label="Success", markersize=22.5, markeredgecolor="black"),
        Line2D([0], [0], marker="^", color="gray", lw=0, label="Timeout", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="X", color="gray", lw=0, label="Fail (Collision, Fall)", markersize=15, markeredgecolor="black"),
        Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
               markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint (0.30m)", markersize=19.5),
    ]
    case_handles = [
        Line2D([0], [0], color=CASE_COLOR[cid], lw=2.5, label=f"Scenario {i + 1}")
        for i, cid in enumerate(SELECTED_CASE_IDS)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.3, bbox_to_anchor=(0.5, 0.035),
    )
    fig.legend(
        handles=case_handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEGEND - 1,
        columnspacing=1.0, bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Trajectory Comparison",
        fontsize=FS_SUPTITLE, fontweight="bold", y=1.015,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.16, hspace=0.24, wspace=0.06)

    output_base = Path("logs/evaluation/figures/corridor_derived_highshift_v5_shared_scenario_comparison")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
