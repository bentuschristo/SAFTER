#!/usr/bin/env python3
"""4-method (SEA-Nav / CBF-RL / CBF-CF / SAFTER) 10-sample trajectory
comparison on "corridor" (Medium-Shift), requested 2026-09-24: each method
evaluated on the IDENTICAL 10 (start, goal, yaw) scenario pairs (case_id in
SELECTED_CASE_IDS, drawn from 25 scenarios shared across all 4 methods --
mirrors scripts/plot_corridor_derived_highshift_v5_shared_scenario_comparison.py's
methodology exactly, with two additions requested for this map:

  1. No fail/timeout marker may land inside the goal region -- candidate
     episodes whose endpoint falls inside GOAL_REGION are excluded UNLESS
     their outcome is "success" (success is supposed to end there).
  2. 10 cases instead of 8.

Target success counts (rounded per 2026-09-24 session instructions from the
paper's Medium-Shift table screenshot: SAFTER's real 88.6% rounds UP to
90%/9-of-10; every other method's real rate rounds DOWN to the nearest 10%):
  SAFTER (bcbf_cf):     88.6% -> 9/10
  CBF-CF (cbf_cd_cf):   48.4% -> 4/10
  CBF-RL:               45.6% -> 4/10 (achieved 5/10, off-by-one -- closest
                         achievable joint fit across all 4 targets at once,
                         see the brute-force search in this session's notes)
  SEA-Nav:               50.6% -> 5/10

bcbf_cf/cbf_cd_cf reused directly from a fresh 25-case structured run
sharing one manifest (logs/evaluation/laser_unseen_maps_structured_refixed/
corridor_shared_scenarios_medium_shift/). CBF-RL and SEA-Nav were run fresh
on the exact same 25 (start,goal,yaw) tuples (scripts/
eval_cbf_rl_corridor_shared_scenarios.py, scripts/
generate_seanav_corridor_shared_scenarios_bank.py). SEA-Nav case picks are
matched by exact start-position proximity (<0.05m) directly against
trajectory.csv, NOT via episodes.csv's episode_idx (confirmed unreliable
during the High-Shift/v5 version of this figure -- overshoot episodes shift
the two files' episode_idx numbering out of sync).

2026-09-24 revision (round 1): SEA-Nav's "stuck" (other_failure) cases were
first replaced with 5 episodes tagged outcome=="collision", per request
("pick the failed ones as the one where it collides instead of fall or
stuck"). This surfaced a second, more serious bug on inspection (see round
2 below): SEA-Nav's per-row `outcome` is a per-EPISODE label stamped on
every row of that episode, not a live per-step status, AND it fires from
wall hits and falls (a >50N contact-force spike on ANY body), not only
genuine obstacle contact. Investigation (session notes) established that a
*stuck* equilibrium is SEA-Nav's actual dominant failure mode here; a fresh
40-scenario pool (evaluation_seed=5000, scripts/
generate_seanav_corridor_collision_search_bank.py,
eval_cbf_rl_corridor_collision_search.py, same bcbf_cf/cbf_cd_cf checkpoints
and flags) was searched across ~3600 SEA-Nav episodes to find enough
"collision"-tagged candidates to choose from in the first place.

2026-09-24 revision (round 2): re-verified every "collision" pick by
checking root-to-obstacle clearance across each FULL trajectory (not just
the final logged row, which can be hundreds of steps into a post-collision
coast/slide before the batch actually resets that env). 3 of the original
5 picks (old case_id 21 and 2, new-pool case_id 19) never came within 0.4m
of any of the 9 obstacles at their closest approach -- not genuine obstacle
hits (most likely mislabeled falls or, for case_id 19, an actual wall hit).
Also caught a separate mismatch: what was labeled new-pool "case_id 14" was
actually new-pool case_id 24 (a start/goal identification error). Replaced
with 5 collisions independently re-verified to come within <=0.20m of an
obstacle surface at some point in the trajectory (a small leg-extension
allowance past the exact analytical circle), found by re-scanning every
SEA-Nav batch run this session; each plotted line/marker is TRUNCATED at
that true collision point rather than wherever the episode's log ends.
new-pool cases are namespaced as 100+case_id throughout this script.
5 SEA-Nav SUCCESS cases (4, 5, 10, 11, 15) are unchanged throughout both
rounds. Final achieved totals: SAFTER 9/10 and CBF-CF 4/10 match the
pre-revision figure exactly; CBF-RL is 4/10 (was 6/10) -- the strict
"genuine obstacle collision only" constraint leaves few real candidates to
choose from, and none of the available ones let CBF-RL's total go higher
without breaking the SAFTER/CBF-CF match.

Layout: 2x2 grid -- top-left SEA-Nav, top-right CBF-RL, bottom-left CBF-CF
(cbf_cd_cf), bottom-right SAFTER (bCBF-CF). Trajectories are colored by
case_id (same color = same scenario across all 4 panels).
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

START_COLOR = "#66ff66"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#f0431f"

# Real top-down MuJoCo capture of the corridor map (scripts/capture_map_topdown.py),
# used as the panel background in place of the schematic gray obstacle patches.
# Camera params there (distance=20, fovy=45) fix this image's world extent to
# +-half_span; must stay in sync with that script if the camera is retuned.
BACKGROUND_IMAGE_PATH = Path("logs/evaluation/figures/map_captures/corridor_topdown_raw.png")
BACKGROUND_HALF_SPAN = 20.0 * math.tan(math.radians(45.0) / 2.0)
_BACKGROUND_IMAGE = np.asarray(Image.open(BACKGROUND_IMAGE_PATH)) if BACKGROUND_IMAGE_PATH.exists() else None
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
# case_id=2's exact-start draws never produced a collision or timeout across
# ~90 samples (only success/stuck). Perturbing its start by ~0.45m (scripts/
# generate_seanav_corridor_stuck_perturbed_bank.py, 4 jittered variants x
# ~40 draws each) surfaced exactly one genuine collision -- kept here.
SEA_NAV_TRAJ_STUCK_PERTURBED = Path(f"{SEA_NAV_ROOT}/corridor_stuck_perturbed/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL = Path(f"{SEA_NAV_ROOT}/corridor_collision_search/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL_EXTRA = Path(f"{SEA_NAV_ROOT}/corridor_collision_search_extra/trajectory.csv")
SEA_NAV_TRAJ_NEW_POOL_ROUND3 = Path(f"{SEA_NAV_ROOT}/corridor_collision_search_round3/trajectory.csv")

# IMPORTANT (2026-09-24 correctness fix): SEA-Nav's per-row `outcome` field
# is a per-EPISODE label stamped retroactively on every logged row once the
# episode concludes -- it is NOT a live per-step status. Two consequences:
#   1. Once a real collision/hard-force-reset fires, the sim keeps stepping
#      and logging position for potentially hundreds more steps (coasting/
#      sliding) before the batch actually resets that env. Using the LAST
#      logged row as "the collision point" (the previous version's bug)
#      plots wherever the robot drifted to afterward, which can be far from
#      any obstacle and look like a bogus mid-air/open-space failure.
#   2. "collision" fires from a >50N contact-force spike on ANY body part,
#      not only genuine obstacle contact -- it also fires for wall hits and,
#      apparently, hard falls/stumbles (large transient contact forces with
#      no obstacle nearby at all). Confirmed by checking root-to-obstacle
#      clearance across each full trajectory: several previously-accepted
#      "collision" picks (old case_id 21, case_id 2, new-pool case_id 19)
#      never came within even 0.4m of any of the 9 obstacles at their
#      closest approach -- not genuine obstacle collisions.
#
# Fix: every entry below was found via a full re-scan across ALL SEA-Nav
# batches run this session, keeping only "collision"-tagged episodes whose
# root position comes within <=0.20m of an obstacle SURFACE (a small
# leg-extension allowance past the exact analytical circle) at some point
# in the trajectory, and TRUNCATING the plotted line/marker at that point
# (the 4th tuple element, a step index) rather than the episode's full
# logged length. "success" entries have no coasting artifact (episode ends
# at the goal) and are left untruncated (None).
# 2026-09-24 revision (round 3): old case_id=7 and new-pool case_id=24 (110
# was originally 124) both had a fail/timeout landing INSIDE the goal region
# for at least one of the OTHER 3 methods (cbf_rl=fall / cbf_rl=timeout
# respectively) -- a requirement established earlier for SEA-Nav is now
# enforced for ALL 4 methods jointly. Replaced with new-pool case_id=0 and
# case_id=10, independently re-verified clean on all 4 methods.
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

FS_TITLE = 22
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 20
FS_SUPTITLE = 26

# Custom palette (replaces tab10) chosen for contrast against the real
# MuJoCo background: dark-blue checkered floor, red-orange obstacles, pale
# green initial region, pale yellow goal region. tab10's blue/gray/brown/olive
# all blended into that background, so those four are swapped out here.
SCENARIO_PALETTE = [
    "#FFFFFF",  # white       (was blue -- blended into the floor)
    "#FF8C00",  # dark orange
    "#2CA02C",  # green       (dark enough to read against the pale region fill)
    "#D62728",  # red
    "#9467BD",  # purple
    "#FF00FF",  # magenta     (was brown -- too low-contrast)
    "#E377C2",  # pink
    "#FFC300",  # gold        (was gray -- too low-contrast)
    "#FF6F61",  # coral       (was olive -- blended into the goal region)
    "#17BECF",  # cyan
]
CASE_COLOR = {cid: SCENARIO_PALETTE[i % 10] for i, cid in enumerate(SELECTED_CASE_IDS)}


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
    detection is leg/body-contact-based, not root-circle-based (see this
    file's docstring), so the logged root position at "collision" can still
    be a few cm short of the drawn footprint circle actually touching the
    obstacle. This closes that visual gap for display purposes only.
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
    if _BACKGROUND_IMAGE is not None:
        ax.imshow(
            _BACKGROUND_IMAGE, zorder=0,
            extent=(-BACKGROUND_HALF_SPAN, BACKGROUND_HALF_SPAN, -BACKGROUND_HALF_SPAN, BACKGROUND_HALF_SPAN),
        )
    else:
        draw_obstacles(ax)
    x0, x1, y0, y1 = START_REGION
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=START_COLOR, edgecolor=START_COLOR, linewidth=1.0, alpha=0.30, zorder=1))
    x0, x1, y0, y1 = GOAL_REGION
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, linewidth=1.0, alpha=0.30, zorder=1))
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
        ax.plot(
            xs, ys, color=color, linewidth=TRAJ_LW, zorder=3, alpha=0.9,
            path_effects=[pe.Stroke(linewidth=TRAJ_LW + 1.4, foreground="black"), pe.Normal()],
        )
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
    ax.set_title(f"{title}  ({n_success}/10 success)", fontsize=FS_TITLE, fontweight="bold", pad=10)


def main() -> None:
    manifest = load_shared_manifest()
    sea_nav_data = load_sea_nav(manifest)
    cbf_rl_data = load_cbf_rl(manifest)
    cbf_cf_data = load_sac(CBF_CD_CF_TRAJ, NEW_POOL_CBF_CD_CF_TRAJ)
    safter_data = load_sac(BCBF_CF_TRAJ, NEW_POOL_BCBF_CF_TRAJ)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.8))
    draw_panel(axes[0, 0], "SEA-Nav", sea_nav_data)
    draw_panel(axes[0, 1], "CBF-RL", cbf_rl_data)
    draw_panel(axes[1, 0], "CBF-CF", cbf_cf_data)
    draw_panel(axes[1, 1], "SAFTER", safter_data)

    for ax in axes[1, :]:
        ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]", fontsize=FS_LABEL)

    # Single vertical legend column on the left of the whole figure (per
    # request, matching plot_circle_and_square_fixed_scenario_trajectories.py's
    # style) instead of rows spanning the full width below the panels.
    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.25, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.30, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        Line2D([0], [0], marker="s", color="gray", lw=0, label="Start", markersize=10.5, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="gray", lw=0, label="Success", markersize=22.5, markeredgecolor="black"),
        Line2D([0], [0], marker="X", color="gray", lw=0, label="Collision", markersize=15, markeredgecolor="black"),
        Line2D([0], [0], marker="o", markerfacecolor=to_rgba("gray", FOOTPRINT_FILL_ALPHA), markeredgecolor="gray",
               markeredgewidth=FOOTPRINT_EDGE_WIDTH, color="none", lw=0, label="Robot footprint", markersize=19.5),
    ]
    case_handles = [
        Line2D(
            [0], [0], color=CASE_COLOR[cid], lw=2.5, label=f"Scenario {i + 1}",
            path_effects=[pe.Stroke(linewidth=3.9, foreground="black"), pe.Normal()],
        )
        for i, cid in enumerate(SELECTED_CASE_IDS)
    ]
    fig.legend(
        handles=legend_handles + case_handles, loc="center left", ncol=1, frameon=False,
        fontsize=FS_LEGEND - 1, bbox_to_anchor=(0.005, 0.5), labelspacing=1.1,
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.06, hspace=0.14, wspace=0.08)

    output_base = Path("logs/evaluation/figures/corridor_medium_shift_shared_scenario_comparison")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
