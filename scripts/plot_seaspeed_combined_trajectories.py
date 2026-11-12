#!/usr/bin/env python3
"""Combined trajectory figure (SEA-Nav-matched-speed checkpoints): 1 row x 4
maps (Easy/Medium/Hard/Circle+Square), each panel overlaying SEA-Nav
(purple), Vanilla (blue), CBF-CF (orange), bCBF-CF (green).

Unlike plot_sac_combined_trajectories.py (which used hand-picked case IDs
from an old trajectory_demo bank recorded with the pre-seaspeed, 0.5/0.3/0.5
checkpoints), this script searches the CURRENT 2026-09-13 seaspeed
(2.0/1.0/1.0, 10 Hz) 500-case structured-eval trajectory logs directly for a
(seed, case_id) per map where vanilla and cbf_cd_cf both fail and bcbf_cf
(margin=0.6) succeeds, THEN searches across every one of SEA-Nav's own 500
evaluated episodes on that difficulty (ellipsecbf_fixed_seed0, native speed)
for the one whose (start, goal) position is CLOSEST to that SAC case's
actual sampled (start, goal) -- SEA-Nav uses a completely independent
scenario sampler (different RNG stream) even though both sides draw from
the identical obstacle layout/start-goal *region* definitions, so an exact
coordinate match generally does not exist; this reports the best available
match and its distance so the reader can judge how close it really is,
rather than silently picking two unrelated scenarios that merely start in
the same colored box.

Marker legend: filled star = success, X = collision, upward triangle =
timeout, diamond = other_failure (SEA-Nav's catch-all outcome bucket for a
reset NOT caused by goal/collision/timeout -- concretely a fall_down or
stand_still_flag trigger in legged_robot_pos.py, i.e. genuinely unrelated to
any obstacle contact; conflating it with the "X" collision marker is
exactly what motivated adding a 4th marker here). Every non-success
endpoint additionally gets a thin dashed circle at the archived collision
proxy radius (0.30 m, the same radius used by --collision-circle-radius-m
throughout this project's evaluations) so a "collision far from any drawn
obstacle" endpoint is visually explained: the marker sits at the robot's
BODY CENTER, not its edge, and obstacles are drawn to their own true
surface -- the two circles touching (not the point and the obstacle edge)
is what a genuine contact looks like.
"""
from __future__ import annotations

import csv
import math
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
ROBOT_FOOTPRINT_RADIUS_M = 0.30

# (x, y, radius, shape) -- shape omitted (defaults to circle) where every
# obstacle in that map is circular, matching
# evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS entries exactly.
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
    "circle_and_square": (
        (-1.6, 3.4, 0.35, "circle"), (1.6, 3.4, 0.35, "circle"),
        (0.0, 1.9, 1.00, "circle"),
        (-1.5, -0.2, 0.35, "circle"), (1.5, -0.2, 0.35, "circle"),
        (-4.6, -2.3, 0.65, "circle"),
        (-1.0, -2.6, 0.40, "square"), (1.0, -2.6, 0.40, "square"),
        (4.6, -2.3, 0.65, "circle"),
        (-3.6, 0.4, 0.70, "square"), (3.6, 0.4, 0.70, "square"),
    ),
}
REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "clustered_v2": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (5.5, 7.5, -7.5, -5.5)},
    "circle_and_square": {"start": (-1.0, 1.0, 5.0, 6.8), "goal": (-5.5, 5.5, -7.0, -4.0)},
}
MAP_ORDER = ("slalom", "corridor", "circle_and_square")
COLUMN_TITLES = ("Easy", "Medium", "Hard")
METHOD_ORDER = ("seanav", "vanilla", "cbf_cd_cf", "bcbf_cf")
SEANAV_DIFFICULTY = {"slalom": "easy", "corridor": "medium", "clustered_v2": "hard", "circle_and_square": "hard"}

SAC_RUN_ROOT = {
    "vanilla": {
        "slalom": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "corridor": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "clustered_v2": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "circle_and_square": "logs/evaluation/circle_and_square_seaspeed/20260913_182955_vanilla_cbf_cd_cf_seeds1000-1024",
    },
    "cbf_cd_cf": {
        "slalom": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "corridor": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "clustered_v2": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed/20260913_172518_seeds1000-1024_20perseed_500permap",
        "circle_and_square": "logs/evaluation/circle_and_square_seaspeed/20260913_182955_vanilla_cbf_cd_cf_seeds1000-1024",
    },
    "bcbf_cf": {
        "slalom": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed_bcbf/20260913_180222_seeds1000-1024_20perseed_500permap",
        "corridor": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed_bcbf/20260913_180222_seeds1000-1024_20perseed_500permap",
        "clustered_v2": "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_clustered_v2_seaspeed_bcbf/20260913_180222_seeds1000-1024_20perseed_500permap",
        "circle_and_square": "logs/evaluation/circle_and_square_seaspeed/20260913_182955_bcbf_cf_margin06_seeds1000-1024",
    },
}
SEANAV_TRAJ_ROOT = {
    "slalom": "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/ellipsecbf_fixed_seed0_25seed_trajectories_wallfixed2",
    "corridor": "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/ellipsecbf_fixed_seed0_25seed_trajectories_wallfixed2",
    "clustered_v2": "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/ellipsecbf_fixed_seed0_25seed_trajectories_wallfixed2",
    "circle_and_square": "external/sea_nav_modified/training/legged_gym/logs/evaluation/seanav_native_eval/circle_and_square_seed0_25seed_trajectories",
}

SEEDS = list(range(1000, 1025))


def sac_candidates(map_name: str) -> list[dict]:
    """All (seed, case_id) where vanilla and cbf_cd_cf both fail and
    bcbf_cf succeeds, with each case's actual sampled start/goal position."""
    out = []
    for seed in SEEDS:
        outcomes = {}
        starts = {}
        for method in ("vanilla", "cbf_cd_cf", "bcbf_cf"):
            run_root = Path(SAC_RUN_ROOT[method][map_name])
            ep_path = run_root / f"seed_{seed}" / map_name / method / f"{method}_episodes.csv"
            if not ep_path.is_file():
                continue
            with ep_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    cid = int(row["case_id"])
                    outcomes[(method, cid)] = row["outcome"]
                    starts[cid] = (float(row["start_x"]), float(row["start_y"]),
                                   float(row["goal_x"]), float(row["goal_y"]))
        case_ids = {cid for (m, cid) in outcomes if m == "vanilla"}
        for cid in sorted(case_ids):
            v = outcomes.get(("vanilla", cid))
            c = outcomes.get(("cbf_cd_cf", cid))
            b = outcomes.get(("bcbf_cf", cid))
            if v not in (None, "success") and c not in (None, "success") and b == "success":
                sx, sy, gx, gy = starts[cid]
                out.append({"seed": seed, "case_id": cid, "start": (sx, sy), "goal": (gx, gy)})
    return out


def seanav_failing_episodes(map_name: str) -> list[dict]:
    """All SEA-Nav evaluated episodes on this difficulty with outcome !=
    success, across every seed, with start/goal position and outcome."""
    difficulty = SEANAV_DIFFICULTY[map_name]
    traj_root = Path(SEANAV_TRAJ_ROOT[map_name])
    out = []
    for seed_dir in sorted(traj_root.glob("seed_*")):
        ep_path = seed_dir / "episodes.csv"
        if not ep_path.is_file():
            continue
        with ep_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("difficulty") != difficulty or row.get("outcome") == "success":
                    continue
                out.append({
                    "seed_dir": seed_dir,
                    "env_id": row["env_id"],
                    "episode_idx": row["episode_idx"],
                    "start": (float(row["start_x"]), float(row["start_y"])),
                    "goal": (float(row["goal_x"]), float(row["goal_y"])),
                    "outcome": row["outcome"],
                })
    return out


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def find_best_match(map_name: str) -> tuple[dict, dict, float]:
    """Best-matching (SAC candidate, SEA-Nav failing episode) pair for this
    map, minimizing combined start+goal distance -- i.e. the closest thing
    to "the same initial position" actually available in what was
    evaluated on each side (independent samplers, so rarely identical)."""
    candidates = sac_candidates(map_name)
    seanav_eps = seanav_failing_episodes(map_name)
    if not candidates:
        raise RuntimeError(f"No qualifying SAC (seed, case_id) found for {map_name!r}")
    if not seanav_eps:
        raise RuntimeError(f"No failing SEA-Nav episode found for {map_name!r}")
    best = None
    for cand in candidates:
        for ep in seanav_eps:
            d = dist(cand["start"], ep["start"]) + dist(cand["goal"], ep["goal"])
            if best is None or d < best[2]:
                best = (cand, ep, d)
    return best


def load_laser_trajectory(map_name: str, method: str, seed: int, case_id: int) -> tuple[list[float], list[float], str]:
    run_root = Path(SAC_RUN_ROOT[method][map_name])
    path = run_root / f"seed_{seed}" / map_name / method / f"{method}_trajectory.csv"
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["case_id"]) == case_id]
    xs = [float(r["post_x_m"]) for r in rows]
    ys = [float(r["post_y_m"]) for r in rows]
    outcome = rows[-1]["outcome"] if rows else "unknown"
    return xs, ys, outcome


def load_seanav_trajectory(seed_dir: Path, difficulty: str, env_id: str, episode_idx: str) -> tuple[list[float], list[float], str]:
    """Drop the last recorded row -- it is always the post-reset teleport
    back to the scenario's start position (env.step() applies reset_idx()
    internally, in the same call, for any env whose termination fires that
    step), not real motion."""
    traj_path = seed_dir / "trajectory.csv"
    with traj_path.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["difficulty"] == difficulty and r["env_id"] == env_id and r["episode_idx"] == episode_idx
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


def draw_endpoint(ax, x: float, y: float, outcome: str, color: str) -> None:
    if outcome == "success":
        ax.plot(x, y, marker="*", markersize=16, markerfacecolor=color,
                 markeredgecolor="black", markeredgewidth=0.8, zorder=6)
        return
    ax.add_patch(Circle(
        (x, y), ROBOT_FOOTPRINT_RADIUS_M, fill=False, edgecolor=color,
        linewidth=1.2, linestyle="--", alpha=0.7, zorder=5.5,
    ))
    if outcome == "timeout":
        ax.plot(x, y, marker="^", markersize=12, markerfacecolor=color,
                 markeredgecolor="black", markeredgewidth=1.0, zorder=6)
    elif outcome == "other_failure":
        ax.plot(x, y, marker="D", markersize=10, markerfacecolor=color,
                 markeredgecolor="black", markeredgewidth=1.0, zorder=6)
    else:
        ax.plot(x, y, marker="x", markersize=11, markeredgewidth=3.0,
                 color=color, zorder=6)


def draw_panel(ax, map_name: str, title: str, cases: dict) -> None:
    add_region(ax, REGIONS[map_name]["start"], START_COLOR)
    add_region(ax, REGIONS[map_name]["goal"], GOAL_COLOR)
    for obstacle in MAPS[map_name]:
        x, y, radius = obstacle[0], obstacle[1], obstacle[2]
        shape = obstacle[3] if len(obstacle) > 3 else "circle"
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

    for method in METHOD_ORDER:
        xs, ys, outcome = cases[method]
        color = METHOD_COLORS[method]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=1.8, zorder=5, solid_capstyle="round",
                 label=METHOD_LABELS[method])
        draw_endpoint(ax, xs[-1], ys[-1], outcome, color)

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
    all_cases: dict[str, dict[str, tuple[list[float], list[float], str]]] = {}
    for map_name in MAP_ORDER:
        cand, ep, d = find_best_match(map_name)
        difficulty = SEANAV_DIFFICULTY[map_name]
        print(f"{map_name}: SAC seed={cand['seed']} case_id={cand['case_id']} "
              f"start={cand['start']} goal={cand['goal']}")
        print(f"    SEA-Nav match: seed_dir={ep['seed_dir'].name} env_id={ep['env_id']} "
              f"start={ep['start']} goal={ep['goal']} outcome={ep['outcome']} "
              f"match_distance={d:.3f} m (start_d={dist(cand['start'], ep['start']):.3f}, "
              f"goal_d={dist(cand['goal'], ep['goal']):.3f})")
        cases = {}
        for method in ("vanilla", "cbf_cd_cf", "bcbf_cf"):
            cases[method] = load_laser_trajectory(map_name, method, cand["seed"], cand["case_id"])
        cases["seanav"] = load_seanav_trajectory(ep["seed_dir"], difficulty, ep["env_id"], ep["episode_idx"])
        all_cases[map_name] = cases
        for method in METHOD_ORDER:
            xs, ys, outcome = cases[method]
            print(f"        {method:10s} outcome={outcome:10s} n_steps={len(xs)}")

    fig, axes = plt.subplots(1, len(MAP_ORDER), figsize=(4.0 * len(MAP_ORDER), 4.4))
    for idx, (ax, map_name) in enumerate(zip(axes, MAP_ORDER)):
        draw_panel(ax, map_name, COLUMN_TITLES[idx], all_cases[map_name])
        if idx == 0:
            ax.set_ylabel("y [m]", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle (circle or square)"),
    ] + [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.0, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ] + [
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10,
                   markeredgewidth=3.0, label="Collision"),
        plt.Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=10,
                   markerfacecolor="black", markeredgecolor="black", label="Timeout"),
        plt.Line2D([0], [0], marker="D", color="black", linestyle="None", markersize=9,
                   markerfacecolor="black", markeredgecolor="black", label="Other failure (fall/stuck)"),
        plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=14,
                   label="Goal reached"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.11), fontsize=11, columnspacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.22, wspace=0.08)

    output_base = Path("logs/evaluation/figures/seaspeed_combined_trajectories")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
