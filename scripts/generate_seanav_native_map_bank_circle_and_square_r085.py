#!/usr/bin/env python3
"""Regenerates the 3-map SeaNavMapBank (slalom=easy, corridor=medium,
circle_and_square=hard) with circle_and_square's MIDDLE-GROUND layout
(2026-09-24): center circle radius 1.00 -> 0.70 (targets the near-start gate
that dominated both bcbf_cf's and SEA-Nav's original-layout failures), but
BOTH small squares near the goal restored (unlike the fully-eased variant,
which removed them and over-corrected SEA-Nav's difficulty past even its
own Medium-Shift number) -- see evaluate_laser_unseen_maps_structured.py's
UNSEEN_MAPS["circle_and_square"] for the SAC-laser side of this same change.

Identical in every other respect to generate_seanav_native_map_bank_
circle_and_square_eased.py (same schema, same sampler/blocking math, same
slalom/corridor obstacle specs and RNG seeding).
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ARENA_WIDTH = 15.0
ARENA_HALF = ARENA_WIDTH / 2.0
WALL_CLEARANCE = 0.85
OBSTACLE_CLEARANCE = 0.30
MIN_ENDPOINT_CLEARANCE = 0.10
BLOCK_MARGIN = 0.10
BLOCKED_T_MIN, BLOCKED_T_MAX = 0.15, 0.85
SCENARIOS_PER_MAP = 200
OBSTACLE_HEIGHT = 1.0

MAPS = {
    "slalom": {
        "difficulty": "easy",
        "obstacles": [
            (-5.2, -2.3, 0.50), (-3.9, 2.1, 0.50), (-2.6, -2.2, 0.50),
            (-1.3, 2.2, 0.50), (0.0, -2.2, 0.50), (1.3, 2.2, 0.50),
            (2.6, -2.2, 0.50), (3.9, 2.1, 0.50), (5.2, -2.3, 0.50),
        ],
        "start": (-4.0, 4.0, 5.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    "corridor": {
        "difficulty": "medium",
        "obstacles": [
            (-3.8, -2.2, 0.50), (-1.9, -2.2, 0.50), (0.0, -2.2, 0.50),
            (1.9, -2.2, 0.50), (3.8, -2.2, 0.50), (-3.8, 2.2, 0.50),
            (-1.9, 2.2, 0.50), (1.9, 2.2, 0.50), (3.8, 2.2, 0.50),
        ],
        "start": (-2.0, 2.0, 4.0, 7.5),
        "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    # Middle-ground 2026-09-24: center circle 1.00 -> 0.70, both small
    # squares near the goal RESTORED (unlike the fully-eased variant).
    "circle_and_square": {
        "difficulty": "hard",
        "obstacles": [
            (-1.6, 3.4, 0.35, "circle"),
            (1.6, 3.4, 0.35, "circle"),
            (0.0, 1.9, 0.85, "circle"),
            (-1.5, -0.2, 0.35, "circle"),
            (1.5, -0.2, 0.35, "circle"),
            (-4.6, -2.3, 0.65, "circle"),
            (-1.0, -2.6, 0.40, "square"),
            (1.0, -2.6, 0.40, "square"),
            (4.6, -2.3, 0.65, "circle"),
            (-3.6, 0.4, 0.70, "square"),
            (3.6, 0.4, 0.70, "square"),
        ],
        "start": (-1.0, 1.0, 5.0, 6.8),
        "goal": (-5.5, 5.5, -7.0, -4.0),
    },
}


def clip_box(box):
    x0, x1, y0, y1 = box
    m = WALL_CLEARANCE
    return (
        max(x0, -ARENA_HALF + m), min(x1, ARENA_HALF - m),
        max(y0, -ARENA_HALF + m), min(y1, ARENA_HALF - m),
    )


def total_clearance(p, obstacles):
    ox, oy = p
    best = min(
        ox - (-ARENA_HALF + WALL_CLEARANCE), (ARENA_HALF - WALL_CLEARANCE) - ox,
        oy - (-ARENA_HALF + WALL_CLEARANCE), (ARENA_HALF - WALL_CLEARANCE) - oy,
    )
    for obstacle in obstacles:
        cx, cy, r = obstacle[0], obstacle[1], obstacle[2]
        best = min(best, math.hypot(ox - cx, oy - cy) - r - OBSTACLE_CLEARANCE)
    return best


def blocking_info(s, g, obstacles):
    seg = (g[0] - s[0], g[1] - s[1])
    seg_len_sq = seg[0] ** 2 + seg[1] ** 2
    if seg_len_sq < 1e-12:
        return None
    best = None
    for obstacle in obstacles:
        cx, cy, r = obstacle[0], obstacle[1], obstacle[2]
        t = ((cx - s[0]) * seg[0] + (cy - s[1]) * seg[1]) / seg_len_sq
        if t < BLOCKED_T_MIN or t > BLOCKED_T_MAX:
            continue
        closest = (s[0] + t * seg[0], s[1] + t * seg[1])
        dist = math.hypot(cx - closest[0], cy - closest[1])
        threshold = r + OBSTACLE_CLEARANCE + BLOCK_MARGIN
        clearance = dist - r - OBSTACLE_CLEARANCE
        blocked = dist <= threshold
        if best is None or (blocked and not best["blocked"]) or (blocked == best["blocked"] and clearance < best["clearance"]):
            best = {"blocked": blocked, "clearance": clearance}
    return best


def sample_scenarios(map_id: int, name: str, spec: dict, n: int, seed: int):
    rng = random.Random(seed)
    start_box = clip_box(spec["start"])
    goal_box = clip_box(spec["goal"])
    obstacles = spec["obstacles"]
    rows = []
    attempts = 0
    max_attempts = 200_000
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        s = (rng.uniform(start_box[0], start_box[1]), rng.uniform(start_box[2], start_box[3]))
        g = (rng.uniform(goal_box[0], goal_box[1]), rng.uniform(goal_box[2], goal_box[3]))
        if total_clearance(s, obstacles) < MIN_ENDPOINT_CLEARANCE:
            continue
        if total_clearance(g, obstacles) < MIN_ENDPOINT_CLEARANCE:
            continue
        if math.hypot(g[0] - s[0], g[1] - s[1]) <= 1.0:
            continue
        info = blocking_info(s, g, obstacles)
        if info is None or not info["blocked"]:
            continue
        yaw = math.atan2(g[1] - s[1], g[0] - s[0])
        rows.append({
            "scenario_id": len(rows), "map_id": map_id,
            "start_x": round(s[0], 4), "start_y": round(s[1], 4),
            "goal_x": round(g[0], 4), "goal_y": round(g[1], 4),
            "initial_yaw_rad": round(yaw, 6),
        })
    if len(rows) < n:
        raise RuntimeError(f"Only sampled {len(rows)}/{n} valid scenarios for {name!r} after {attempts} attempts.")
    return rows


def main() -> None:
    out_root = Path("generated/procedural_maps/seanav_native_circle_and_square_r085")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    manifest_maps = []
    all_scenarios = []
    for map_id, (name, spec) in enumerate(MAPS.items()):
        manifest_maps.append({"map_id": map_id, "difficulty": spec["difficulty"], "file": f"maps/map_{map_id:03d}.json"})
        map_json = {
            "map_id": map_id,
            "name": name,
            "difficulty": spec["difficulty"],
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {
                    "x": obstacle[0], "y": obstacle[1], "radius": obstacle[2],
                    "height": OBSTACLE_HEIGHT,
                    "shape": obstacle[3] if len(obstacle) > 3 else "circle",
                }
                for obstacle in spec["obstacles"]
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        all_scenarios.extend(sample_scenarios(map_id, name, spec, SCENARIOS_PER_MAP, seed=1000 + map_id))
        print(f"{name} (map_id={map_id}, {spec['difficulty']}): {SCENARIOS_PER_MAP} scenarios OK")

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_native_map_bank_circle_and_square_r085.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": SCENARIOS_PER_MAP,
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"\nWrote bank to {out_root}/")


if __name__ == "__main__":
    main()
