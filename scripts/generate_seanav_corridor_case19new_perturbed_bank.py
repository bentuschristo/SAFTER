#!/usr/bin/env python3
"""4 jittered (~0.45m) variants of the collision-search pool's case_id=19
(start=(0.837,6.778), goal=(1.353,-5.639)) -- its only observed genuine
collision draw was a 27-step near-instant one right at spawn; perturbing
searches for a richer collision, mirroring
scripts/generate_seanav_corridor_stuck_perturbed_bank.py.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0
OBSTACLES = [
    (-3.8, -2.2, 0.50, "circle"), (-1.9, -2.2, 0.50, "circle"), (0.0, -2.2, 0.50, "circle"),
    (1.9, -2.2, 0.50, "circle"), (3.8, -2.2, 0.50, "circle"), (-3.8, 2.2, 0.50, "circle"),
    (-1.9, 2.2, 0.50, "circle"), (1.9, 2.2, 0.50, "circle"), (3.8, 2.2, 0.50, "circle"),
]
START_REGION = (-2.0, 2.0, 4.0, 7.5)
BASE_START = (0.8373592495918274, 6.778175354003906)
GOAL = (1.3530491590499878, -5.639278888702393)
N_VARIANTS = 4


def jitter(xy, radius, seed):
    rng = random.Random(seed)
    r = radius * rng.random() ** 0.5
    theta = rng.uniform(0.0, 2.0 * math.pi)
    x = min(max(xy[0] + r * math.cos(theta), START_REGION[0]), START_REGION[1])
    y = min(max(xy[1] + r * math.sin(theta), START_REGION[2]), START_REGION[3])
    return x, y


def main() -> None:
    scenarios = []
    for i in range(N_VARIANTS):
        jx, jy = jitter(BASE_START, 0.45, seed=1900 + i)
        yaw = math.atan2(GOAL[1] - jy, GOAL[0] - jx)
        scenarios.append({"scenario_id": i, "start_x": jx, "start_y": jy,
                           "goal_x": GOAL[0], "goal_y": GOAL[1], "initial_yaw_rad": yaw})

    out_root = Path("generated/procedural_maps/seanav_corridor_case19new_perturbed")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)
    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    all_scenarios = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id, "name": "corridor", "difficulty": difficulty,
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [{"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT, "shape": shape}
                          for (x, y, r, shape) in OBSTACLES],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        manifest_maps.append({"map_id": map_id, "difficulty": difficulty, "file": f"maps/map_{map_id:03d}.json"})
        for s in scenarios:
            all_scenarios.append({**s, "map_id": map_id})

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1, "generator": "generate_seanav_corridor_case19new_perturbed_bank.py",
        "num_maps": len(manifest_maps), "scenarios_per_map": len(scenarios), "maps": manifest_maps,
    }, indent=2))
    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)
    print(f"Wrote bank to {out_root}/")


if __name__ == "__main__":
    main()
