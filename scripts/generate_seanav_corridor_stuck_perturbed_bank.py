#!/usr/bin/env python3
"""SeaNavMapBank with small (~0.45m) start-position jitter around corridor's
shared case_id 2, 8, 18, 19 -- these 4 scenarios never produced a genuine
collision or timeout across ~90 draws each of the EXACT shared start/goal
(scripts/generate_seanav_corridor_shared_scenarios_bank.py's data), only
"success" or the "stuck" mechanism (other_failure). Requested 2026-09-24:
check whether a small perturbation of the start surfaces a genuine collision
instead, the same way scripts/eval_cbf_rl_corridor_two_case_repeated.py did
for CBF-RL's two stuck-in-start-region cases.

4 jittered variants per case (radius=0.45m, deterministic), written as 16
distinct scenario_ids under one map_id per difficulty label (medium is the
one actually used).
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
CASES = {
    2: (-1.2677196264266968, 5.018768787384033, -2.9440345764160156, -5.258774757385254),
    8: (-1.9029620885849, 4.415013790130615, -2.034708261489868, -4.278390407562256),
    18: (-1.7752678394317627, 4.121930122375488, -0.8131601214408875, -4.0915045738220215),
    19: (-0.3018221855163574, 4.398306369781494, 3.732717752456665, -5.149234294891357),
}
JITTER_RADIUS_M = 0.45
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
    scenario_id = 0
    case_of_scenario = {}
    for case_id, (sx, sy, gx, gy) in CASES.items():
        for i in range(N_VARIANTS):
            jx, jy = jitter((sx, sy), JITTER_RADIUS_M, seed=case_id * 100 + i)
            yaw = math.atan2(gy - jy, gx - jx)
            scenarios.append({"scenario_id": scenario_id, "start_x": jx, "start_y": jy,
                               "goal_x": gx, "goal_y": gy, "initial_yaw_rad": yaw})
            case_of_scenario[scenario_id] = case_id
            scenario_id += 1

    out_root = Path("generated/procedural_maps/seanav_corridor_stuck_perturbed")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    all_scenarios = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id, "name": "corridor", "difficulty": difficulty,
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT, "shape": shape}
                for (x, y, r, shape) in OBSTACLES
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        manifest_maps.append({"map_id": map_id, "difficulty": difficulty, "file": f"maps/map_{map_id:03d}.json"})
        for s in scenarios:
            all_scenarios.append({**s, "map_id": map_id})

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_corridor_stuck_perturbed_bank.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": len(scenarios),
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"Wrote bank to {out_root}/ ({len(scenarios)} jittered scenarios x 3 difficulty labels)")
    for sid, cid in case_of_scenario.items():
        s = scenarios[sid]
        print(f"  scenario_id={sid} case={cid} start=({s['start_x']:.3f},{s['start_y']:.3f})")


if __name__ == "__main__":
    main()
