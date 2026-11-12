#!/usr/bin/env python3
"""SeaNavMapBank carrying the SAME 20 shared (start, goal, yaw) scenarios
used for bcbf_cf/cbf_cd_cf (evaluation_seed=1000, case_id 0-19) and CBF-RL
(scripts/eval_cbf_rl_v5_shared_scenarios.py) -- part of the 4-method 8-sample
trajectory comparison figure requested 2026-09-24. Unlike
generate_seanav_v5_fixed_scenario_bank.py (one point repeated), each
scenario_id here is a DISTINCT case, matching case_id in the SAC-laser
manifest 1:1.

Obstacle layout copied verbatim from
scripts/generate_seanav_native_map_bank_corridor_derived_highshift_v5.py's
"corridor_derived_highshift_v5" entry. The same physical map is written
under all 3 difficulty labels (terrain.py's SeaNavMapBank needs a fixed
3-row grid) -- only the "hard" tier's results are used.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ARENA_WIDTH = 15.0
OBSTACLE_HEIGHT = 1.0

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

SHARED_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/"
    "seed_1000/corridor_derived_highshift_v5/scenarios.csv"
)
NUM_SHARED_CASES = 20


def load_shared_scenarios() -> list[dict[str, float]]:
    with SHARED_MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if int(r["case_id"]) < NUM_SHARED_CASES]
    rows.sort(key=lambda r: int(r["case_id"]))
    assert len(rows) == NUM_SHARED_CASES, f"expected {NUM_SHARED_CASES} rows, got {len(rows)}"
    return [
        {"case_id": int(r["case_id"]), "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
         "start_yaw": float(r["start_yaw"]), "goal_x": float(r["goal_x"]), "goal_y": float(r["goal_y"])}
        for r in rows
    ]


def main() -> None:
    shared = load_shared_scenarios()
    out_root = Path("generated/procedural_maps/seanav_corridor_derived_highshift_v5_shared_scenarios")
    (out_root / "maps").mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    manifest_maps = []
    all_scenarios = []
    for map_id, difficulty in enumerate(difficulties):
        map_json = {
            "map_id": map_id,
            "name": "corridor_derived_highshift_v5",
            "difficulty": difficulty,
            "arena": {"width": ARENA_WIDTH, "length": ARENA_WIDTH},
            "obstacles": [
                {"x": x, "y": y, "radius": r, "height": OBSTACLE_HEIGHT, "shape": shape}
                for (x, y, r, shape) in OBSTACLES
            ],
        }
        (out_root / "maps" / f"map_{map_id:03d}.json").write_text(json.dumps(map_json, indent=2))
        manifest_maps.append({"map_id": map_id, "difficulty": difficulty, "file": f"maps/map_{map_id:03d}.json"})
        for s in shared:
            all_scenarios.append({
                "scenario_id": s["case_id"], "map_id": map_id,
                "start_x": s["start_x"], "start_y": s["start_y"],
                "goal_x": s["goal_x"], "goal_y": s["goal_y"],
                "initial_yaw_rad": s["start_yaw"],
            })

    (out_root / "bank.json").write_text(json.dumps({
        "schema_version": 1,
        "generator": "generate_seanav_v5_shared_scenarios_bank.py",
        "num_maps": len(manifest_maps),
        "scenarios_per_map": len(shared),
        "maps": manifest_maps,
    }, indent=2))

    fieldnames = ["scenario_id", "map_id", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw_rad"]
    with open(out_root / "scenarios.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_scenarios)

    print(f"Wrote bank to {out_root}/ ({len(shared)} shared scenarios x 3 difficulty labels)")


if __name__ == "__main__":
    main()
