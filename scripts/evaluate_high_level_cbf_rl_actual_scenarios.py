"""Evaluate a CBF-RL PPO checkpoint on the LITERAL 200 curated
ProceduralScenario start/goal pairs per handcrafted2 map (not freshly
resampled region-box cases), for filter ON and OFF, aggregated per map and
pooled.

This isolates whether the low pooled success rate seen in
evaluate_high_level_cbf_rl_training_maps.py (~37%) is explained by
eval-case-difficulty mismatch versus the actual bank scenarios the policy
was trained on, by scoring the checkpoint on those exact 1000 scenarios
(200/map x 5 maps) via env.set_fixed_scenarios(...) -- the same mechanism
evaluate_high_level_cbf_rl_unseen_maps.py uses for its fixed obstacle
layouts.

Example:
  python scripts/evaluate_high_level_cbf_rl_actual_scenarios.py \\
      --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-17_11-54-21_dual_handcrafted2_seed0_4096envs_1000iters/model_999.pt \\
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
      --device cuda:0
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.utils.torch import configure_torch_backends

from SAFTER_src.high_level_laser_sac.procedural_maps import load_procedural_map_bank

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_high_level_cbf_rl_unseen_maps import evaluate_map, load_policy  # noqa: E402
from evaluate_high_level_cbf_rl_training_maps import EvalTrainingMapsConfig, build_training_map_env  # noqa: E402
from src.high_level_cbf_rl.config import CbfRlConfig

TASK_ID = "Unitree-Go2-Safe-Navigation"


@dataclass
class Config:
    checkpoint: str
    low_level_checkpoint: str
    procedural_map_bank: str = "generated/procedural_maps/handcrafted2"
    expected_num_maps: int = 5
    expected_scenarios_per_map: int = 200
    expected_max_obstacles: int = 12
    num_envs: int = 50
    episode_duration_s: float | None = None
    seed: int = 1000
    stochastic_policy: bool = False
    training_low_level: bool = True
    device: str = "cuda:0"
    laser_num_rays: int = 120
    laser_fov_deg: float = 360.0
    enable_lidar_noise: bool = True
    output_dir: str = "logs/evaluation/cbf_rl_actual_training_scenarios"
    run_name: str = ""
    config: CbfRlConfig = field(default_factory=CbfRlConfig)


def build_manifest_from_scenarios(map_obj, map_name: str) -> list[dict[str, object]]:
    return [
        {
            "map_name": map_name,
            "case_id": s.scenario_id,
            "reset_seed": s.scenario_id,
            "start_x": float(s.start_xy[0]),
            "start_y": float(s.start_xy[1]),
            "start_yaw": float(s.start_yaw),
            "goal_x": float(s.goal_xy[0]),
            "goal_y": float(s.goal_xy[1]),
        }
        for s in map_obj.scenarios
    ]


def run(cfg: Config) -> None:
    configure_torch_backends()
    device = cfg.device

    bank = load_procedural_map_bank(
        cfg.procedural_map_bank,
        expected_num_maps=cfg.expected_num_maps,
        expected_scenarios_per_map=cfg.expected_scenarios_per_map,
        expected_max_obstacles=cfg.expected_max_obstacles,
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    out_dir = Path(cfg.output_dir) / f"{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    all_summary_rows: list[dict[str, object]] = []
    for filter_active in (True, False):
        condition = "filter_on" if filter_active else "filter_off"
        pooled_records: list[dict[str, object]] = []
        for map_index, map_obj in enumerate(bank.maps):
            map_name = f"train_map_{map_index}"
            print(f"\n[EVAL:{condition}] Building env for {map_name}")
            eval_cfg = EvalTrainingMapsConfig(
                checkpoint=cfg.checkpoint,
                low_level_checkpoint=cfg.low_level_checkpoint,
                procedural_map_bank=cfg.procedural_map_bank,
                expected_num_maps=cfg.expected_num_maps,
                expected_scenarios_per_map=cfg.expected_scenarios_per_map,
                expected_max_obstacles=cfg.expected_max_obstacles,
                num_envs=cfg.num_envs,
                # build_training_map_env sizes env_cfg.scene.num_envs as
                # min(num_envs, cases_per_seed) -- pin cases_per_seed to
                # num_envs so the env's actual env count matches the batch
                # size evaluate_map() computes from our 200-scenario manifest
                # (min(num_envs, len(manifest))), not the unrelated
                # 20-per-seed convention this field name implies elsewhere.
                cases_per_seed=cfg.num_envs,
                episode_duration_s=cfg.episode_duration_s,
                seed=cfg.seed + map_index,
                stochastic_policy=cfg.stochastic_policy,
                training_low_level=cfg.training_low_level,
                config=cfg.config,
                device=device,
                use_cbf_action_filtering=filter_active,
                laser_num_rays=cfg.laser_num_rays,
                laser_fov_deg=cfg.laser_fov_deg,
                enable_lidar_noise=cfg.enable_lidar_noise,
            )
            env = build_training_map_env(TASK_ID, map_obj, eval_cfg, device)
            policy = load_policy(env, TASK_ID, cfg.checkpoint, device)
            manifest = build_manifest_from_scenarios(map_obj, map_name)
            # Keep the same reset seeds in both filter conditions.
            for case in manifest:
                case["reset_seed"] += cfg.seed
            records = evaluate_map(
                env=env, policy=policy, manifest=manifest, num_envs=cfg.num_envs,
                print_every=5, stochastic_policy=cfg.stochastic_policy,
            )
            env.close()

            n = len(records)
            success = sum(r["success"] for r in records)
            collision = sum(r["collision"] for r in records)
            timeout = sum(r["timeout"] for r in records)
            fallen = sum(r["fallen"] for r in records)
            print(f"[EVAL:{condition}] {map_name}: n={n} success={success}/{n} "
                  f"({100.0*success/n:.1f}%) collision={100.0*collision/n:.1f}% "
                  f"timeout={100.0*timeout/n:.1f}% fall={100.0*fallen/n:.1f}%")

            with (out_dir / f"{condition}_{map_name}_episodes.csv").open("w", newline="") as file:
                if records:
                    writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)

            all_summary_rows.append({
                "condition": condition, "map_name": map_name, "cases": n,
                "success_rate": success / n if n else float("nan"),
                "collision_rate": collision / n if n else float("nan"),
                "timeout_rate": timeout / n if n else float("nan"),
                "fall_rate": fallen / n if n else float("nan"),
            })
            pooled_records.extend(records)

        n = len(pooled_records)
        success = sum(r["success"] for r in pooled_records)
        collision = sum(r["collision"] for r in pooled_records)
        timeout = sum(r["timeout"] for r in pooled_records)
        fallen = sum(r["fallen"] for r in pooled_records)
        print(f"\n[EVAL:{condition}] POOLED (all 5 maps, n={n}): success={100.0*success/n:.1f}% "
              f"collision={100.0*collision/n:.1f}% timeout={100.0*timeout/n:.1f}% fall={100.0*fallen/n:.1f}%")
        all_summary_rows.append({
            "condition": condition, "map_name": "POOLED", "cases": n,
            "success_rate": success / n if n else float("nan"),
            "collision_rate": collision / n if n else float("nan"),
            "timeout_rate": timeout / n if n else float("nan"),
            "fall_rate": fallen / n if n else float("nan"),
        })

    with (out_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"\n[EVAL] Wrote results to {out_dir}")


def main() -> None:
    cfg = tyro.cli(Config)
    run(cfg)


if __name__ == "__main__":
    main()
