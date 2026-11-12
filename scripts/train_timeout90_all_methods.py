#!/usr/bin/env python3
"""Train no-CBF, CBF-RL, and bCBF-RL sequentially on the original scene."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_script(repo: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = repo / "scripts" / name
        if path.exists():
            return path
    raise FileNotFoundError("None found: " + ", ".join(candidates))


def run_logged(command: list[str], log_file: Path, env: dict[str, str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 78)
    print(" ".join(command))
    print(f"Log: {log_file}")
    print("=" * 78, flush=True)
    with log_file.open("w") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "unitree_rl_mjlab")
    parser.add_argument("--task", default="Unitree-Go2-Safe-Navigation")
    parser.add_argument("--low-level-checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--vector-steps", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=("no_cbf", "cbf_rl", "bcbf_rl"), default=("no_cbf", "cbf_rl", "bcbf_rl"))
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    low = Path(args.low_level_checkpoint).expanduser()
    if not low.is_absolute():
        low = repo / low
    if not low.exists():
        raise FileNotFoundError(f"Low-level checkpoint not found: {low}")

    standard = find_script(repo, ("train_high_level_sac_cbf_cf.py", "train_high_level_sac.py"))
    backup = find_script(repo, ("train_high_level_sac_bcbf_cf.py",))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = repo / "logs" / "training_batches" / f"timeout90_{stamp}"
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    common = [
        args.task,
        f"--low-level-checkpoint={low}",
        f"--num-envs={args.num_envs}",
        f"--device={args.device}",
        f"--vector-steps={args.vector_steps}",
        f"--save-interval={args.save_interval}",
        f"--log-interval={args.log_interval}",
        f"--seed={args.seed}",
        "--config.navigation.episode-duration-s=90",
        "--config.reward.timeout-penalty=10",
    ]

    for method in args.methods:
        if method == "no_cbf":
            script = standard
            extra = ["--config.use-safety-filter=False", "--experiment-name=go2_safe_navigation_sac_timeout90_no_cbf", "--run-name=original_scene_timeout10_duration90"]
        elif method == "cbf_rl":
            script = standard
            extra = ["--config.use-safety-filter=True", "--experiment-name=go2_safe_navigation_sac_timeout90_cbf", "--run-name=original_scene_timeout10_duration90"]
        else:
            script = backup
            extra = ["--experiment-name=go2_safe_navigation_sac_timeout90_bcbf", "--run-name=original_scene_timeout10_duration90"]
        run_logged([sys.executable, str(script), *common, *extra], log_dir / f"{method}.log", env)

    print(f"\nAll requested runs completed. Logs: {log_dir}")


if __name__ == "__main__":
    main()
