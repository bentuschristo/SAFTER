#!/usr/bin/env python3
"""Play one timeout-90 policy in the original scene with safe random resets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_script(repo: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = repo / "scripts" / name
        if path.exists():
            return path
    raise FileNotFoundError("None found: " + ", ".join(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("no_cbf", "cbf_rl", "bcbf_rl"))
    parser.add_argument("--repo", type=Path, default=Path.home() / "unitree_rl_mjlab")
    parser.add_argument("--task", default="Unitree-Go2-Safe-Navigation")
    parser.add_argument("--low-level-checkpoint", required=True)
    parser.add_argument("--high-level-checkpoint", required=True)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="native")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if args.method == "bcbf_rl":
        script = find_script(repo, ("play_high_level_sac_bcbf_cf.py",))
        method_flags: list[str] = []
    else:
        script = find_script(repo, ("play_high_level_sac_cbf_cf.py", "play_high_level_sac.py"))
        enabled = "True" if args.method == "cbf_rl" else "False"
        method_flags = [f"--deployment-safety-filter={enabled}"]

    command = [
        sys.executable, str(script), args.task,
        f"--low-level-checkpoint={args.low_level_checkpoint}",
        f"--high-level-checkpoint={args.high_level_checkpoint}",
        "--num-envs=1",
        f"--device={args.device}",
        f"--viewer={args.viewer}",
        f"--print-every={args.print_every}",
        "--config.navigation.episode-duration-s=90",
        "--config.reward.timeout-penalty=10",
        *method_flags,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
