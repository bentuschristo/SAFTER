#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_TASK_ID = "Unitree-Go2-Safe-Navigation"
DEFAULT_LOW_LEVEL = "~/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"
DEFAULT_NO_CBF = "~/unitree_rl_mjlab/logs/sac/go2_safe_navigation_sac_cbf_cf/2026-07-01_16-10-51_No-CBF-Env256-Step200000-Timeout5-UnifiedCollision/model_step_200000.pt"
DEFAULT_CBF = "~/unitree_rl_mjlab/logs/sac/go2_safe_navigation_sac_cbf_cf/2026-07-01_19-11-12_CBF-CF-Env256-Step200000-Timeout5-UnifiedCollision/model_step_200000.pt"
DEFAULT_BCBF = "~/unitree_rl_mjlab/logs/sac/go2_safe_navigation_sac_bcbf_cf/2026-07-02_15-58-26_BCBF-CF-v4-HOCBF-VelocityTerminal-Env256-Step200000-Timeout5/model_step_200000.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Run original and unseen SAC evaluations sequentially.")
    p.add_argument("--repo", default="~/unitree_rl_mjlab")
    p.add_argument("--task-id", default=DEFAULT_TASK_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-duration-s", type=float, default=90.0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--original-cases", type=int, default=100)
    p.add_argument("--cases-per-map", type=int, default=20)
    p.add_argument("--maps", nargs="+", default=["shifted", "corridor", "clustered", "slalom", "bottleneck"])
    p.add_argument("--low-level-checkpoint", default=DEFAULT_LOW_LEVEL)
    p.add_argument("--no-cbf-checkpoint", default=DEFAULT_NO_CBF)
    p.add_argument("--cbf-checkpoint", default=DEFAULT_CBF)
    p.add_argument("--bcbf-checkpoint", default=DEFAULT_BCBF)
    p.add_argument("--output-root", default="logs/evaluation")
    p.add_argument("--terminal-log-root", default="logs/evaluation_terminal")
    p.add_argument("--skip-original", action="store_true")
    p.add_argument("--skip-unseen", action="store_true")
    p.add_argument("--log-trajectories", action="store_true")
    p.add_argument("--no-overwrite-manifest", action="store_true")
    return p.parse_args()


def resolve(value: str, repo: Path | None = None) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute() and repo is not None:
        p = repo / p
    return p.resolve()


def run_and_tee(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(shlex.quote(x) for x in cmd))
    print(f"Log: {log_path}")
    print("=" * 100, flush=True)

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = proc.wait()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> int:
    a = parse_args()
    repo = resolve(a.repo)
    if not repo.is_dir():
        print(f"[ERROR] Repository not found: {repo}", file=sys.stderr)
        return 2

    original_script = repo / "scripts/evaluate_random_manifest_all_methods.py"
    unseen_script = repo / "scripts/evaluate_unseen_maps_all_methods.py"

    required = {
        "Original evaluator": original_script,
        "Unseen evaluator": unseen_script,
        "Low-level checkpoint": resolve(a.low_level_checkpoint),
        "No-CBF checkpoint": resolve(a.no_cbf_checkpoint),
        "CBF checkpoint": resolve(a.cbf_checkpoint),
        "bCBF checkpoint": resolve(a.bcbf_checkpoint),
    }
    for label, path in required.items():
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 2

    output_root = resolve(a.output_root, repo)
    log_root = resolve(a.terminal_log_root, repo)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common = [
        "--seed", str(a.seed),
        "--episode-duration-s", str(a.episode_duration_s),
        "--print-every", str(a.print_every),
        "--device", a.device,
        "--low-level-checkpoint", str(required["Low-level checkpoint"]),
        "--no-cbf-checkpoint", str(required["No-CBF checkpoint"]),
        "--cbf-checkpoint", str(required["CBF checkpoint"]),
        "--bcbf-checkpoint", str(required["bCBF checkpoint"]),
        "--methods", "no_cbf", "cbf_rl", "bcbf_rl",
    ]
    overwrite = [] if a.no_overwrite_manifest else ["--overwrite-manifest"]

    try:
        if not a.skip_original:
            run_and_tee(
                [
                    sys.executable,
                    str(original_script),
                    a.task_id,
                    "--cases", str(a.original_cases),
                    *common,
                    "--output-root", str(output_root / f"original_map_100ep_v4_{stamp}"),
                    *overwrite,
                ],
                repo,
                log_root / f"original_map_100ep_v4_{stamp}.log",
            )

        if not a.skip_unseen:
            cmd = [
                sys.executable,
                str(unseen_script),
                a.task_id,
                "--cases-per-map", str(a.cases_per_map),
                "--maps", *a.maps,
                *common,
                "--output-root", str(output_root / f"unseen_maps_100ep_v4_{stamp}"),
                *overwrite,
            ]
            if a.log_trajectories:
                cmd.append("--log-trajectories")
            run_and_tee(
                cmd,
                repo,
                log_root / f"unseen_maps_100ep_v4_{stamp}.log",
            )

    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Evaluation failed with return code {exc.returncode}.", file=sys.stderr)
        print("The next evaluation was not started.", file=sys.stderr)
        return exc.returncode

    print("\nAll requested evaluations completed successfully.")
    print(f"Outputs: {output_root}")
    print(f"Logs:    {log_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
