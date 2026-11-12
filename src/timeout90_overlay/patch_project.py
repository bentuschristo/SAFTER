from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TARGETS = (
    Path("src/high_level_rl/config.py"),
    Path("src/high_level_rl/navigation_env.py"),
    Path("src/high_level_sac/config.py"),
    Path("src/high_level_sac/sac_navigation_env.py"),
    Path("src/high_level_bcbf_sac/config.py"),
    Path("src/high_level_bcbf_sac/sac_navigation_env.py"),
)


def backup(repo: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = repo / "backups" / f"before_timeout90_{stamp}"
    for rel in TARGETS:
        src = repo / rel
        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    restore = out / "RESTORE.sh"
    restore.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO="${1:-$HOME/unitree_rl_mjlab}"\n'
        f'BACKUP="{out}"\n'
        'cd "$REPO"\n'
        'cp -a "$BACKUP/src/." src/\n'
        'echo "Restored source files from: $BACKUP"\n'
    )
    restore.chmod(0o755)
    return out


def set_episode_duration(path: Path) -> None:
    text = path.read_text()
    updated, count = re.subn(
        r"^(\s*)episode_duration_s:\s*float\s*=\s*[^\n]+$",
        r"\1episode_duration_s: float = 90.0",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"{path}: could not uniquely set episode_duration_s")
    path.write_text(updated)


def add_timeout_field(path: Path) -> None:
    text = path.read_text()
    if re.search(r"^\s*timeout_penalty:\s*float\s*=", text, re.MULTILINE):
        return

    pattern = r"^(\s*)time_penalty:\s*float\s*=\s*[^\n]+$"
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{path}: time_penalty field not found")

    indent = match.group(1)
    replacement = match.group(0) + f"\n{indent}timeout_penalty: float = 10.0"
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"{path}: failed to add timeout_penalty")
    path.write_text(updated)


def add_timeout_component(path: Path) -> None:
    text = path.read_text()
    if '"timeout"' in text and "timeout_penalty" in text:
        return

    patterns = (
        r'(?P<indent>\s*)"time":\s*torch\.full_like\(\s*progress,\s*-r\.time_penalty\s*\),',
        r'(?P<indent>\s*)"time":\s*torch\.full_like\(\s*progress,\s*-self\.sac_reward_cfg\.time_penalty\s*\),',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            insert = match.group(0) + "\n" + match.group("indent") + '"timeout": -r.timeout_penalty * timeout.float(),'
            if "self.sac_reward_cfg" in match.group(0):
                insert = match.group(0) + "\n" + match.group("indent") + '"timeout": -self.sac_reward_cfg.timeout_penalty * timeout.float(),'
            path.write_text(text[:match.start()] + insert + text[match.end():])
            return

    if "timeout_penalty * timeout.float()" in text:
        return

    for pattern, line in (
        (r"(?P<indent>\s*)-\s*r\.time_penalty\s*$", "- r.timeout_penalty * timeout.float()"),
        (r"(?P<indent>\s*)-\s*self\.reward_cfg\.time_penalty\s*$", "- self.reward_cfg.timeout_penalty * timeout.float()"),
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            insert = match.group(0) + "\n" + match.group("indent") + line
            path.write_text(text[:match.start()] + insert + text[match.end():])
            return

    raise RuntimeError(f"{path}: reward expression format was not recognized")


def verify_original_scene(repo: Path) -> None:
    cfg = (repo / "src/high_level_rl/config.py").read_text()
    required = (
        "-3.0", "-2.0", "-1.5", "2.2", "1.2", "-1.0", "3.0", "2.3",
        "0.3", "3.8", "-5.0", "4.5", "-4.5", "5.0",
    )
    missing = [item for item in required if item not in cfg]
    if missing:
        raise RuntimeError(f"Original nine-obstacle scene not verified; missing: {missing}")

    nav = (repo / "src/high_level_rl/navigation_env.py").read_text()
    required_sampling = (
        "minimum_start_goal_distance", "spawn_extra_clearance",
        "sample_margin_from_walls", "_sample_free_positions",
    )
    combined = cfg + nav
    missing_sampling = [item for item in required_sampling if item not in combined]
    if missing_sampling:
        raise RuntimeError(f"Safe reset sampling not verified: {missing_sampling}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "unitree_rl_mjlab")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    verify_original_scene(repo)
    backup_dir = backup(repo)
    try:
        set_episode_duration(repo / "src/high_level_rl/config.py")
        for rel in (
            Path("src/high_level_rl/config.py"),
            Path("src/high_level_sac/config.py"),
            Path("src/high_level_bcbf_sac/config.py"),
        ):
            add_timeout_field(repo / rel)
        for rel in (
            Path("src/high_level_rl/navigation_env.py"),
            Path("src/high_level_sac/sac_navigation_env.py"),
            Path("src/high_level_bcbf_sac/sac_navigation_env.py"),
        ):
            add_timeout_component(repo / rel)
        subprocess.run([sys.executable, "-m", "py_compile", *[str(repo / p) for p in TARGETS]], check=True)
    except Exception:
        print(f"Patch failed. Backup remains at: {backup_dir}", file=sys.stderr)
        raise

    print("Timeout-90 overlay applied successfully.")
    print("  episode_duration_s = 90.0")
    print("  timeout_penalty    = 10.0")
    print("  scene              = original static nine-obstacle scene")
    print("  reset              = randomized safe start/goal/yaw")
    print(f"  backup             = {backup_dir}")
    print(f"  restore            = {backup_dir / 'RESTORE.sh'}")


if __name__ == "__main__":
    main()
