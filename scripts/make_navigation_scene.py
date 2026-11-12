"""Generate the 15x15 m arena and patch a copy of an existing Go2 scene MJCF."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from src.high_level_rl.arena import arena_to_json, patch_scene_with_arena, write_arena_fragment
from src.high_level_rl.config import ArenaConfig


@dataclass
class MakeSceneConfig:
    base_scene_xml: str
    output_dir: str = "generated/navigation_scene"
    output_scene_name: str = "scene_go2_navigation.xml"
    arena: ArenaConfig = field(default_factory=ArenaConfig)


def main():
    cfg = tyro.cli(MakeSceneConfig)
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fragment = write_arena_fragment(cfg.arena, out / "navigation_arena.xml")
    scene = patch_scene_with_arena(cfg.base_scene_xml, fragment, out / cfg.output_scene_name)
    arena_json = arena_to_json(cfg.arena, out / "arena.json")
    print(f"[SCENE] Arena fragment: {fragment}")
    print(f"[SCENE] Patched Go2 scene: {scene}")
    print(f"[SCENE] Arena configuration: {arena_json}")
    print("\nPoint the Go2 task's scene/XML asset path to the patched scene before training or play.")


if __name__ == "__main__":
    main()
