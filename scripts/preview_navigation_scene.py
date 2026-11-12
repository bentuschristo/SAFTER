"""Open a generated MJCF scene with the Go2 and arena, without training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import tyro


@dataclass
class PreviewConfig:
    scene_xml: str
    duration_s: float = 0.0
    step_physics: bool = False


def main():
    cfg = tyro.cli(PreviewConfig)
    scene = Path(cfg.scene_xml).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    print(f"[PREVIEW] Loaded {scene}")
    print("[PREVIEW] Close the viewer window or press Ctrl+C to exit.")
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if cfg.step_physics:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
            if cfg.duration_s > 0 and time.time() - start >= cfg.duration_s:
                break


if __name__ == "__main__":
    main()
