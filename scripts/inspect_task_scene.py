"""Print every XML/MJCF-looking path inside a registered task configuration."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import tyro

from mjlab.tasks.registry import list_tasks, load_env_cfg


def walk(obj, prefix="cfg", seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (str, Path)):
        value = str(obj)
        if value.lower().endswith((".xml", ".mjcf")):
            print(f"{prefix} = {value}")
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            walk(getattr(obj, f.name), f"{prefix}.{f.name}", seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{prefix}[{k!r}]", seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]", seen)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("_"):
                walk(v, f"{prefix}.{k}", seen)


def main():
    import mjlab.tasks  # noqa
    import src.tasks  # noqa
    task, _ = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = load_env_cfg(task, play=True)
    walk(cfg)


if __name__ == "__main__":
    main()
