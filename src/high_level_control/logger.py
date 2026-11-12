from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STANDARD_FIELDS = [
    "time_s",
    "mode",
    "trajectory",
    "x", "y", "yaw",
    "x_ref", "y_ref", "yaw_ref",
    "vx_meas", "vy_meas", "wz_meas",
    "vx_nom", "vy_nom", "wz_nom",
    "vx_cmd", "vy_cmd", "wz_cmd",
    "ex", "ey", "eyaw",
    "position_error_norm",
    "velocity_tracking_error_norm",
    "h",
    "cbf_value_nominal",
    "cbf_active",
    "cbf_correction_norm",
    "extra_json",
]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class ExperimentLogger:
    """CSV experiment logger shared by classical, CBF, and future RL runs.

    Core controller and safety quantities use fixed columns for convenient
    analysis. Additional algorithm-specific values (reward, value estimate,
    actor loss, episode index, etc.) can be supplied through ``extras`` and are
    stored in the ``extra_json`` column without changing the CSV schema.
    """

    def __init__(
        self,
        root_dir: str | Path = "logs/high_level",
        run_name: str | None = None,
        flush_every: int = 10,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{run_name}" if run_name else ""
        self.run_dir = Path(root_dir).expanduser().resolve() / f"{timestamp}{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "data.csv"
        self.metadata_path = self.run_dir / "metadata.json"
        self.flush_every = max(1, int(flush_every))
        self._row_count = 0
        self._closed = False
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=STANDARD_FIELDS)
        self._writer.writeheader()

        if metadata is not None:
            self.write_metadata(metadata)

        print(f"[LOGGER] Writing data to: {self.csv_path}", flush=True)

    def write_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.metadata_path.write_text(
            json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(
        self,
        *,
        time_s: float,
        mode: str,
        trajectory: str,
        pose: np.ndarray,
        reference_pose: np.ndarray | None,
        measured_velocity: np.ndarray,
        nominal_command: np.ndarray,
        applied_command: np.ndarray,
        h: float,
        cbf_value_nominal: float,
        cbf_active: bool,
        cbf_correction_norm: float,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return

        pose = np.asarray(pose, dtype=np.float64)
        vel = np.asarray(measured_velocity, dtype=np.float64)
        u_nom = np.asarray(nominal_command, dtype=np.float64)
        u_cmd = np.asarray(applied_command, dtype=np.float64)

        if reference_pose is None:
            ref = np.full(3, np.nan, dtype=np.float64)
            error = np.full(3, np.nan, dtype=np.float64)
        else:
            ref = np.asarray(reference_pose, dtype=np.float64)
            error = ref - pose
            error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))

        row = {
            "time_s": float(time_s),
            "mode": mode,
            "trajectory": trajectory,
            "x": pose[0], "y": pose[1], "yaw": pose[2],
            "x_ref": ref[0], "y_ref": ref[1], "yaw_ref": ref[2],
            "vx_meas": vel[0], "vy_meas": vel[1], "wz_meas": vel[2],
            "vx_nom": u_nom[0], "vy_nom": u_nom[1], "wz_nom": u_nom[2],
            "vx_cmd": u_cmd[0], "vy_cmd": u_cmd[1], "wz_cmd": u_cmd[2],
            "ex": error[0], "ey": error[1], "eyaw": error[2],
            "position_error_norm": float(np.linalg.norm(error[:2])) if np.all(np.isfinite(error[:2])) else np.nan,
            "velocity_tracking_error_norm": float(np.linalg.norm(vel - u_cmd)),
            "h": float(h),
            "cbf_value_nominal": float(cbf_value_nominal),
            "cbf_active": int(bool(cbf_active)),
            "cbf_correction_norm": float(cbf_correction_norm),
            "extra_json": json.dumps(_jsonable(extras or {}), separators=(",", ":")),
        }
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True
        print(f"[LOGGER] Saved {self._row_count} rows to: {self.csv_path}", flush=True)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
