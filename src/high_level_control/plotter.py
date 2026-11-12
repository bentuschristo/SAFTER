from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_log(csv_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Log is empty: {csv_path}")

    out = Path(output_dir).expanduser().resolve() if output_dir else csv_path.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"], label="Robot")
    if df[["x_ref", "y_ref"]].notna().any().all():
        ax.plot(df["x_ref"], df["y_ref"], "--", label="Reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    path = out / "xy_trajectory.png"; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    labels = [("x", "x_ref", "x [m]"), ("y", "y_ref", "y [m]"), ("yaw", "yaw_ref", "yaw [rad]")]
    for ax, (state, ref, ylabel) in zip(axes, labels):
        ax.plot(df["time_s"], df[state], label="Measured")
        if df[ref].notna().any():
            ax.plot(df["time_s"], df[ref], "--", label="Reference")
        ax.set_ylabel(ylabel); ax.grid(True); ax.legend()
    axes[-1].set_xlabel("Time [s]")
    path = out / "state_tracking.png"; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    labels = [("vx_nom", "vx_cmd", "vx_meas", "v_x"), ("vy_nom", "vy_cmd", "vy_meas", "v_y"), ("wz_nom", "wz_cmd", "wz_meas", "omega_z")]
    for ax, (nom, cmd, meas, ylabel) in zip(axes, labels):
        ax.plot(df["time_s"], df[nom], label="Nominal")
        ax.plot(df["time_s"], df[cmd], "--", label="Applied")
        ax.plot(df["time_s"], df[meas], ":", label="Measured")
        ax.set_ylabel(ylabel); ax.grid(True); ax.legend()
    axes[-1].set_xlabel("Time [s]")
    path = out / "velocity_commands.png"; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(df["time_s"], df["position_error_norm"])
    axes[0].set_ylabel("Position error [m]"); axes[0].grid(True)
    axes[1].plot(df["time_s"], df["velocity_tracking_error_norm"])
    axes[1].set_ylabel("Velocity error norm"); axes[1].set_xlabel("Time [s]"); axes[1].grid(True)
    path = out / "tracking_errors.png"; _save(fig, path); paths.append(path)

    if np.isfinite(df["h"]).any():
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        axes[0].plot(df["time_s"], df["h"])
        axes[0].axhline(0.0, linestyle="--")
        axes[0].set_ylabel("h"); axes[0].grid(True)
        axes[1].plot(df["time_s"], df["cbf_correction_norm"], label="Correction norm")
        axes[1].step(df["time_s"], df["cbf_active"], where="post", label="Active")
        axes[1].set_ylabel("CBF"); axes[1].set_xlabel("Time [s]"); axes[1].grid(True); axes[1].legend()
        path = out / "cbf_metrics.png"; _save(fig, path); paths.append(path)

    return paths
