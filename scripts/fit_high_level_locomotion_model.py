from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_axis(
    current_velocity: np.ndarray,
    command: np.ndarray,
    next_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack(
        (
            current_velocity,
            command,
            np.ones_like(current_velocity),
        )
    )

    theta, *_ = np.linalg.lstsq(X, next_velocity, rcond=None)
    prediction = X @ theta
    residual = next_velocity - prediction
    return theta, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.trajectory_csv)

    # Do not fit across episode boundaries.
    same_episode = (
        df["episode"].to_numpy()[1:]
        == df["episode"].to_numpy()[:-1]
    )

    axes = [
        ("vx", "vx_body_mps", "safe_vx_mps"),
        ("vy", "vy_body_mps", "safe_vy_mps"),
        ("wz", "wz_body_radps", "safe_wz_radps"),
    ]

    for name, velocity_column, command_column in axes:
        velocity = df[velocity_column].to_numpy(dtype=float)
        command = df[command_column].to_numpy(dtype=float)

        v_k = velocity[:-1][same_episode]
        u_k = command[:-1][same_episode]
        v_next = velocity[1:][same_episode]

        theta, residual = fit_axis(v_k, u_k, v_next)
        f, g, bias = theta

        print(f"\n{name}")
        print(f"  F       = {f:+.8f}")
        print(f"  G       = {g:+.8f}")
        print(f"  bias    = {bias:+.8f}")
        print(
            "  |residual| q95/q99/max = "
            f"{np.quantile(np.abs(residual), 0.95):.6f} / "
            f"{np.quantile(np.abs(residual), 0.99):.6f} / "
            f"{np.max(np.abs(residual)):.6f}"
        )


if __name__ == "__main__":
    main()
