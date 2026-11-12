from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.trajectory_csv)

    v_columns = [
        "vx_body_mps",
        "vy_body_mps",
        "wz_body_radps",
    ]
    u_columns = [
        "safe_vx_mps",
        "safe_vy_mps",
        "safe_wz_radps",
    ]

    regressors = []
    targets = []

    for _, episode in df.groupby("episode"):
        episode = episode.sort_values("step")

        v = episode[v_columns].to_numpy(dtype=float)
        u = episode[u_columns].to_numpy(dtype=float)

        if len(episode) < 2:
            continue

        # v[k+1] = F v[k] + G u[k] + c
        x = np.concatenate(
            [
                v[:-1],
                u[:-1],
                np.ones((len(v) - 1, 1)),
            ],
            axis=1,
        )
        y = v[1:]

        regressors.append(x)
        targets.append(y)

    X = np.concatenate(regressors, axis=0)
    Y = np.concatenate(targets, axis=0)

    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)

    F = theta[:3, :].T
    G = theta[3:6, :].T
    c = theta[6, :]

    prediction = X @ theta
    residual = Y - prediction

    print("Discrete model:")
    print("v[k+1] = F v[k] + G u[k] + c")
    print()
    print("F =")
    print(F)
    print()
    print("G =")
    print(G)
    print()
    print("c =")
    print(c)

    print("\nResidual absolute-error statistics:")
    for index, name in enumerate(v_columns):
        absolute = np.abs(residual[:, index])
        print(
            f"{name:16s} "
            f"mean={absolute.mean():.6f}, "
            f"q95={np.quantile(absolute, 0.95):.6f}, "
            f"q99={np.quantile(absolute, 0.99):.6f}, "
            f"max={absolute.max():.6f}"
        )

    # Diagonal continuous-time approximation, where physically meaningful.
    print("\nDiagonal continuous-time approximation:")
    for index, name in enumerate(v_columns):
        f_ii = F[index, index]

        if 0.0 < f_ii < 1.0:
            b_i = -np.log(f_ii) / args.dt
            print(
                f"{name:16s}: "
                f"F_ii={f_ii:.6f}, B_i={b_i:.6f} 1/s"
            )
        else:
            print(
                f"{name:16s}: "
                f"F_ii={f_ii:.6f}, no stable positive B_i inferred"
            )


if __name__ == "__main__":
    main()
