#!/usr/bin/env python3
"""Mean +/- std training REWARD curve across seeds 0-4 for the
SEA-Nav-matched-speed (10 Hz) vanilla/cbf_cd_cf/bcbf_cf/cbf_qp/bcbf_qp runs,
sourced from the TensorBoard CSV exports under
`New Training/360-HighSpeed/Reward/`.

SEA-Nav's own curve is intentionally NOT plotted (2026-09-22 revision, per
request -- matches the success-rate script's own SEA-Nav removal).

Each SAC per-seed CSV's "Value" column is already TensorBoard's own
100-episode rolling-mean reward, so no extra smoothing is applied to those --
only interpolation onto a shared step grid so seeds can be averaged
pointwise. Title/axis labels match the existing
New Training/360-new/figures/reward.png reference figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path("New Training/360-HighSpeed/Reward")
OUT_DIR = Path("New Training/360-HighSpeed/figure")

METHODS = {
    "vanilla": ("Vanilla SAC", "#1f77b4", "-"),
    "cbf_cd_cf": ("CBF-CF", "#ff7f0e", "-"),
    "bcbf_cf": ("bCBF-CF", "#2ca02c", "-"),
    "cbf_qp": ("CBF-QP", "#9467bd", "--"),
    "bcbf_qp": ("bCBF-QP", "#8c564b", "--"),
}
FILE_RE = re.compile(r"^(vanilla|cbf_cd_cf|bcbf_cf|cbf_qp|bcbf_qp)_seed(\d+)_\.\.csv$")


def load_run(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "step": pd.to_numeric(raw["Step"], errors="coerce"),
        "value": pd.to_numeric(raw["Value"], errors="coerce"),
    }).dropna()
    return df.sort_values("step").drop_duplicates("step", keep="last").reset_index(drop=True)


def mean_std_curve(items: list[tuple[int, pd.DataFrame]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_steps = np.unique(np.concatenate([df["step"].to_numpy() for _, df in items]))
    seed_values = []
    for seed, df in items:
        values = np.interp(all_steps, df["step"], df["value"])
        values[(all_steps < df["step"].iloc[0]) | (all_steps > df["step"].iloc[-1])] = np.nan
        seed_values.append(values)
    matrix = np.vstack(seed_values)
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    sd = np.full_like(mean, np.nan)
    valid = count >= 2
    sd[valid] = np.nanstd(matrix[:, valid], axis=0, ddof=1)
    return all_steps, mean, sd


# Second/third figure: capped at roughly half the total training run (and a
# square-aspect version of that same half-training view, matching
# New Training/plot_training_rates_multiseed_no_seanav.py's
# CONVERGENCE_FIGSIZES["_square"] = (6.4, 4.8) exactly).
HALF_TRAINING_TRANSITIONS_M = 10.25
HALF_TRAINING_SQUARE_FIGSIZE = (6.4, 4.8)


def make_figure(
    runs: dict[str, list[tuple[int, pd.DataFrame]]],
    xlim_m: float,
    filename_suffix: str,
    figsize: tuple[float, float] = (6.0, 3.7),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for key, (label, color, linestyle) in METHODS.items():
        items = sorted(runs[key])
        all_steps, mean, sd = mean_std_curve(items)
        ax.fill_between(all_steps / 1e6, mean - sd, mean + sd, color=color, alpha=0.20, linewidth=0)
        ax.plot(all_steps / 1e6, mean, color=color, lw=1.6, linestyle=linestyle, label=label)

    ax.set_title("Training Reward", fontsize=13, pad=10)
    ax.set_xlabel("Environment Transitions (million)", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_xlim(0, xlim_m)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.75)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"seaspeed_seeds0to4_reward_mean_std{filename_suffix}.{ext}"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[tuple[int, pd.DataFrame]]] = {k: [] for k in METHODS}
    for path in sorted(CSV_DIR.glob("*_seed*_..csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        key, seed_text = match.groups()
        runs[key].append((int(seed_text), load_run(path)))

    missing = [key for key, items in runs.items() if len(items) != 5]
    if missing:
        counts = ", ".join(f"{key}={len(runs[key])}" for key in missing)
        raise SystemExit(f"Expected five seeds per method; found {counts}")

    make_figure(runs, 20.5, "")
    make_figure(runs, HALF_TRAINING_TRANSITIONS_M, "_half_training")
    make_figure(
        runs, HALF_TRAINING_TRANSITIONS_M, "_half_training_square",
        figsize=HALF_TRAINING_SQUARE_FIGSIZE,
    )


if __name__ == "__main__":
    main()
