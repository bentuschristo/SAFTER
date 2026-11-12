#!/usr/bin/env python3
"""Mean +/- std training success-rate curve across seeds 0-4 for the
SEA-Nav-matched-speed (10 Hz) vanilla/cbf_cd_cf/bcbf_cf/cbf_qp/bcbf_qp runs,
sourced from the TensorBoard CSV exports the user places under
`New Training/360-HighSpeed/Success Rates/`.

SEA-Nav's own curve is intentionally NOT plotted (2026-09-21 revision, per
request) -- only its per-seed peak is still reported to the console as a
diagnostic, sourced from `New Training/360-new/seanav-successrate-360-seed*.csv`.

Each SAC per-seed CSV's "Value" column is already TensorBoard's own
100-episode rolling-mean success rate (see the exporting run's own logging
code), so no extra smoothing is applied to those -- only interpolation onto
a shared step grid so seeds can be averaged pointwise.

Produces two figures: the full run, and a second capped at HALF_TRAINING_
TRANSITIONS_M (~half the total run) for a training-progress-so-far view.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path("New Training/360-HighSpeed/Success Rates")
SEANAV_CSV_DIR = Path("New Training/360-new")
OUT_DIR = Path("New Training/360-HighSpeed/figure")

METHODS = {
    "vanilla": ("Vanilla SAC", "#1f77b4", "-"),
    "cbf_cd_cf": ("CBF-CF", "#ff7f0e", "-"),
    "bcbf_cf": ("SAFTER", "#2ca02c", "-"),
    "cbf_qp": ("CBF-QP", "#9467bd", "--"),
    "bcbf_qp": ("bCBF-QP", "#8c564b", "--"),
}
FILE_RE = re.compile(r"^(vanilla|cbf_cd_cf|bcbf_cf|cbf_qp|bcbf_qp)_seed(\d+)_\.\.csv$")
SEANAV_FILE_RE = re.compile(r"^seanav-successrate-360-seed(\d+)\.csv$")
SEANAV_SMOOTH_WINDOW = 20

# Font sizes (2026-09-22: matched to New Training/plot_training_rates_
# multiseed_no_seanav.py's success_rate_convergence_10m_square style).
FS_TITLE = 19
FS_LABEL = 17
FS_TICK = 14
FS_LEGEND = 11

# Full-run x-axis upper bound (million transitions).
FULL_XLIM_M = 20.5
# Second figure: capped at roughly half the total training run.
HALF_TRAINING_TRANSITIONS_M = 10.25


def load_run(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "step": pd.to_numeric(raw["Step"], errors="coerce"),
        "value": pd.to_numeric(raw["Value"], errors="coerce"),
    }).dropna()
    return df.sort_values("step").drop_duplicates("step", keep="last").reset_index(drop=True)


def load_seanav_run(path: Path) -> pd.DataFrame:
    df = load_run(path)
    df["value"] = df["value"].rolling(SEANAV_SMOOTH_WINDOW, min_periods=1).mean()
    return df


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


def report_seanav_peak(items: list[tuple[int, pd.DataFrame]]) -> None:
    """Per-seed peak (highest smoothed value reached, i.e. "converges to"
    before any later drop) and the transition at which it occurs -- no
    threshold criterion, just the actual peak. Console-only diagnostic;
    SEA-Nav is no longer plotted on the figure itself."""
    peaks, peak_steps = [], []
    for seed, df in sorted(items):
        idx = df["value"].idxmax()
        peaks.append(float(df.loc[idx, "value"]))
        peak_steps.append(float(df.loc[idx, "step"]))
        print(f"  SEA-Nav seed{seed}: peak={peaks[-1] * 100:.1f}% at {peak_steps[-1] / 1e6:.2f}M transitions")
    peaks_arr, steps_arr = np.array(peaks), np.array(peak_steps)
    print(f"  SEA-Nav MEAN peak success rate = {peaks_arr.mean() * 100:.1f} +/- {peaks_arr.std(ddof=1) * 100:.1f} %")
    print(f"  SEA-Nav MEAN convergence transition = {steps_arr.mean() / 1e6:.2f} +/- {steps_arr.std(ddof=1) / 1e6:.2f} M")


def make_figure(runs: dict[str, list[tuple[int, pd.DataFrame]]], xlim_m: float, filename_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for key, (label, color, linestyle) in METHODS.items():
        items = sorted(runs[key])
        all_steps, mean, sd = mean_std_curve(items)
        ax.fill_between(all_steps / 1e6, (mean - sd) * 100, (mean + sd) * 100, color=color, alpha=0.20, linewidth=0)
        ax.plot(all_steps / 1e6, mean * 100, color=color, lw=2.3, linestyle=linestyle, label=label)

    ax.axhline(90, color="red", linestyle="--", linewidth=1.8, label="90% Threshold", zorder=5)

    ax.set_title("Training Success Rate", fontsize=FS_TITLE, pad=10)
    ax.set_xlabel("Environment Transitions (million)", fontsize=FS_LABEL)
    ax.set_ylabel("Success Rate (%)", fontsize=FS_LABEL)
    ax.set_xlim(0, xlim_m)
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="lower right", ncol=2, fontsize=FS_LEGEND, frameon=True, framealpha=0.65)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"seaspeed_seeds0to4_success_rate_mean_std{filename_suffix}.{ext}"
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

    empty = [key for key, items in runs.items() if len(items) == 0]
    if empty:
        raise SystemExit(f"No seed CSVs found for: {', '.join(empty)}")
    for key, items in runs.items():
        if len(items) != 5:
            print(f"[note] {key}: plotting with {len(items)} seed(s) (expected 5)")

    seanav_runs: list[tuple[int, pd.DataFrame]] = []
    for path in sorted(SEANAV_CSV_DIR.glob("seanav-successrate-360-seed*.csv")):
        match = SEANAV_FILE_RE.match(path.name)
        if not match:
            continue
        seanav_runs.append((int(match.group(1)), load_seanav_run(path)))
    if len(seanav_runs) == 5:
        print("SEA-Nav convergence (peak value, no threshold; console-only, not plotted):")
        report_seanav_peak(seanav_runs)
    else:
        print(f"[note] Expected five SEA-Nav seeds for the console diagnostic; found {len(seanav_runs)} -- skipping.")

    make_figure(runs, FULL_XLIM_M, "")
    make_figure(runs, HALF_TRAINING_TRANSITIONS_M, "_half_training")


if __name__ == "__main__":
    main()
