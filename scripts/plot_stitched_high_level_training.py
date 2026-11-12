#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    if not list(run_dir.rglob("events.out.tfevents*")):
        raise FileNotFoundError(f"No TensorBoard event file under {run_dir}")

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()

    result = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        result[tag] = pd.DataFrame(
            {
                "local_step": [e.step for e in events],
                "wall_time": [e.wall_time for e in events],
                "value": [e.value for e in events],
            }
        )
    return result


def safe_name(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.90)
    args = parser.parse_args()

    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("--smoothing must be in [0, 1).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for run in args.runs:
        print(f"Loading: {run}")
        loaded.append(load_run(run))

    all_tags = sorted(set().union(*(set(run) for run in loaded)))

    print("\nAvailable scalar tags:")
    for tag in all_tags:
        print(f"  {tag}")

    terms = [
        "mean_reward",
        "mean_episode_length",
        "reward_progress",
        "distance_to_goal",
        "success_rate",
        "collision_rate",
        "fall_rate",
        "smoothness_cost",
        "cbf_active_rate",
        "cbf_correction_norm",
        "value_function",
        "surrogate",
        "mean_noise_std",
    ]

    selected = [
        tag for tag in all_tags
        if any(term.lower() in tag.lower() for term in terms)
    ]

    summary_rows = []

    for tag in selected:
        pieces = []
        next_global_step = 0
        boundaries = []

        for phase_index, run in enumerate(loaded):
            if tag not in run or run[tag].empty:
                continue

            df = run[tag].copy().sort_values(
                ["local_step", "wall_time"]
            )
            df = df.drop_duplicates("local_step", keep="last")

            local_start = int(df["local_step"].min())
            df["step"] = (
                df["local_step"] - local_start + next_global_step
            )
            df["phase"] = phase_index + 1

            if pieces:
                boundaries.append(next_global_step)

            next_global_step = int(df["step"].max()) + 1
            pieces.append(df)

        if not pieces:
            continue

        combined = pd.concat(pieces, ignore_index=True)
        combined = combined.sort_values("step").reset_index(drop=True)

        alpha = 1.0 - args.smoothing
        combined["smoothed"] = combined["value"].ewm(
            alpha=alpha,
            adjust=False,
        ).mean()

        filename = safe_name(tag)
        combined.to_csv(
            args.output_dir / f"{filename}.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(9, 4.8))

        ax.plot(
            combined["step"],
            combined["value"],
            linewidth=0.65,
            alpha=0.22,
            label="Raw",
        )
        ax.plot(
            combined["step"],
            combined["smoothed"],
            linewidth=1.8,
            label="Smoothed",
        )

        for index, boundary in enumerate(boundaries):
            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1.0,
                label="Training resumed" if index == 0 else None,
            )

        ax.set_xlabel("Training iteration")
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            args.output_dir / f"{filename}.png",
            dpi=200,
            bbox_inches="tight",
        )
        fig.savefig(
            args.output_dir / f"{filename}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        summary_rows.append(
            {
                "tag": tag,
                "first_step": int(combined["step"].min()),
                "last_step": int(combined["step"].max()),
                "last_raw_value": float(combined["value"].iloc[-1]),
                "last_smoothed_value": float(
                    combined["smoothed"].iloc[-1]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print(f"\nSaved plots to: {args.output_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
