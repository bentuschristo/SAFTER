#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir: Path) -> dict[str, pd.DataFrame]:
    event_files = list(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()

    scalars: dict[str, pd.DataFrame] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = pd.DataFrame(
            {
                "step": [event.step for event in events],
                "wall_time": [event.wall_time for event in events],
                "value": [event.value for event in events],
            }
        )

    return scalars


def stitch(
    first: dict[str, pd.DataFrame],
    second: dict[str, pd.DataFrame],
    tag: str,
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []

    if tag in first and not first[tag].empty:
        df1 = first[tag].copy()
        df1["phase"] = "Initial training"
        frames.append(df1)
        continuation_start = int(df1["step"].max()) + 1
    else:
        continuation_start = 0

    if tag in second and not second[tag].empty:
        df2 = second[tag].copy()

        if frames:
            second_min = int(df2["step"].min())
            second_max = int(df2["step"].max())
            first_max = int(frames[0]["step"].max())

            # The resumed run usually starts its TensorBoard step again at zero.
            if second_min <= first_max:
                df2["step"] = (
                    df2["step"] - second_min + continuation_start
                )

        df2["phase"] = "Continued training"
        frames.append(df2)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["step", "wall_time"])
    combined = combined.drop_duplicates(subset="step", keep="last")
    return combined.reset_index(drop=True)


def filename_from_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.90)
    args = parser.parse_args()

    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("--smoothing must be in [0, 1).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    first = load_scalars(args.first_run)
    second = load_scalars(args.second_run)
    all_tags = sorted(set(first) | set(second))

    print("Available scalar tags:")
    for tag in all_tags:
        print(f"  {tag}")

    search_terms = [
        "mean_reward",
        "mean_episode_length",
        "reward_progress",
        "distance_to_goal",
        "success_rate",
        "collision_rate",
        "fall_rate",
        "cbf_active_rate",
        "cbf_correction_norm",
        "value_function",
        "surrogate",
        "mean_noise_std",
    ]

    selected_tags: list[str] = []
    for term in search_terms:
        selected_tags.extend(
            tag for tag in all_tags if term.lower() in tag.lower()
        )
    selected_tags = list(dict.fromkeys(selected_tags))

    if not selected_tags:
        raise RuntimeError(
            "No expected scalar tags were found. "
            "Review the printed Available scalar tags list."
        )

    summary_rows = []

    for tag in selected_tags:
        df = stitch(first, second, tag)
        if df is None or df.empty:
            continue

        alpha = 1.0 - args.smoothing
        df["smoothed"] = df["value"].ewm(
            alpha=alpha,
            adjust=False,
        ).mean()

        name = filename_from_tag(tag)
        df.to_csv(args.output_dir / f"{name}.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            df["step"],
            df["value"],
            linewidth=0.7,
            alpha=0.25,
            label="Raw",
        )
        ax.plot(
            df["step"],
            df["smoothed"],
            linewidth=1.8,
            label="Smoothed",
        )

        if tag in first and not first[tag].empty:
            boundary = int(first[tag]["step"].max()) + 1
            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1.0,
                label="Training resumed",
            )

        ax.set_xlabel("Training iteration")
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            args.output_dir / f"{name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        fig.savefig(
            args.output_dir / f"{name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        summary_rows.append(
            {
                "tag": tag,
                "first_step": int(df["step"].min()),
                "last_step": int(df["step"].max()),
                "last_raw_value": float(df["value"].iloc[-1]),
                "last_smoothed_value": float(df["smoothed"].iloc[-1]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print(f"\nSaved plots to: {args.output_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
