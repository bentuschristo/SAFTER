#!/usr/bin/env python3
"""Extract TensorBoard scalar curves to CSV and recreate plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logdirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--list-tags", action="store_true")
    p.add_argument("--output-dir", default="logs/evaluation/tensorboard_plots")
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--x-axis", choices=["step", "wall_time"], default="step")
    return p.parse_args()


def safe_name(text: str) -> str:
    text = text.strip().replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def load_event_scalars(logdir: Path) -> dict[str, pd.DataFrame]:
    if not logdir.exists():
        raise FileNotFoundError(logdir)

    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame(
            {
                "wall_time": [e.wall_time for e in events],
                "step": [e.step for e in events],
                "value": [e.value for e in events],
            }
        )
    return data


def main() -> None:
    args = parse_args()

    logdirs = [Path(x).expanduser() for x in args.logdirs]
    if args.labels is None:
        labels = [p.name for p in logdirs]
    else:
        labels = args.labels
        if len(labels) != len(logdirs):
            raise ValueError("--labels length must match --logdirs length.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for label, logdir in zip(labels, logdirs):
        all_data[label] = load_event_scalars(logdir)

    if args.list_tags:
        print("\nAvailable scalar tags:")
        common = None
        for label, data in all_data.items():
            tags = sorted(data.keys())
            print(f"\n[{label}] {len(tags)} scalar tags")
            for tag in tags:
                print(f"  {tag}")
            common = set(tags) if common is None else common & set(tags)

        if common:
            print("\nCommon scalar tags:")
            for tag in sorted(common):
                print(f"  {tag}")
        return

    if not args.tags:
        raise ValueError("Pass --tags TAG1 TAG2 ... or use --list-tags first.")

    for tag in args.tags:
        combined_rows = []
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

        any_found = False
        for label, data in all_data.items():
            if tag not in data:
                print(f"[WARN] tag not found for {label}: {tag}")
                continue

            df = data[tag].copy()
            df["run"] = label
            df["tag"] = tag

            if args.smooth_window > 1:
                df["smoothed_value"] = (
                    df["value"]
                    .rolling(args.smooth_window, min_periods=1)
                    .mean()
                )
                y_col = "smoothed_value"
            else:
                y_col = "value"

            combined_rows.append(df)
            ax.plot(df[args.x_axis], df[y_col], label=label)
            any_found = True

        if not any_found:
            print(f"[WARN] no data found for tag: {tag}")
            plt.close(fig)
            continue

        ax.set_title(tag)
        ax.set_xlabel("Step" if args.x_axis == "step" else "Wall time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        stem = output_dir / safe_name(tag)
        for fmt in args.formats:
            fig.savefig(stem.with_suffix(f".{fmt}"), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(stem.with_suffix(".csv"), index=False)

        print(f"Saved plot/CSV for tag '{tag}' to: {stem}")


if __name__ == "__main__":
    main()
