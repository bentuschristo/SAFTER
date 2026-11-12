#!/usr/bin/env python3
"""Fast TensorBoard scalar extractor/plotter without TensorFlow dependency.

This version uses TensorBoard's own tensor_util.make_ndarray, so it works in
environments that have tensorboard installed but do not have tensorflow.

Examples
--------
List scalar tags:
    python scripts/extract_plot_tensorboard_scalars_fast.py \
      --logdirs RUN1 RUN2 RUN3 \
      --labels "No CBF" "CBF-CF" "bCBF-CF" \
      --list-tags --max-events-for-tags 5000

Plot selected tags:
    python scripts/extract_plot_tensorboard_scalars_fast.py \
      --logdirs RUN1 RUN2 RUN3 \
      --labels "No CBF" "CBF-CF" "bCBF-CF" \
      --tags "charts/episodic_return" \
      --output-dir logs/evaluation/tensorboard_plots \
      --smooth-window 25 \
      --formats png pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logdirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--tag-regex", default=None)
    p.add_argument("--list-tags", action="store_true")
    p.add_argument("--max-events-for-tags", type=int, default=20000)
    p.add_argument("--output-dir", default="logs/evaluation/tensorboard_plots")
    p.add_argument("--smooth-window", type=int, default=1)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--x-axis", choices=["step", "wall_time"], default="step")
    p.add_argument("--max-points", type=int, default=0, help="Downsample each curve to at most this many points; 0 disables.")
    return p.parse_args()


def safe_name(text: str) -> str:
    text = text.strip().replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def event_files(logdir: Path) -> list[Path]:
    files = sorted(logdir.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")
    return files


def scalar_from_value(value) -> float | None:
    # Old scalar summaries.
    if value.HasField("simple_value"):
        return float(value.simple_value)

    # New tensor summaries. This does NOT require tensorflow.
    if value.HasField("tensor"):
        try:
            arr = tensor_util.make_ndarray(value.tensor)
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            return None

    return None


def stream_tags(logdir: Path, max_events: int) -> set[str]:
    tags: set[str] = set()
    seen_events = 0
    for file in event_files(logdir):
        loader = EventFileLoader(str(file))
        for event in loader.Load():
            seen_events += 1
            if event.HasField("summary"):
                for value in event.summary.value:
                    if scalar_from_value(value) is not None:
                        tags.add(value.tag)
            if max_events > 0 and seen_events >= max_events:
                return tags
    return tags


def stream_scalars(logdir: Path, wanted_tags: set[str], tag_re: re.Pattern | None) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict[str, float]]] = {}
    for file in event_files(logdir):
        print(f"  reading {file}", flush=True)
        loader = EventFileLoader(str(file))
        for event in loader.Load():
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                tag = value.tag
                if wanted_tags and tag not in wanted_tags:
                    continue
                if tag_re is not None and not tag_re.search(tag):
                    continue
                scalar = scalar_from_value(value)
                if scalar is None:
                    continue
                rows.setdefault(tag, []).append(
                    {
                        "wall_time": float(event.wall_time),
                        "step": int(event.step),
                        "value": scalar,
                    }
                )
    return {tag: pd.DataFrame(v).sort_values("step") for tag, v in rows.items()}


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    positions = [round(i * (len(df) - 1) / (max_points - 1)) for i in range(max_points)]
    positions = sorted(set(positions))
    return df.iloc[positions].copy()


def main() -> None:
    args = parse_args()
    logdirs = [Path(x).expanduser() for x in args.logdirs]
    labels = args.labels if args.labels is not None else [p.name for p in logdirs]
    if len(labels) != len(logdirs):
        raise ValueError("--labels length must match --logdirs length.")

    if args.list_tags:
        common = None
        for label, logdir in zip(labels, logdirs):
            print(f"\n[{label}] scanning up to {args.max_events_for_tags} events from {logdir}", flush=True)
            tags = stream_tags(logdir, args.max_events_for_tags)
            print(f"Found {len(tags)} scalar-like tags:")
            for tag in sorted(tags):
                print(f"  {tag}")
            common = tags if common is None else common & tags
        if common:
            print("\nCommon scalar-like tags:")
            for tag in sorted(common):
                print(f"  {tag}")
        return

    if not args.tags and not args.tag_regex:
        raise ValueError("Use --tags, --tag-regex, or --list-tags.")

    wanted_tags = set(args.tags or [])
    tag_re = re.compile(args.tag_regex) if args.tag_regex else None

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    all_tags = set()
    for label, logdir in zip(labels, logdirs):
        print(f"\n[{label}]", flush=True)
        data = stream_scalars(logdir, wanted_tags, tag_re)
        all_data[label] = data
        all_tags.update(data.keys())

    plot_tags = args.tags if args.tags else sorted(all_tags)

    for tag in plot_tags:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        combined = []
        any_found = False

        for label in labels:
            data = all_data[label]
            if tag not in data:
                print(f"[WARN] missing tag for {label}: {tag}")
                continue

            df = data[tag].copy()
            if df.empty:
                continue

            df = downsample(df, args.max_points)
            df["run"] = label
            df["tag"] = tag

            if args.smooth_window > 1:
                df["smoothed_value"] = df["value"].rolling(args.smooth_window, min_periods=1).mean()
                y = df["smoothed_value"]
            else:
                y = df["value"]

            ax.plot(df[args.x_axis], y, label=label)
            combined.append(df)
            any_found = True

        if not any_found:
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

        pd.concat(combined, ignore_index=True).to_csv(stem.with_suffix(".csv"), index=False)
        print(f"Saved: {stem}.{{{','.join(args.formats)}}} and CSV")


if __name__ == "__main__":
    main()
