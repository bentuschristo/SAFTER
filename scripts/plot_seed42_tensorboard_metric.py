#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "no_cbf": ROOT
    / "logs/sac/go2_timeout90_server_no_cbf"
    / "2026-06-29_12-09-49_seed42_env256_12p8m",

    "cbf_cf": ROOT
    / "logs/sac/go2_timeout90_server_cbf_cf"
    / "2026-06-29_12-48-44_seed42_env256_12p8m",

    "bcbf_cf": ROOT
    / "logs/sac/go2_timeout90_server_bcbf_cf"
    / "2026-06-29_13-27-13_seed42_env256_12p8m",
}

DISPLAY_NAMES = {
    "no_cbf": "SAC",
    "cbf_cf": "SAC + CBF-CF",
    "bcbf_cf": "SAC + bCBF-CF",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", text).strip("_")


def load_scalar(run_dir: Path, tag: str) -> pd.DataFrame:
    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(
            f"Tag {tag!r} not found in {run_dir}\n"
            f"Available tags:\n" + "\n".join(tags)
        )

    events = accumulator.Scalars(tag)

    return pd.DataFrame(
        {
            "step": [event.step for event in events],
            "value": [event.value for event in events],
            "wall_time": [event.wall_time for event in events],
        }
    )


def smooth(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values

    return values.rolling(
        window=window,
        min_periods=1,
    ).mean()


def save_all(output_stem: Path, formats: list[str], dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    for extension in formats:
        output_path = output_stem.with_suffix(f".{extension}")
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(f"Saved: {output_path}")


def plot_combined(
    data: dict[str, pd.DataFrame],
    *,
    title: str,
    ylabel: str,
    smoothing_window: int,
    show_raw: bool,
    output_stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    plt.figure(figsize=(8.5, 5.2))

    for method, dataframe in data.items():
        if show_raw:
            plt.plot(
                dataframe["step"],
                dataframe["value"],
                linewidth=0.8,
                alpha=0.16,
            )

        plt.plot(
            dataframe["step"],
            smooth(dataframe["value"], smoothing_window),
            linewidth=2.0,
            label=DISPLAY_NAMES[method],
        )

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_all(output_stem, formats, dpi)
    plt.close()


def plot_separate(
    data: dict[str, pd.DataFrame],
    *,
    title: str,
    ylabel: str,
    smoothing_window: int,
    show_raw: bool,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    for method, dataframe in data.items():
        plt.figure(figsize=(8.5, 5.2))

        if show_raw:
            plt.plot(
                dataframe["step"],
                dataframe["value"],
                linewidth=0.8,
                alpha=0.20,
                label="Raw",
            )

        plt.plot(
            dataframe["step"],
            smooth(dataframe["value"], smoothing_window),
            linewidth=2.0,
            label=f"Smoothed ({smoothing_window}-point mean)",
        )

        plt.xlabel("Training step")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {DISPLAY_NAMES[method]}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        save_all(
            output_dir / method,
            formats,
            dpi,
        )
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one TensorBoard scalar for the seed-42 SAC runs."
    )

    parser.add_argument("--tag", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--ylabel", default=None)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(RUNS.keys()),
        default=list(RUNS.keys()),
    )

    parser.add_argument(
        "--mode",
        choices=("combined", "separate", "both"),
        default="both",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--show-raw",
        action="store_true",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "figures/training_seed42",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    title = args.title or args.tag
    ylabel = args.ylabel or args.tag

    data = {
        method: load_scalar(RUNS[method], args.tag)
        for method in args.methods
    }

    metric_dir = args.output_root / safe_name(args.tag)

    if args.mode in ("combined", "both"):
        plot_combined(
            data,
            title=title,
            ylabel=ylabel,
            smoothing_window=args.smooth_window,
            show_raw=args.show_raw,
            output_stem=metric_dir / "combined",
            formats=args.formats,
            dpi=args.dpi,
        )

    if args.mode in ("separate", "both"):
        plot_separate(
            data,
            title=title,
            ylabel=ylabel,
            smoothing_window=args.smooth_window,
            show_raw=args.show_raw,
            output_dir=metric_dir,
            formats=args.formats,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
