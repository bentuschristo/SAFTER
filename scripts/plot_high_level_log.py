"""Create standard plots from a high-level experiment CSV log."""
from __future__ import annotations

import argparse

from src.high_level_control.plotter import plot_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to data.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    paths = plot_log(args.csv, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
