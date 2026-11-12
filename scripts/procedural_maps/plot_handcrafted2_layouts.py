#!/usr/bin/env python3
"""Render handcrafted2 layouts in the paper evaluation-environment style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Patch, Rectangle


# Exact palette used by scripts_august/scripts_v4/plot_five_unseen_maps_regions.py.
START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
REGION_ALPHA = 0.32
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 15


def load_bank(bank_dir: Path):
    config = json.loads((bank_dir / "generation_config.json").read_text())
    maps = []
    for path in sorted((bank_dir / "maps").glob("map_*.json")):
        maps.append(json.loads(path.read_text()))
    if len(maps) != 5:
        raise ValueError(f"Expected five maps in {bank_dir}, found {len(maps)}")
    return config, maps


def add_region(ax, bounds, color):
    x_min, x_max, y_min, y_max = map(float, bounds)
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=REGION_ALPHA,
            zorder=1,
        )
    )


def draw_map(ax, map_data, regions, *, title=None, show_x=True, show_y=True):
    arena = map_data["arena"]
    x_limit = float(arena["width"]) / 2.0
    y_limit = float(arena["length"]) / 2.0

    add_region(ax, regions["start_region"], START_COLOR)
    add_region(ax, regions["goal_region"], GOAL_COLOR)

    for obstacle in map_data["obstacles"]:
        ax.add_patch(
            Circle(
                (float(obstacle["x"]), float(obstacle["y"])),
                float(obstacle["radius"]),
                facecolor=OBSTACLE_COLOR,
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )
        )

    ax.add_patch(
        Rectangle(
            (-x_limit, -y_limit),
            2 * x_limit,
            2 * y_limit,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            zorder=4,
        )
    )
    ax.set_xlim(-x_limit - 0.15, x_limit + 0.15)
    ax.set_ylim(-y_limit - 0.15, y_limit + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=TICK_SIZE, length=4)
    ax.set_xlabel("x [m]" if show_x else "", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("y [m]" if show_y else "", fontsize=AXIS_LABEL_SIZE)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=8)


def legend_handles():
    return [
        Patch(facecolor=START_COLOR, alpha=REGION_ALPHA, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, alpha=REGION_ALPHA, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]


def save(fig, stem: Path, dpi: int):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=Path("generated/procedural_maps/handcrafted2"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    bank_dir = args.bank_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else bank_dir / "paper_layouts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config, maps = load_bank(bank_dir)
    region_map = config["per_map_endpoint_regions"]

    for map_data in maps:
        map_id = int(map_data["map_id"])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        draw_map(ax, map_data, region_map[str(map_id)], title=f"Training map {map_id + 1}")
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.91)
        save(fig, output_dir / f"handcrafted2_map_{map_id:03d}_layout", args.dpi)
        plt.close(fig)

    # Six equal GridSpec columns give three equal panels above two centered panels.
    fig = plt.figure(figsize=(13.8, 9.0))
    grid = GridSpec(2, 6, figure=fig, wspace=0.55, hspace=0.32)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    for index, (ax, map_data) in enumerate(zip(axes, maps)):
        map_id = int(map_data["map_id"])
        draw_map(
            ax,
            map_data,
            region_map[str(map_id)],
            title=f"Training map {map_id + 1}",
            show_x=True,
            show_y=index in (0, 3),
        )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.95)
    save(fig, output_dir / "handcrafted2_all_maps_layout", args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 0.8))
    ax.axis("off")
    ax.legend(handles=legend_handles(), loc="center", ncol=3, frameon=False, fontsize=12)
    save(fig, output_dir / "handcrafted2_layout_legend", args.dpi)
    plt.close(fig)
    print(f"Saved handcrafted2 paper layouts to {output_dir}")


if __name__ == "__main__":
    main()
