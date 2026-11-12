#!/usr/bin/env python3
"""Plot all unseen-map layouts with shaded obstacles and optional trajectories.

Examples
--------
Map layouts only:
    python scripts/plot_all_unseen_maps_shaded.py \
      --run-dir logs/evaluation/unseen_maps/20260703_022535_seed42_20permap \
      --output-dir logs/evaluation/figures_v4/maps

Overlay one selected case when trajectory CSVs are available:
    python scripts/plot_all_unseen_maps_shaded.py \
      --run-dir logs/evaluation/unseen_maps/20260703_105125_seed42_20permap \
      --maps corridor \
      --case-id 3 \
      --output-dir logs/evaluation/figures_v4/trajectories
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd


MAP_ORDER = ["shifted", "corridor", "clustered", "slalom", "bottleneck"]
METHODS = ["no_cbf", "cbf_rl", "bcbf_rl"]
METHOD_LABELS = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-CF",
    "bcbf_rl": "bCBF-CF",
}
METHOD_COLORS = {
    "no_cbf": "tab:gray",
    "cbf_rl": "tab:blue",
    "bcbf_rl": "tab:orange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--maps",
        nargs="+",
        default=MAP_ORDER,
        choices=MAP_ORDER,
    )
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--robot-radius", type=float, default=0.30)
    parser.add_argument("--safety-margin", type=float, default=0.30)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def draw_map(
    ax: plt.Axes,
    map_df: pd.DataFrame,
    *,
    robot_radius: float,
    safety_margin: float,
) -> None:
    x_col = find_column(map_df, ["x", "center_x", "obstacle_x", "cx"])
    y_col = find_column(map_df, ["y", "center_y", "obstacle_y", "cy"])
    r_col = find_column(map_df, ["radius", "obstacle_radius", "r"])

    if x_col and y_col and r_col:
        for _, row in map_df.iterrows():
            x = float(row[x_col])
            y = float(row[y_col])
            radius = float(row[r_col])

            inflated = radius + robot_radius + safety_margin
            ax.add_patch(
                Circle(
                    (x, y),
                    inflated,
                    facecolor="tab:red",
                    edgecolor="none",
                    alpha=0.12,
                    zorder=1,
                )
            )
            ax.add_patch(
                Circle(
                    (x, y),
                    radius,
                    facecolor="tab:red",
                    edgecolor="darkred",
                    linewidth=1.2,
                    alpha=0.38,
                    zorder=2,
                )
            )

    # Optional rectangular obstacles.
    xmin_col = find_column(map_df, ["xmin", "x_min", "left"])
    xmax_col = find_column(map_df, ["xmax", "x_max", "right"])
    ymin_col = find_column(map_df, ["ymin", "y_min", "bottom"])
    ymax_col = find_column(map_df, ["ymax", "y_max", "top"])
    if xmin_col and xmax_col and ymin_col and ymax_col:
        for _, row in map_df.iterrows():
            xmin = float(row[xmin_col])
            xmax = float(row[xmax_col])
            ymin = float(row[ymin_col])
            ymax = float(row[ymax_col])
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor="tab:red",
                    edgecolor="darkred",
                    alpha=0.38,
                    linewidth=1.2,
                    zorder=2,
                )
            )


def trajectory_xy(df: pd.DataFrame) -> tuple[str, str]:
    x_col = find_column(
        df,
        [
            "post_x_m",
            "x",
            "position_x",
            "base_x",
            "robot_x",
            "px",
            "pre_x_m",
        ],
    )
    y_col = find_column(
        df,
        [
            "post_y_m",
            "y",
            "position_y",
            "base_y",
            "robot_y",
            "py",
            "pre_y_m",
        ],
    )

    if x_col is None or y_col is None:
        raise KeyError(
            f"Could not identify trajectory x/y columns. Columns: {list(df.columns)}"
        )

    return x_col, y_col


def plot_one_map(
    run_dir: Path,
    map_name: str,
    *,
    case_id: int | None,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    robot_radius: float,
    safety_margin: float,
) -> None:
    map_dir = run_dir / map_name
    map_csv = map_dir / "map.csv"
    scenarios_csv = map_dir / "scenarios.csv"

    if not map_csv.is_file():
        raise FileNotFoundError(map_csv)

    map_df = pd.read_csv(map_csv)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))

    draw_map(
        ax,
        map_df,
        robot_radius=robot_radius,
        safety_margin=safety_margin,
    )

    title_suffix = ""
    if case_id is not None:
        if not scenarios_csv.is_file():
            raise FileNotFoundError(scenarios_csv)
        scenarios = pd.read_csv(scenarios_csv)
        selected = scenarios[scenarios["case_id"] == case_id]
        if selected.empty:
            raise ValueError(f"Case {case_id} not found in {scenarios_csv}")
        case = selected.iloc[0]

        ax.scatter(
            [case["start_x"]],
            [case["start_y"]],
            marker="o",
            s=85,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            label="Start",
            zorder=10,
        )
        ax.scatter(
            [case["goal_x"]],
            [case["goal_y"]],
            marker="*",
            s=170,
            facecolor="gold",
            edgecolor="black",
            linewidth=0.8,
            label="Goal",
            zorder=10,
        )

        found_trajectory = False
        for method in METHODS:
            trajectory_csv = map_dir / method / f"{method}_trajectory.csv"
            if not trajectory_csv.is_file():
                continue
            traj = pd.read_csv(trajectory_csv)
            if "case_id" not in traj.columns:
                continue
            traj = traj[traj["case_id"] == case_id]
            if traj.empty:
                continue
            x_col, y_col = trajectory_xy(traj)
            ax.plot(
                traj[x_col],
                traj[y_col],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linewidth=2.2,
                zorder=6,
            )
            found_trajectory = True

        if not found_trajectory:
            print(
                f"[WARN] No trajectory CSV found for {map_name}, case {case_id}. "
                "Plotting map, start, and goal only."
            )
        title_suffix = f" — case {case_id}"

    ax.set_title(f"{map_name.capitalize()} map{title_suffix}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    # ax.legend(loc="best")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
    	ax.legend(loc="best")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"{map_name}_case_{case_id}_shaded"
        if case_id is not None
        else f"{map_name}_layout_shaded"
    )
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    for map_name in args.maps:
        plot_one_map(
            run_dir,
            map_name,
            case_id=args.case_id,
            output_dir=output_dir,
            formats=args.formats,
            dpi=args.dpi,
            robot_radius=args.robot_radius,
            safety_margin=args.safety_margin,
        )


if __name__ == "__main__":
    main()
