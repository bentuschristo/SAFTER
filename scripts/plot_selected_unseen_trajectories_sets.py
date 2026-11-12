from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd


ROOT = Path(
    "logs/evaluation/unseen_maps/"
    "20260629_131229_seed10_100permap"
)
MAP_NAME = "shifted"
CASES = [26, 78]

METHODS = ["no_cbf", "cbf_rl", "bcbf_rl"]

METHOD_LABELS = {
    "no_cbf": "No CBF",
    "cbf_rl": "CBF-RL",
    "bcbf_rl": "bCBF-RL",
}

# Arena used by the unseen-map evaluation.
X_MIN, X_MAX = -7.5, 7.5
Y_MIN, Y_MAX = -7.5, 7.5

# Robot footprint used by the collision test.
#
# From the logs:
# obstacle radius 0.55 m + robot radius 0.30 m = 0.85 m.
ROBOT_RADIUS_M = 0.30

# Extra geometric margin used here to visualize the backup set.
#
# IMPORTANT:
# Set this equal to the terminal/backup clearance margin in your bCBF
# implementation. If the implementation uses 0.15 m, retain 0.15.
BACKUP_MARGIN_M = 0.15

# Plot resolution for set shading.
GRID_POINTS = 500

# Number of sampled body-heading arrows per trajectory.
# Initial and final samples are always included.
HEADING_SAMPLE_COUNT = 6
HEADING_ARROW_LENGTH_M = 0.55

OUT = ROOT / "figures_with_sets"
OUT.mkdir(parents=True, exist_ok=True)


def load_method_data(
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_dir = ROOT / MAP_NAME / method

    trajectory = pd.read_csv(
        method_dir / f"{method}_trajectory.csv"
    )
    episodes = pd.read_csv(
        method_dir / f"{method}_episodes.csv"
    )

    return trajectory, episodes


def build_xy_path(
    case_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    rows = case_rows.sort_values("step")

    x = np.concatenate(
        (
            np.array([rows["pre_x_m"].iloc[0]]),
            rows["post_x_m"].to_numpy(),
        )
    )
    y = np.concatenate(
        (
            np.array([rows["pre_y_m"].iloc[0]]),
            rows["post_y_m"].to_numpy(),
        )
    )

    return x, y


def compute_set_masks(
    map_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(X_MIN, X_MAX, GRID_POINTS)
    y_values = np.linspace(Y_MIN, Y_MAX, GRID_POINTS)
    xx, yy = np.meshgrid(x_values, y_values)

    safe_clearance = np.full_like(xx, np.inf, dtype=float)
    backup_clearance = np.full_like(xx, np.inf, dtype=float)

    for obstacle in map_df.itertuples(index=False):
        distance = np.sqrt(
            (xx - float(obstacle.x)) ** 2
            + (yy - float(obstacle.y)) ** 2
        )

        safe_radius = (
            float(obstacle.radius) + ROBOT_RADIUS_M
        )
        backup_radius = safe_radius + BACKUP_MARGIN_M

        safe_clearance = np.minimum(
            safe_clearance,
            distance - safe_radius,
        )
        backup_clearance = np.minimum(
            backup_clearance,
            distance - backup_radius,
        )

    safe_mask = safe_clearance >= 0.0
    backup_mask = backup_clearance >= 0.0

    return xx, yy, safe_mask, backup_mask


def draw_sets(
    ax: plt.Axes,
    map_df: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    safe_mask: np.ndarray,
    backup_mask: np.ndarray,
) -> None:
    # Safe-only region C_S \\ C_B:
    # the band between the original safe boundary and the more
    # conservative backup boundary.
    safe_only_mask = safe_mask & (~backup_mask)

    ax.contourf(
        xx,
        yy,
        safe_only_mask.astype(float),
        levels=[0.5, 1.5],
        alpha=0.90,
        colors=["#39ff88"],
        zorder=1,
    )

    # Backup set C_B: the region satisfying the larger clearance.
    ax.contourf(
        xx,
        yy,
        backup_mask.astype(float),
        levels=[0.5, 1.5],
        alpha=0.16,
        colors=["#7aa6e8"],
        zorder=0,
    )

    for obstacle in map_df.itertuples(index=False):
        center = (float(obstacle.x), float(obstacle.y))
        obstacle_radius = float(obstacle.radius)
        safe_radius = obstacle_radius + ROBOT_RADIUS_M
        backup_radius = safe_radius + BACKUP_MARGIN_M

        # Backup boundary.
        ax.add_patch(
            Circle(
                center,
                backup_radius,
                facecolor="none",
                edgecolor="#2855a6",
                linewidth=1.3,
                linestyle="-.",
                zorder=2,
            )
        )

        # Safe/unsafe boundary.
        ax.add_patch(
            Circle(
                center,
                safe_radius,
                facecolor="none",
                edgecolor="#27864b",
                linewidth=1.5,
                zorder=3,
            )
        )

        # Red region between physical obstacle and safe boundary.
        ax.add_patch(
            Circle(
                center,
                safe_radius,
                facecolor="#e85d68",
                edgecolor="none",
                alpha=0.50,
                zorder=4,
            )
        )

        # Physical obstacle on top of the red annulus.
        ax.add_patch(
            Circle(
                center,
                obstacle_radius,
                facecolor="#d3d3d3",
                edgecolor="black",
                linewidth=1.2,
                zorder=5,
            )
        )

        ax.text(
            float(obstacle.x),
            float(obstacle.y),
            str(int(obstacle.obstacle_id)),
            ha="center",
            va="center",
            fontsize=7,
            zorder=6,
        )


def draw_start_goal(
    ax: plt.Axes,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
) -> None:
    ax.scatter(
        start_x,
        start_y,
        marker="o",
        s=85,
        edgecolors="black",
        linewidths=1.0,
        zorder=12,
    )

    ax.scatter(
        goal_x,
        goal_y,
        marker="*",
        s=190,
        edgecolors="black",
        linewidths=1.0,
        zorder=12,
    )

    ax.plot(
        [start_x, goal_x],
        [start_y, goal_y],
        linestyle="--",
        linewidth=1.1,
        alpha=0.55,
        zorder=7,
    )


def configure_axis(ax: plt.Axes) -> None:
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(True, alpha=0.23, zorder=20)



def draw_heading_samples(
    ax: plt.Axes,
    case_rows: pd.DataFrame,
) -> None:
    """Draw body-heading arrows with distance-based spacing."""
    rows = case_rows.sort_values("step").reset_index(drop=True)

    if "yaw_rad" not in rows.columns:
        raise RuntimeError(
            "Trajectory CSV does not contain yaw_rad."
        )

    x_all = rows["pre_x_m"].to_numpy(dtype=float)
    y_all = rows["pre_y_m"].to_numpy(dtype=float)
    yaw_all = rows["yaw_rad"].to_numpy(dtype=float)

    if len(rows) == 1:
        indices = np.array([0], dtype=int)
    else:
        segment_lengths = np.hypot(
            np.diff(x_all),
            np.diff(y_all),
        )
        cumulative_distance = np.concatenate(
            ([0.0], np.cumsum(segment_lengths))
        )
        total_distance = float(cumulative_distance[-1])

        # Initial and final arrows are always included.
        # Add roughly one intermediate sample per 2 m of travel.
        desired_count = int(np.ceil(total_distance / 2.0)) + 1
        sample_count = int(np.clip(desired_count, 2, 6))

        target_distances = np.linspace(
            0.0,
            total_distance,
            sample_count,
        )

        indices = np.array(
            [
                int(np.argmin(np.abs(cumulative_distance - target)))
                for target in target_distances
            ],
            dtype=int,
        )
        indices = np.unique(indices)

    x = x_all[indices]
    y = y_all[indices]
    yaw = yaw_all[indices]

    u = HEADING_ARROW_LENGTH_M * np.cos(yaw)
    v = HEADING_ARROW_LENGTH_M * np.sin(yaw)

    ax.quiver(
        x,
        y,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0045,
        headwidth=4.2,
        headlength=5.0,
        headaxislength=4.5,
        pivot="tail",
        color="black",
        zorder=15,
    )

    ax.scatter(
        x,
        y,
        s=11,
        color="black",
        zorder=16,
    )


def plot_case(
    case_id: int,
    map_df: pd.DataFrame,
    method_data: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
    xx: np.ndarray,
    yy: np.ndarray,
    safe_mask: np.ndarray,
    backup_mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.7),
        sharex=True,
        sharey=True,
    )

    for ax, method in zip(axes, METHODS):
        trajectory, episodes = method_data[method]

        case_rows = trajectory[
            trajectory["case_id"].astype(int) == case_id
        ].copy()

        if case_rows.empty:
            raise RuntimeError(
                f"No trajectory for {method}, case {case_id}"
            )

        episode_row = episodes[
            episodes["case_id"].astype(int) == case_id
        ].iloc[0]

        x, y = build_xy_path(case_rows)

        start_x = float(case_rows["start_x_m"].iloc[0])
        start_y = float(case_rows["start_y_m"].iloc[0])
        goal_x = float(case_rows["goal_x_m"].iloc[0])
        goal_y = float(case_rows["goal_y_m"].iloc[0])

        draw_sets(
            ax,
            map_df,
            xx,
            yy,
            safe_mask,
            backup_mask,
        )
        draw_start_goal(
            ax,
            start_x,
            start_y,
            goal_x,
            goal_y,
        )

        ax.plot(
            x,
            y,
            linewidth=2.4,
            zorder=10,
        )

        draw_heading_samples(
            ax,
            case_rows,
        )

        outcome = str(episode_row["outcome"])
        final_color = "green" if outcome == "success" else "red"

        ax.scatter(
            x[-1],
            y[-1],
            marker="X",
            s=95,
            linewidths=2,
            color=final_color,
            zorder=13,
        )
        path_length = float(episode_row["path_length_m"])
        duration = float(
            episode_row["episode_duration_s"]
        )
        clearance = float(
            episode_row["minimum_physical_clearance_m"]
        )

        ax.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"Outcome: {outcome}\n"
            f"Path: {path_length:.2f} m, "
            f"Time: {duration:.1f} s\n"
            f"Minimum clearance: {clearance:+.3f} m"
        )

        configure_axis(ax)

    legend_handles = [
        Patch(
            facecolor="#7aa6e8",
            alpha=0.20,
            label=r"Backup set $\mathcal{C}_B$",
        ),
        Patch(
            facecolor="#39ff88",
            alpha=0.90,
            label=r"Safe set $\mathcal{C}_S$",
        ),
        Patch(
            facecolor="#e85d68",
            alpha=0.65,
            label="Robot-clearance region",
        ),
        Patch(
            facecolor="#d3d3d3",
            edgecolor="black",
            label="Physical obstacle",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markeredgecolor="black",
            markersize=9,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markeredgecolor="black",
            markersize=13,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1.2,
            label="Straight-line reference",
        ),
        Line2D(
            [0],
            [0],
            linewidth=2.4,
            label="Robot trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markersize=9,
            label="Final position",
        ),
        Line2D(
            [0],
            [0],
            marker=">",
            linestyle="none",
            markersize=15,
            color="black",
            label="Body heading",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"Unseen-map trajectory comparison — Case {case_id}",
        fontsize=15,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    fig.savefig(
        OUT / f"trajectory_case_{case_id}_sets.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / f"trajectory_case_{case_id}_sets.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    map_df = pd.read_csv(
        ROOT / MAP_NAME / "map.csv"
    )

    method_data = {
        method: load_method_data(method)
        for method in METHODS
    }

    xx, yy, safe_mask, backup_mask = compute_set_masks(
        map_df
    )

    for case_id in CASES:
        plot_case(
            case_id,
            map_df,
            method_data,
            xx,
            yy,
            safe_mask,
            backup_mask,
        )

    print(f"Figures written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
