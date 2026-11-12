from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("logs/evaluation/go2_safe_navigation")
OUTPUT = ROOT / "plots_final_with_bcbf"
OUTPUT.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "raw_sac": "Raw SAC",
    "baseline_sac_deployment_cbf": "SAC + deployment CBF",
    "integrated_sac_cbf_cf": "Integrated SAC-CBF-CF",
    "integrated_sac_bcbf_cf": "Integrated SAC-bCBF-LSE-CF",
}

METHOD_ORDER = [
    "raw_sac",
    "baseline_sac_deployment_cbf",
    "integrated_sac_cbf_cf",
    "integrated_sac_bcbf_cf",
]

SCENARIOS = ["easy", "medium", "hard"]

SCENARIO_LABELS = {
    "easy": "Easy",
    "medium": "Shifted medium",
    "hard": "Hard",
}

# Fixed obstacle map used by the evaluation environment.
OBSTACLES = [
    (-3.0, -2.0, 0.55),
    (-1.5, 2.2, 0.45),
    (1.2, -1.0, 0.60),
    (3.0, 2.3, 0.50),
    (0.3, 3.8, 0.40),
    (-5.0, 4.5, 0.50),
    (-5.0, -4.5, 0.45),
    (5.0, 4.5, 0.55),
    (5.0, -4.5, 0.50),
]

ROBOT_RADIUS = 0.30


def latest_file(pattern: str) -> Path:
    files = list(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No evaluation file matches:\n  {ROOT / pattern}"
        )
    return max(files, key=lambda p: p.stat().st_mtime)


def summary_pattern(method: str, scenario: str) -> str:
    if scenario == "medium":
        mapped = {
            "raw_sac": "raw_sac_shifted_medium",
            "baseline_sac_deployment_cbf":
                "baseline_sac_deployment_cbf_shifted_medium",
            "integrated_sac_cbf_cf":
                "integrated_sac_cbf_cf_shifted_medium",
            "integrated_sac_bcbf_cf":
                "integrated_sac_bcbf_cf_shifted_medium",
        }[method]

        return f"*_{mapped}_medium_100ep_summary.csv"

    return f"*_{method}_{scenario}_100ep_summary.csv"


def episode_pattern(method: str, scenario: str) -> str:
    return summary_pattern(method, scenario).replace(
        "_summary.csv",
        "_episodes.csv",
    )


def trajectory_pattern(method: str, scenario: str) -> str:
    return summary_pattern(method, scenario).replace(
        "_summary.csv",
        "_trajectory.csv",
    )


def load_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for scenario in SCENARIOS:
        for method in METHOD_ORDER:
            path = latest_file(summary_pattern(method, scenario))
            frame = pd.read_csv(path)

            # Normalize names so all methods share common plotting labels.
            frame["method"] = method
            frame["scenario"] = scenario
            frame["_source"] = path.name
            frames.append(frame)

            print(
                f"[LOAD] {method:32s} {scenario:6s} <- {path.name}"
            )

    results = pd.concat(frames, ignore_index=True)

    results["method"] = pd.Categorical(
        results["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    results["scenario"] = pd.Categorical(
        results["scenario"],
        categories=SCENARIOS,
        ordered=True,
    )

    return results.sort_values(["scenario", "method"])


def grouped_rate_plot(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    x = np.arange(len(SCENARIOS))
    width = 0.19

    for i, method in enumerate(METHOD_ORDER):
        subset = (
            results[results["method"] == method]
            .set_index("scenario")
        )

        values = [
            100.0 * float(subset.loc[scenario, metric])
            for scenario in SCENARIOS
        ]

        offset = (
            i - (len(METHOD_ORDER) - 1) / 2.0
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS[method],
        )

        ax.bar_label(
            bars,
            labels=[f"{v:.0f}" for v in values],
            padding=2,
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIOS]
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 110.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


def grouped_mean_std_plot(
    results: pd.DataFrame,
    mean_metric: str,
    std_metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    x = np.arange(len(SCENARIOS))
    width = 0.19

    for i, method in enumerate(METHOD_ORDER):
        subset = (
            results[results["method"] == method]
            .set_index("scenario")
        )

        means = []
        stds = []

        for scenario in SCENARIOS:
            mean_value = subset.loc[scenario, mean_metric]
            std_value = subset.loc[scenario, std_metric]

            means.append(
                float(mean_value)
                if pd.notna(mean_value)
                else np.nan
            )
            stds.append(
                float(std_value)
                if pd.notna(std_value)
                else 0.0
            )

        offset = (
            i - (len(METHOD_ORDER) - 1) / 2.0
        ) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABELS[s] for s in SCENARIOS]
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


def representative_episode(episodes: pd.DataFrame) -> int:
    successes = episodes[
        episodes["success"].astype(int) == 1
    ].copy()

    if not successes.empty:
        median_path = successes["path_length_m"].median()
        index = (
            successes["path_length_m"] - median_path
        ).abs().idxmin()
        return int(successes.loc[index, "episode"])

    collisions = episodes[
        episodes["collision"].astype(int) == 1
    ].copy()

    if not collisions.empty:
        median_path = collisions["path_length_m"].median()
        index = (
            collisions["path_length_m"] - median_path
        ).abs().idxmin()
        return int(collisions.loc[index, "episode"])

    if "episode_steps" in episodes.columns:
        index = episodes["episode_steps"].idxmax()
        return int(episodes.loc[index, "episode"])

    return int(episodes.iloc[0]["episode"])


def plot_trajectories(scenario: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2))

    for ox, oy, radius in OBSTACLES:
        ax.add_patch(
            plt.Circle(
                (ox, oy),
                radius,
                fill=False,
                linewidth=1.4,
            )
        )
        ax.add_patch(
            plt.Circle(
                (ox, oy),
                radius + ROBOT_RADIUS,
                fill=False,
                linestyle="--",
                linewidth=0.8,
            )
        )

    start = None
    goal = None

    for method in METHOD_ORDER:
        episodes_path = latest_file(
            episode_pattern(method, scenario)
        )
        trajectory_path = latest_file(
            trajectory_pattern(method, scenario)
        )

        episodes = pd.read_csv(episodes_path)
        trajectory = pd.read_csv(trajectory_path)

        episode_number = representative_episode(episodes)

        episode_trajectory = trajectory[
            trajectory["episode"].astype(int)
            == episode_number
        ].copy()

        if episode_trajectory.empty:
            print(
                f"[WARN] Empty trajectory for "
                f"{method}/{scenario}/episode {episode_number}"
            )
            continue

        terminal_row = episodes[
            episodes["episode"].astype(int)
            == episode_number
        ].iloc[0]

        outcome = str(terminal_row["outcome"])

        ax.plot(
            episode_trajectory["post_x_m"],
            episode_trajectory["post_y_m"],
            linewidth=2,
            label=(
                f"{METHOD_LABELS[method]} "
                f"(ep. {episode_number}, {outcome})"
            ),
        )

        if start is None:
            start = (
                float(episode_trajectory["post_x_m"].iloc[0]),
                float(episode_trajectory["post_y_m"].iloc[0]),
            )
            goal = (
                float(episode_trajectory["goal_x_m"].iloc[0]),
                float(episode_trajectory["goal_y_m"].iloc[0]),
            )

    if start is not None:
        ax.scatter(
            *start,
            marker="o",
            s=80,
            label="Start",
            zorder=5,
        )

    if goal is not None:
        ax.scatter(
            *goal,
            marker="*",
            s=170,
            label="Goal",
            zorder=5,
        )

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Representative trajectories — "
        f"{SCENARIO_LABELS[scenario]}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"trajectory_{scenario}.png",
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    results = load_results()

    output_columns = [
        "method",
        "scenario",
        "episodes",
        "successes",
        "collisions",
        "timeouts",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_success_path_length_m",
        "std_success_path_length_m",
        "mean_goal_time_s",
        "std_goal_time_s",
        "mean_all_episode_path_length_m",
        "checkpoint",
        "_source",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in results.columns
    ]

    combined = results[output_columns].copy()
    combined.to_csv(
        OUTPUT / "combined_final_results_with_bcbf.csv",
        index=False,
    )

    display_columns = [
        "method",
        "scenario",
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_success_path_length_m",
        "mean_goal_time_s",
    ]

    print("\nCombined results")
    print("-" * 110)
    print(
        results[display_columns].to_string(
            index=False,
        )
    )

    grouped_rate_plot(
        results,
        "success_rate",
        "Success rate [%]",
        "Goal-reaching success rate",
        "success_rate.png",
    )

    grouped_rate_plot(
        results,
        "collision_rate",
        "Collision rate [%]",
        "Geometric collision rate",
        "collision_rate.png",
    )

    grouped_rate_plot(
        results,
        "timeout_rate",
        "Timeout rate [%]",
        "Episode timeout rate",
        "timeout_rate.png",
    )

    grouped_mean_std_plot(
        results,
        "mean_success_path_length_m",
        "std_success_path_length_m",
        "Path length [m]",
        "Successful-episode path length",
        "successful_path_length.png",
    )

    grouped_mean_std_plot(
        results,
        "mean_goal_time_s",
        "std_goal_time_s",
        "Time [s]",
        "Successful-episode goal-reaching time",
        "goal_time.png",
    )

    for scenario in SCENARIOS:
        plot_trajectories(scenario)

    print(f"\nSaved combined CSV and plots to:\n  {OUTPUT}")


if __name__ == "__main__":
    main()
