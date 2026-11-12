from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("logs/evaluation/go2_safe_navigation")
OUTPUT = ROOT / "plots"
OUTPUT.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "raw_sac": "Raw SAC",
    "baseline_sac_deployment_cbf": "SAC + deployment CBF",
    "integrated_sac_cbf_cf": "Integrated SAC-CBF-CF",
}
METHOD_ORDER = [
    "raw_sac",
    "baseline_sac_deployment_cbf",
    "integrated_sac_cbf_cf",
]
SCENARIO_ORDER = ["easy", "medium", "hard"]

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


def load_latest_summaries() -> pd.DataFrame:
    files = sorted(ROOT.glob("*100ep_summary.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No *100ep_summary.csv files found.")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["_mtime"] = path.stat().st_mtime
        frame["_source"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    required = {"method", "scenario"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    return (
        df.sort_values("_mtime")
        .drop_duplicates(["method", "scenario"], keep="last")
        .copy()
    )


def grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str, *, percentage: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for i, method in enumerate(METHOD_ORDER):
        sub = df[df["method"] == method].set_index("scenario")
        vals = []
        for scenario in SCENARIO_ORDER:
            value = float(sub.loc[scenario, metric]) if scenario in sub.index else np.nan
            vals.append(value * 100.0 if percentage else value)
        ax.bar(x + (i - 1) * width, vals, width, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SCENARIO_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=250)
    plt.close(fig)


def grouped_bar_error(df: pd.DataFrame, mean_metric: str, std_metric: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for i, method in enumerate(METHOD_ORDER):
        sub = df[df["method"] == method].set_index("scenario")
        means, stds = [], []
        for scenario in SCENARIO_ORDER:
            if scenario in sub.index:
                means.append(float(sub.loc[scenario, mean_metric]))
                stds.append(float(sub.loc[scenario, std_metric]))
            else:
                means.append(np.nan)
                stds.append(0.0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SCENARIO_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=250)
    plt.close(fig)


def latest_file(pattern: str) -> Path | None:
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def representative_episode(episodes: pd.DataFrame) -> int:
    success = episodes[episodes["success"].astype(int) == 1].copy()
    if not success.empty:
        median_path = success["path_length_m"].median()
        idx = (success["path_length_m"] - median_path).abs().idxmin()
        return int(success.loc[idx, "episode"])
    if "episode_steps" in episodes.columns:
        return int(episodes.loc[episodes["episode_steps"].idxmax(), "episode"])
    return int(episodes.iloc[0]["episode"])


def plot_trajectories(scenario: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    for ox, oy, radius in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), radius, fill=False, linewidth=1.4))
        ax.add_patch(plt.Circle((ox, oy), radius + ROBOT_RADIUS, fill=False, linestyle="--", linewidth=0.8))

    start = goal = None
    for method in METHOD_ORDER:
        ep_file = latest_file(f"*_{method}_{scenario}_100ep_episodes.csv")
        tr_file = latest_file(f"*_{method}_{scenario}_100ep_trajectory.csv")
        if ep_file is None or tr_file is None:
            print(f"Warning: missing files for {method}/{scenario}")
            continue

        episodes = pd.read_csv(ep_file)
        traj = pd.read_csv(tr_file)
        ep_no = representative_episode(episodes)
        ep = traj[traj["episode"].astype(int) == ep_no].copy()
        if ep.empty:
            continue

        ax.plot(ep["post_x_m"], ep["post_y_m"], linewidth=2, label=f"{METHOD_LABELS[method]} (ep {ep_no})")
        if start is None:
            start = (float(ep["post_x_m"].iloc[0]), float(ep["post_y_m"].iloc[0]))
            goal = (float(ep["goal_x_m"].iloc[0]), float(ep["goal_y_m"].iloc[0]))

    if start is not None:
        ax.scatter(*start, marker="o", s=80, label="Start")
    if goal is not None:
        ax.scatter(*goal, marker="*", s=150, label="Goal")

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Representative trajectories — {scenario.capitalize()}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"trajectory_{scenario}.png", dpi=250)
    plt.close(fig)


def main() -> None:
    df = load_latest_summaries()

    columns = [
        "method", "scenario", "episodes", "success_rate", "collision_rate", "timeout_rate",
        "mean_success_path_length_m", "std_success_path_length_m", "mean_goal_time_s", "std_goal_time_s",
    ]
    columns = [c for c in columns if c in df.columns]
    combined = df[columns].sort_values(["scenario", "method"])
    combined.to_csv(ROOT / "combined_100ep_results.csv", index=False)
    print(combined.to_string(index=False))

    grouped_bar(df, "success_rate", "Success rate [%]", "Goal-reaching success rate", "success_rate.png", percentage=True)
    grouped_bar(df, "collision_rate", "Collision rate [%]", "Geometric collision rate", "collision_rate.png", percentage=True)
    grouped_bar(df, "timeout_rate", "Timeout rate [%]", "Episode timeout rate", "timeout_rate.png", percentage=True)
    grouped_bar_error(df, "mean_success_path_length_m", "std_success_path_length_m", "Path length [m]", "Successful-episode path length", "successful_path_length.png")
    grouped_bar_error(df, "mean_goal_time_s", "std_goal_time_s", "Time [s]", "Successful-episode goal-reaching time", "goal_time.png")

    for scenario in SCENARIO_ORDER:
        plot_trajectories(scenario)

    print(f"\nSaved plots to: {OUTPUT}")


if __name__ == "__main__":
    main()
