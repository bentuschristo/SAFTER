from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("logs/evaluation/go2_safe_navigation")
OUTPUT = ROOT / "plots_final_shifted_medium"
OUTPUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "raw_sac",
    "baseline_sac_deployment_cbf",
    "integrated_sac_cbf_cf",
]

METHOD_LABELS = {
    "raw_sac": "Raw SAC",
    "baseline_sac_deployment_cbf": "SAC + deployment CBF",
    "integrated_sac_cbf_cf": "Integrated SAC-CBF-CF",
}

SCENARIOS = ["easy", "medium", "hard"]


def newest(pattern: str) -> Path:
    files = sorted(
        ROOT.glob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No file found for: {pattern}")
    return files[-1]


def summary_path(method: str, scenario: str) -> Path:
    if scenario == "medium":
        mapped = {
            "raw_sac": "raw_sac_shifted_medium",
            "baseline_sac_deployment_cbf":
                "baseline_sac_deployment_cbf_shifted_medium",
            "integrated_sac_cbf_cf":
                "integrated_sac_cbf_cf_shifted_medium",
        }[method]

        return newest(
            f"*_{mapped}_medium_100ep_summary.csv"
        )

    return newest(
        f"*_{method}_{scenario}_100ep_summary.csv"
    )


frames = []

for scenario in SCENARIOS:
    for method in METHODS:
        path = summary_path(method, scenario)
        frame = pd.read_csv(path)

        # Normalize names so the three scenarios compare under common labels.
        frame["method"] = method
        frame["scenario"] = scenario
        frame["source_file"] = path.name
        frames.append(frame)

results = pd.concat(frames, ignore_index=True)

results["scenario"] = pd.Categorical(
    results["scenario"],
    categories=SCENARIOS,
    ordered=True,
)

results["method"] = pd.Categorical(
    results["method"],
    categories=METHODS,
    ordered=True,
)

results = results.sort_values(["scenario", "method"])

columns = [
    "method",
    "scenario",
    "episodes",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_success_path_length_m",
    "std_success_path_length_m",
    "mean_goal_time_s",
    "std_goal_time_s",
    "source_file",
]

combined_path = OUTPUT / "combined_final_results.csv"
results[columns].to_csv(combined_path, index=False)

print("\nFinal combined results:\n")
print(results[columns[:-1]].to_string(index=False))
print(f"\nSaved table: {combined_path}")


def rate_plot(
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(SCENARIOS))
    width = 0.24

    for index, method in enumerate(METHODS):
        subset = results[results["method"] == method].set_index("scenario")
        values = [
            100.0 * float(subset.loc[scenario, metric])
            for scenario in SCENARIOS
        ]

        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
        )

        ax.bar_label(
            bars,
            labels=[f"{value:.0f}" for value in values],
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Shifted medium", "Hard"])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 108.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=250)
    plt.close(fig)


def successful_metric_plot(
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(SCENARIOS))
    width = 0.24

    for index, method in enumerate(METHODS):
        subset = results[results["method"] == method].set_index("scenario")

        means = []
        errors = []

        for scenario in SCENARIOS:
            mean = subset.loc[scenario, mean_column]
            std = subset.loc[scenario, std_column]

            means.append(float(mean) if pd.notna(mean) else np.nan)
            errors.append(float(std) if pd.notna(std) else 0.0)

        ax.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=METHOD_LABELS[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Shifted medium", "Hard"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=250)
    plt.close(fig)


rate_plot(
    "success_rate",
    "Success rate [%]",
    "Goal-reaching success rate",
    "success_rate.png",
)

rate_plot(
    "collision_rate",
    "Collision rate [%]",
    "Geometric collision rate",
    "collision_rate.png",
)

rate_plot(
    "timeout_rate",
    "Timeout rate [%]",
    "Episode timeout rate",
    "timeout_rate.png",
)

successful_metric_plot(
    "mean_success_path_length_m",
    "std_success_path_length_m",
    "Path length [m]",
    "Successful-episode path length",
    "successful_path_length.png",
)

successful_metric_plot(
    "mean_goal_time_s",
    "std_goal_time_s",
    "Time [s]",
    "Successful-episode goal-reaching time",
    "goal_time.png",
)

print(f"Saved plots under: {OUTPUT}")
