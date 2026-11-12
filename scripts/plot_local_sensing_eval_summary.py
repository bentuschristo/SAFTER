from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "no_cbf": "No-CBF",
    "cbf_rl": "CBF-CF",
    "bcbf_rl": "bCBF-CF",
}

METHOD_ORDER = ["no_cbf", "cbf_rl", "bcbf_rl"]
RADIUS_ORDER = ["R2", "R3", "R4", "R5", "full"]
RADIUS_LABELS = {
    "R2": "2 m",
    "R3": "3 m",
    "R4": "4 m",
    "R5": "5 m",
    "full": "Full",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["method"].isin(METHOD_ORDER)]
    df = df[df["sensing_radius_label"].isin(RADIUS_ORDER)]

    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["radius_label"] = pd.Categorical(
        df["sensing_radius_label"],
        categories=RADIUS_ORDER,
        ordered=True,
    )
    df["radius_plot_label"] = df["sensing_radius_label"].map(RADIUS_LABELS)
    return df


def plot_metric_over_sensing(df, metric, ylabel, title, out_path, percent=False):
    agg = (
        df.groupby(["sensing_radius_label", "method"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    x = np.arange(len(RADIUS_ORDER))
    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].set_index("sensing_radius_label")
        y = [sub.loc[r, metric] if r in sub.index else np.nan for r in RADIUS_ORDER]
        if percent:
            y = [100.0 * v for v in y]
        ax.plot(x, y, marker="o", linewidth=2, label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([RADIUS_LABELS[r] for r in RADIUS_ORDER])
    ax.set_xlabel("Sensing radius")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_per_map_success(df, out_path):
    maps = sorted(df["map_name"].unique())
    fig, axes = plt.subplots(len(maps), 1, figsize=(7.2, 2.3 * len(maps)), sharex=True)

    if len(maps) == 1:
        axes = [axes]

    x = np.arange(len(RADIUS_ORDER))

    for ax, map_name in zip(axes, maps):
        mdf = df[df["map_name"] == map_name]
        for method in METHOD_ORDER:
            sub = mdf[mdf["method"] == method].set_index("sensing_radius_label")
            y = [100.0 * sub.loc[r, "success_rate"] if r in sub.index else np.nan for r in RADIUS_ORDER]
            ax.plot(x, y, marker="o", linewidth=2, label=METHOD_LABELS[method])
        ax.set_ylabel("Success (%)")
        ax.set_title(map_name)
        ax.grid(True, alpha=0.35)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([RADIUS_LABELS[r] for r in RADIUS_ORDER])
    axes[-1].set_xlabel("Sensing radius")
    axes[0].legend(frameon=True, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_bar_by_method(df, metric, ylabel, title, out_path, percent=False):
    agg = df.groupby("method", observed=True)[metric].mean().reindex(METHOD_ORDER)

    values = agg.values
    if percent:
        values = 100.0 * values

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar([METHOD_LABELS[m] for m in METHOD_ORDER], values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}" if percent else f"{v:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = prepare(df)

    # Overall sensing-radius trends.
    plot_metric_over_sensing(
        df,
        metric="success_rate",
        ylabel="Success rate (%)",
        title="Success rate vs sensing radius",
        out_path=args.out_dir / "success_rate_vs_sensing_radius",
        percent=True,
    )

    plot_metric_over_sensing(
        df,
        metric="collision_rate",
        ylabel="Collision rate (%)",
        title="Collision rate vs sensing radius",
        out_path=args.out_dir / "collision_rate_vs_sensing_radius",
        percent=True,
    )

    plot_metric_over_sensing(
        df,
        metric="mean_minimum_clearance_m",
        ylabel="Mean minimum clearance (m)",
        title="Minimum clearance vs sensing radius",
        out_path=args.out_dir / "minimum_clearance_vs_sensing_radius",
        percent=False,
    )

    plot_metric_over_sensing(
        df,
        metric="mean_success_goal_time_s",
        ylabel="Goal time for successful trials (s)",
        title="Goal time vs sensing radius",
        out_path=args.out_dir / "goal_time_vs_sensing_radius",
        percent=False,
    )

    plot_metric_over_sensing(
        df,
        metric="mean_filter_active_fraction",
        ylabel="Filter active fraction",
        title="Filter intervention vs sensing radius",
        out_path=args.out_dir / "filter_active_fraction_vs_sensing_radius",
        percent=False,
    )

    plot_metric_over_sensing(
        df,
        metric="mean_visible_obstacles",
        ylabel="Mean visible obstacles",
        title="Visible obstacles vs sensing radius",
        out_path=args.out_dir / "visible_obstacles_vs_sensing_radius",
        percent=False,
    )

    # Per-map success panel.
    plot_per_map_success(
        df,
        out_path=args.out_dir / "per_map_success_rate_vs_sensing_radius",
    )

    # Aggregate bars across all maps and sensing radii.
    plot_metric_bar_by_method(
        df,
        metric="success_rate",
        ylabel="Mean success rate (%)",
        title="Aggregate success rate",
        out_path=args.out_dir / "aggregate_success_rate_by_method",
        percent=True,
    )

    plot_metric_bar_by_method(
        df,
        metric="collision_rate",
        ylabel="Mean collision rate (%)",
        title="Aggregate collision rate",
        out_path=args.out_dir / "aggregate_collision_rate_by_method",
        percent=True,
    )

    # Save aggregate table too.
    summary = (
        df.groupby("method", observed=True)
        .agg(
            success_rate=("success_rate", "mean"),
            collision_rate=("collision_rate", "mean"),
            timeout_rate=("timeout_rate", "mean"),
            mean_minimum_clearance_m=("mean_minimum_clearance_m", "mean"),
            mean_success_goal_time_s=("mean_success_goal_time_s", "mean"),
            mean_filter_active_fraction=("mean_filter_active_fraction", "mean"),
            mean_command_correction_norm=("mean_command_correction_norm", "mean"),
        )
        .reindex(METHOD_ORDER)
    )
    summary.index = [METHOD_LABELS[m] for m in METHOD_ORDER]
    summary.to_csv(args.out_dir / "aggregate_summary_by_method.csv")

    print("Saved plots to:", args.out_dir)
    print(summary)


if __name__ == "__main__":
    main()
