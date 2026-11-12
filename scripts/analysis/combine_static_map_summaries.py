#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

vanilla_cbf_dir = Path(
    "logs/evaluation/laser_unseen_maps_structured_180_r61_vanilla_cbfcd_50case_pc/"
    "20260731_104129_seed42_50permap"
)

bcbf_dir = Path(
    "logs/evaluation/laser_unseen_maps_structured_180_r61_bcbf_50case_server/"
    "20260731_145558_seed42_50permap"
)

paths = [
    vanilla_cbf_dir / "combined_summary_by_map.csv",
    bcbf_dir / "combined_summary_by_map.csv",
]

dfs = []
for p in paths:
    if not p.is_file():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    print(f"\nLoaded: {p}")
    print("Columns:", df.columns.tolist())

    # Normalize possible map-column names.
    map_candidates = ["map", "map_name", "Map", "scenario", "layout"]
    map_col = next((c for c in map_candidates if c in df.columns), None)
    if map_col is None:
        raise KeyError(
            f"No map column found in {p}. Available columns are: {df.columns.tolist()}"
        )
    if map_col != "map":
        df = df.rename(columns={map_col: "map"})

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

method_map = {
    "vanilla": "No CBF",
    "no_cbf": "No CBF",
    "cbf_cd_cf": "CBF-CD-CF",
    "bcbf_cf": "bCBF-CF",
}
df["Method"] = df["method"].map(method_map).fillna(df["method"])

map_order = ["slalom", "bottleneck", "clustered", "corridor", "shifted"]
method_order = ["No CBF", "CBF-CD-CF", "bCBF-CF"]

df["map_order"] = df["map"].map({m: i for i, m in enumerate(map_order)})
df["method_order"] = df["Method"].map({m: i for i, m in enumerate(method_order)})
df = df.sort_values(["map_order", "method_order"]).drop(
    columns=["map_order", "method_order"]
)

out_dir = Path("logs/evaluation/laser_unseen_maps_structured_180_r61_combined_50case")
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "combined_summary_by_map_all_methods_full.csv", index=False)

paper = df[
    [
        "map",
        "Method",
        "cases",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "fall_rate",
        "mean_return",
        "mean_success_path_length_m",
        "mean_success_goal_time_s",
        "mean_minimum_clearance_m",
    ]
].copy()

for c in ["success_rate", "collision_rate", "timeout_rate", "fall_rate"]:
    paper[c] = 100.0 * paper[c]

paper = paper.rename(
    columns={
        "map": "Map",
        "cases": "Cases",
        "success_rate": "Success (%)",
        "collision_rate": "Collision (%)",
        "timeout_rate": "Timeout (%)",
        "fall_rate": "Fall (%)",
        "mean_return": "Return",
        "mean_success_path_length_m": "Path (m)",
        "mean_success_goal_time_s": "Time (s)",
        "mean_minimum_clearance_m": "Clearance (m)",
    }
)

paper.to_csv(out_dir / "combined_summary_by_map_paper.csv", index=False)

# A compact table for main paper.
compact = paper[
    ["Map", "Method", "Success (%)", "Collision (%)", "Return", "Clearance (m)"]
].copy()
compact.to_csv(out_dir / "combined_summary_by_map_compact_paper.csv", index=False)

latex = compact.to_latex(
    index=False,
    float_format="%.3f",
    escape=False,
    column_format="llrrrr",
)
(out_dir / "combined_summary_by_map_compact_paper_table.tex").write_text(latex)

print("\nCompact per-map table:")
print(compact.to_string(index=False))

print("\nSaved:")
print(out_dir / "combined_summary_by_map_all_methods_full.csv")
print(out_dir / "combined_summary_by_map_paper.csv")
print(out_dir / "combined_summary_by_map_compact_paper.csv")
print(out_dir / "combined_summary_by_map_compact_paper_table.tex")
