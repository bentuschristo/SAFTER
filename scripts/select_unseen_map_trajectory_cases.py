from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "logs/evaluation/unseen_maps/"
    "20260628_005743_seed10_100permap"
)
MAP_NAME = "shifted"

METHOD_FILES = {
    "no_cbf": ROOT / MAP_NAME / "no_cbf" / "no_cbf_episodes.csv",
    "cbf_rl": ROOT / MAP_NAME / "cbf_rl" / "cbf_rl_episodes.csv",
    "bcbf_rl": ROOT / MAP_NAME / "bcbf_rl" / "bcbf_rl_episodes.csv",
}


def main() -> None:
    frames = {}

    for method, path in METHOD_FILES.items():
        df = pd.read_csv(path)
        frames[method] = df.set_index("case_id")

    common_ids = sorted(
        set(frames["no_cbf"].index)
        & set(frames["cbf_rl"].index)
        & set(frames["bcbf_rl"].index)
    )

    rows = []

    for case_id in common_ids:
        baseline = frames["no_cbf"].loc[case_id]
        cbf = frames["cbf_rl"].loc[case_id]
        bcbf = frames["bcbf_rl"].loc[case_id]

        baseline_failed = int(baseline["success"]) == 0
        cbf_succeeded = int(cbf["success"]) == 1
        bcbf_succeeded = int(bcbf["success"]) == 1

        if baseline_failed and (cbf_succeeded or bcbf_succeeded):
            rows.append(
                {
                    "case_id": case_id,
                    "reset_seed": int(baseline["reset_seed"]),
                    "start_x": baseline["start_x"],
                    "start_y": baseline["start_y"],
                    "goal_x": baseline["goal_x"],
                    "goal_y": baseline["goal_y"],
                    "direct_path_blocked": int(
                        baseline["direct_path_blocked"]
                    ),
                    "no_cbf_outcome": baseline["outcome"],
                    "cbf_rl_outcome": cbf["outcome"],
                    "bcbf_rl_outcome": bcbf["outcome"],
                    "no_cbf_min_clearance": baseline[
                        "minimum_physical_clearance_m"
                    ],
                    "cbf_rl_min_clearance": cbf[
                        "minimum_physical_clearance_m"
                    ],
                    "bcbf_rl_min_clearance": bcbf[
                        "minimum_physical_clearance_m"
                    ],
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        print("No baseline-failure/safety-success cases were found.")
        return

    result = result.sort_values(
        [
            "direct_path_blocked",
            "no_cbf_min_clearance",
        ],
        ascending=[False, True],
    )

    output = ROOT / "trajectory_comparison_candidates.csv"
    result.to_csv(output, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
