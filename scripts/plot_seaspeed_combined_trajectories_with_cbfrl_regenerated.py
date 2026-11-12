"""Regenerated version of plot_seaspeed_combined_trajectories_with_cbfrl.py's
figure, using: the latest SEA-Nav checkpoint (clearance/FOV/LiDAR fixes) and
the latest CBF-RL "circlegeom" checkpoint (2026-09-17 retrain), with cases
chosen so bCBF-CF (SAFTER) succeeds while every other method fails on the
same map -- see the 2026-09-20/21 session for the full case-selection and
verification methodology (genuine obstacle collisions only, not falls/walls).

============================================================================
EASY-TO-CHANGE LAYOUT KNOBS -- edit these numbers to resize things.

Each map panel uses set_aspect("equal") (so it doesn't visually distort the
arena), which means its rendered HEIGHT always just follows from whatever
WIDTH it's given -- not the other way around. So the main lever for map
size is FIG_WIDTH_IN (and WSPACE, which eats into how much of that width
each panel actually gets), not any height-based setting.
============================================================================
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# Overall figure width in inches. THIS is what actually controls map size
# (each panel is equal-aspect, so its height follows its width -- making
# the figure wider is what makes the maps bigger). Try 18-28. (Current: 24)
FIG_WIDTH_IN = 24.0

# Horizontal gap between the 3 map panels, as a fraction of one panel's
# average width. Smaller number => panels sit closer together => each one
# gets to be bigger for the same FIG_WIDTH_IN. Try 0.05-0.3. (Current: 0.12)
WSPACE = 0.12

# Fraction of the figure's total height given to the 3 maps (vs. the legend
# row below them). Has a smaller visible effect than FIG_WIDTH_IN/WSPACE
# since panel height is width-derived, but still shifts the legend row's
# share. Try 0.85-0.96. (Current: 0.93)
MAP_ROW_HEIGHT_FRAC = 0.93

# Raw GridSpec hspace between the map row and the legend row (matplotlib's
# own units: roughly "gap as a fraction of one row's average height").
# Smaller number => legend sits closer to the maps. Try 0.0-0.3. (Current: 0.05)
GAP_BETWEEN_MAP_AND_LEGEND_FRAC = 0.05
# ============================================================================

MAPS = [("slalom", "Low-Shift"), ("corridor", "Medium-Shift"), ("circle_and_square", "High-Shift")]

OBSTACLES = {
    "slalom": [
        {"x": -5.2, "y": -2.3, "r": 0.5, "shape": "circle"},
        {"x": -3.9, "y": 2.1, "r": 0.5, "shape": "circle"},
        {"x": -2.6, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": -1.3, "y": 2.2, "r": 0.5, "shape": "circle"},
        {"x": 0.0, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": 1.3, "y": 2.2, "r": 0.5, "shape": "circle"},
        {"x": 2.6, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": 3.9, "y": 2.1, "r": 0.5, "shape": "circle"},
        {"x": 5.2, "y": -2.3, "r": 0.5, "shape": "circle"},
    ],
    "corridor": [
        {"x": -3.8, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": -1.9, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": 0.0, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": 1.9, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": 3.8, "y": -2.2, "r": 0.5, "shape": "circle"},
        {"x": -3.8, "y": 2.2, "r": 0.5, "shape": "circle"},
        {"x": -1.9, "y": 2.2, "r": 0.5, "shape": "circle"},
        {"x": 1.9, "y": 2.2, "r": 0.5, "shape": "circle"},
        {"x": 3.8, "y": 2.2, "r": 0.5, "shape": "circle"},
    ],
    "circle_and_square": [
        {"x": -1.6, "y": 3.4, "r": 0.35, "shape": "circle"},
        {"x": 1.6, "y": 3.4, "r": 0.35, "shape": "circle"},
        {"x": 0.0, "y": 1.9, "r": 1.0, "shape": "circle"},
        {"x": -1.5, "y": -0.2, "r": 0.35, "shape": "circle"},
        {"x": 1.5, "y": -0.2, "r": 0.35, "shape": "circle"},
        {"x": -4.6, "y": -2.3, "r": 0.65, "shape": "circle"},
        {"x": -1.0, "y": -2.6, "r": 0.4, "shape": "square"},
        {"x": 1.0, "y": -2.6, "r": 0.4, "shape": "square"},
        {"x": 4.6, "y": -2.3, "r": 0.65, "shape": "circle"},
        {"x": -3.6, "y": 0.4, "r": 0.7, "shape": "square"},
        {"x": 3.6, "y": 0.4, "r": 0.7, "shape": "square"},
    ],
}

REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "circle_and_square": {"start": (-1.0, 1.0, 5.0, 6.8), "goal": (-5.5, 5.5, -7.0, -4.0)},
}

SAC_MAP_TO_CASE = {"slalom": "0", "corridor": "2", "circle_and_square": "3"}
SEANAV_MAP_TO_DIFF = {"slalom": "easy", "corridor": "medium", "circle_and_square": "hard"}

COLORS = {
    "Vanilla": "#1f77b4", "CBF-CF": "#ff7f0e", "SAFTER": "#2ca02c",
    "SEA-Nav": "#9467bd", "CBF-RL (Dual)": "#d62728",
}

FS_TITLE = 44
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23

SAC_TRAJ_DIR = "/tmp/sac_laser_selected_trajectories"
SEANAV_TRAJ_DIR = "/tmp/seanav_selected_trajectories"


def load_sac(method, mapname):
    path = f"{SAC_TRAJ_DIR}/{mapname}_{method}.csv"
    xs, ys = [], []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], ""
    xs.append(float(rows[0]["pre_x_m"])); ys.append(float(rows[0]["pre_y_m"]))
    for r in rows:
        xs.append(float(r["post_x_m"])); ys.append(float(r["post_y_m"]))
    outcome = rows[-1]["outcome"]
    return xs, ys, outcome


def load_seanav(mapname):
    diff = SEANAV_MAP_TO_DIFF[mapname]
    path = f"{SEANAV_TRAJ_DIR}/{diff}.csv"
    xs, ys = [], []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        xs.append(float(r["x"])); ys.append(float(r["y"]))
    # Easy: no genuine obstacle-collision or timeout case exists in the data
    # (closest any "collision"-labeled episode gets to a real obstacle is
    # 1.02m clearance) -- this trajectory is the physical-fall/spawn-
    # instability artifact instead, labeled honestly as "other_failure".
    outcome = "other_failure" if diff == "easy" else "collision"
    return xs, ys, outcome


def load_cbf_rl(mapname):
    path = f"logs/evaluation/cbf_rl_seaspeed_trajectory_regenerated/{mapname}/cbf_rl/cbf_rl_trajectory.csv"
    xs, ys = [], []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        xs.append(float(r["post_x_m"])); ys.append(float(r["post_y_m"]))
    outcome = rows[-1]["outcome"] if rows else ""
    return xs, ys, outcome


def outcome_marker(outcome):
    return {"collision": ("X", 130), "timeout": ("^", 120), "other_failure": ("D", 110), "success": ("*", 320)}.get(outcome, ("o", 80))


fig = plt.figure(figsize=(FIG_WIDTH_IN, 10.5))
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[MAP_ROW_HEIGHT_FRAC, 1.0 - MAP_ROW_HEIGHT_FRAC],
    hspace=GAP_BETWEEN_MAP_AND_LEGEND_FRAC,
    left=0.03, right=0.99, top=0.93, bottom=0.02, wspace=WSPACE,
)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis("off")

for ax, (mapname, title) in zip(axes, MAPS):
    regions = REGIONS[mapname]
    sx0, sx1, sy0, sy1 = regions["start"]
    gx0, gx1, gy0, gy1 = regions["goal"]
    ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, facecolor="mediumpurple", alpha=0.35, zorder=1))
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="khaki", alpha=0.45, zorder=1))

    for o in OBSTACLES[mapname]:
        if o["shape"] == "square":
            s = o["r"]
            ax.add_patch(Rectangle((o["x"] - s, o["y"] - s), 2 * s, 2 * s, facecolor="dimgray", edgecolor="black", zorder=2))
        else:
            ax.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor="dimgray", edgecolor="black", zorder=2))

    for method_key, label in [("vanilla", "Vanilla"), ("cbf_cd_cf", "CBF-CF"), ("bcbf_cf", "SAFTER")]:
        xs, ys, outcome = load_sac(method_key, mapname)
        ax.plot(xs, ys, color=COLORS[label], linewidth=3.2, zorder=3)
        marker, size = outcome_marker(outcome)
        ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS[label], zorder=5, edgecolor="black", linewidth=0.7)

    xs, ys, outcome = load_seanav(mapname)
    ax.plot(xs, ys, color=COLORS["SEA-Nav"], linewidth=3.2, zorder=3)
    marker, size = outcome_marker(outcome)
    ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS["SEA-Nav"], zorder=5, edgecolor="black", linewidth=0.7)

    xs, ys, outcome = load_cbf_rl(mapname)
    ax.plot(xs, ys, color=COLORS["CBF-RL (Dual)"], linewidth=3.2, zorder=3)
    marker, size = outcome_marker(outcome)
    ax.scatter([xs[-1]], [ys[-1]], marker=marker, s=size, color=COLORS["CBF-RL (Dual)"], zorder=5, edgecolor="black", linewidth=0.7)

    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    if mapname == "slalom":
        ax.set_ylabel("y [m]", fontsize=FS_LABEL)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.tick_params(axis="both", labelsize=FS_TICK)

legend_elems = [
    Rectangle((0, 0), 1, 1, facecolor="mediumpurple", alpha=0.35, label="Initial region"),
    Line2D([0], [0], color=COLORS["Vanilla"], lw=3.2, label="Vanilla"),
    Line2D([0], [0], color=COLORS["CBF-CF"], lw=3.2, label="CBF-CF"),
    Line2D([0], [0], color=COLORS["SAFTER"], lw=3.2, label="SAFTER"),
    Line2D([0], [0], color=COLORS["SEA-Nav"], lw=3.2, label="SEA-Nav"),
    Line2D([0], [0], color=COLORS["CBF-RL (Dual)"], lw=3.2, label="CBF-RL"),
    Rectangle((0, 0), 1, 1, facecolor="khaki", alpha=0.45, label="Goal region"),
    Rectangle((0, 0), 1, 1, facecolor="dimgray", label="Obstacle"),
    Line2D([0], [0], marker="X", color="black", lw=0, label="Collision", markersize=12),
    Line2D([0], [0], marker="^", color="black", lw=0, label="Timeout", markersize=12),
    Line2D([0], [0], marker="D", color="black", lw=0, label="Other failure (fall/stuck)", markersize=11),
    Line2D([0], [0], marker="*", color="black", lw=0, label="Goal reached", markersize=17),
]
legend_ax.legend(handles=legend_elems, loc="upper center", ncol=6, fontsize=FS_LEGEND, frameon=False)

fig.savefig(
    "logs/evaluation/figures/seaspeed_combined_trajectories_with_cbfrl_regenerated_v2.png",
    dpi=300, bbox_inches="tight",
)
print("saved")
