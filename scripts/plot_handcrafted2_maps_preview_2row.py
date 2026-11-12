"""2-row preview of the 5 handcrafted2 training maps, restyled to match
scripts/plot_seaspeed_combined_trajectories_with_cbfrl_regenerated.py's
look (purple/yellow region shading, dark-gray obstacles, same font sizes).

Per-map initial/goal regions are the bounding box of that map's own 200
scenarios' start_x/y and goal_x/y (generated/procedural_maps/handcrafted2/
scenarios.csv) -- there's no explicit stored region rectangle, but this
reproduces the original preview's regions exactly (verified against
generated/procedural_maps/handcrafted2/previews/handcrafted2_maps_row.png).

Panels are placed with EXPLICIT inch-based coordinates (fig.add_axes)
rather than GridSpec wspace/hspace. All 5 arenas are exactly 15x15 m
(square), so a square axes box in inches gives set_aspect("equal") no
work to do and the box stays exactly where it's placed -- no more of the
GridSpec+equal-aspect "axes centered in its slot, wspace barely moves the
real gap" problem from earlier revisions of this script.

============================================================================
EASY-TO-CHANGE LAYOUT KNOBS (all in inches)
============================================================================
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MultipleLocator

# Size of one square map panel. This is the main lever for map size.
PANEL_IN = 3.4
# Horizontal gap between panels within a row.
COL_GAP_IN = 0.15
# Vertical gap between row 1 (3 maps) and row 2 (2 maps) -- must be big
# enough to fit row 1's "x [m]" label underneath it.
ROW_GAP_IN = 1.7

LEFT_MARGIN_IN = 0.95
RIGHT_MARGIN_IN = 0.15
BOTTOM_MARGIN_IN = 0.85
SUPTITLE_H_IN = 0.55
GAP_SUPTITLE_LEGEND_IN = 0.15
LEGEND_H_IN = 0.55
GAP_LEGEND_MAPS_IN = 0.55

FS_SUPTITLE = 30
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23

BANK_DIR = Path("generated/procedural_maps/handcrafted2")
OUT_PATH = BANK_DIR / "previews" / "handcrafted2_maps_2row.png"


def load_maps():
    scenarios = list(csv.DictReader(open(BANK_DIR / "scenarios.csv")))
    maps = []
    for mid in range(5):
        d = json.load(open(BANK_DIR / "maps" / f"map_{mid:03d}.json"))
        obstacles = [(o["x"], o["y"], o["radius"]) for o in d["obstacles"]]
        rows = [r for r in scenarios if r["map_id"] == str(mid)]
        sx = [float(r["start_x"]) for r in rows]; sy = [float(r["start_y"]) for r in rows]
        gx = [float(r["goal_x"]) for r in rows]; gy = [float(r["goal_y"]) for r in rows]
        start_region = (min(sx), max(sx), min(sy), max(sy))
        goal_region = (min(gx), max(gx), min(gy), max(gy))
        arena = d["arena"]
        maps.append({"obstacles": obstacles, "start_region": start_region,
                      "goal_region": goal_region, "arena": arena})
    return maps


def main():
    maps = load_maps()

    top_margin_in = SUPTITLE_H_IN + GAP_SUPTITLE_LEGEND_IN + LEGEND_H_IN + GAP_LEGEND_MAPS_IN
    fig_w = LEFT_MARGIN_IN + 3 * PANEL_IN + 2 * COL_GAP_IN + RIGHT_MARGIN_IN
    fig_h = top_margin_in + 2 * PANEL_IN + ROW_GAP_IN + BOTTOM_MARGIN_IN

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("Training Maps", fontsize=FS_SUPTITLE, fontweight="bold",
                  y=1 - (SUPTITLE_H_IN * 0.55) / fig_h)

    legend_ax = fig.add_axes([
        LEFT_MARGIN_IN / fig_w,
        (fig_h - SUPTITLE_H_IN - GAP_SUPTITLE_LEGEND_IN - LEGEND_H_IN) / fig_h,
        (fig_w - LEFT_MARGIN_IN - RIGHT_MARGIN_IN) / fig_w,
        LEGEND_H_IN / fig_h,
    ])
    legend_ax.axis("off")

    row1_span_in = 3 * PANEL_IN + 2 * COL_GAP_IN
    row2_span_in = 2 * PANEL_IN + COL_GAP_IN
    row2_x_offset_in = (row1_span_in - row2_span_in) / 2.0

    y_bot1_in = BOTTOM_MARGIN_IN + ROW_GAP_IN + PANEL_IN
    y_bot2_in = BOTTOM_MARGIN_IN

    panel_specs = []
    for i in range(3):
        x_in = LEFT_MARGIN_IN + i * (PANEL_IN + COL_GAP_IN)
        panel_specs.append((x_in, y_bot1_in, i == 0))
    for j in range(2):
        x_in = LEFT_MARGIN_IN + row2_x_offset_in + j * (PANEL_IN + COL_GAP_IN)
        panel_specs.append((x_in, y_bot2_in, j == 0))

    for idx, (x_in, y_in, is_leftmost) in enumerate(panel_specs):
        ax = fig.add_axes([x_in / fig_w, y_in / fig_h, PANEL_IN / fig_w, PANEL_IN / fig_h])
        m = maps[idx]
        half_w = m["arena"]["width"] / 2.0
        half_l = m["arena"]["length"] / 2.0

        sx0, sx1, sy0, sy1 = m["start_region"]
        gx0, gx1, gy0, gy1 = m["goal_region"]
        ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, facecolor="mediumpurple", alpha=0.35, zorder=1))
        ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="khaki", alpha=0.45, zorder=1))

        for (ox, oy, r) in m["obstacles"]:
            ax.add_patch(Circle((ox, oy), r, facecolor="dimgray", edgecolor="black", linewidth=1.2, zorder=2))

        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_l, half_l)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=FS_LABEL)
        if is_leftmost:
            ax.set_ylabel("y [m]", fontsize=FS_LABEL)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.tick_params(axis="both", labelsize=FS_TICK)
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)

    legend_elems = [
        Rectangle((0, 0), 1, 1, facecolor="mediumpurple", alpha=0.35, label="Initial region"),
        Rectangle((0, 0), 1, 1, facecolor="khaki", alpha=0.45, label="Goal region"),
        Rectangle((0, 0), 1, 1, facecolor="dimgray", label="Obstacle"),
    ]
    legend_ax.legend(handles=legend_elems, loc="center", ncol=3, fontsize=FS_LEGEND, frameon=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
