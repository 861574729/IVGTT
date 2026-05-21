"""
IVGTT Analysis for KBI202601003 - Jikang Study
Generates glucose/insulin curves, AUC/Kg bar charts, % change charts, and a summary report.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy import stats
import os

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\AI\projects\IVGTT"
STUDY_ID = "KBI202601003"
BASELINE_DATE = "2026/02/11 (Day -14)"
TREATMENT_DATE = "2026/04/08 (Day 43)"
TIME_POINTS = [0, 1, 3, 5, 10, 20, 40, 60]  # minutes

# ── Raw data extracted from IVGTT sheet (rows 5-22, 1-indexed) ─────────────
# Each animal: [group, animal_id, cage, glu_base[8], ins_base[8], glu_treat[8], ins_treat[8]]
# Group labels cleaned up for display

GROUPS_DISPLAY = {
    "1\nVehicle": "Vehicle",
    "2\nSema": "Sema",
    "3\nJKL-010": "JKL-010",
    "4 \nJKL-010\n+\nSema": "JKL-010+Sema",
}

raw_data = [
    # Group 1: Vehicle (rows 5-7)
    {"group": "Vehicle", "id": "170509", "cage": "207-01",
     "glu_base":   [74, 386, 311, 290, 244, 164, 89, 53],
     "ins_base":   [73.64, 193.1, 156.1, 194.2, 324.7, 320, 93.59, 62.68],
     "glu_treat":  [71, 377, 295, 264, 234, 173, 81, 54],
     "ins_treat":  [70.98, 312.6, 291.2, 314.3, 499.1, 474.6, 181.1, 100.1]},
    {"group": "Vehicle", "id": "110427", "cage": "207-02",
     "glu_base":   [92, 423, 396, 339, 281, 206, 138, 89],
     "ins_base":   [369.2, 340.9, 453.8, 437.6, 465, 517.6, 521.8, 294.5],
     "glu_treat":  [81, 415, 348, 301, 264, 205, 125, 80],
     "ins_treat":  [180.9, 227, 408.5, 311.5, 455.4, 488.8, 391.2, 404]},
    {"group": "Vehicle", "id": "180455", "cage": "207-05",
     "glu_base":   [78, 399, 338, 309, 256, 190, 94, 51],
     "ins_base":   [47.1, 165, 120.6, 113.5, 162.7, 182.3, 134, 35.52],
     "glu_treat":  [73, 449, 330, 311, 251, 193, 94, 89],
     "ins_treat":  [23.57, 122.4, 97.02, 71.67, 108.2, 122.2, 78.53, 41.28]},

    # Group 2: Sema (rows 8-12)
    {"group": "Sema", "id": "182101", "cage": "204-03",
     "glu_base":   [79, 408, 343, 311, 262, 189, 99, 61],
     "ins_base":   [68.86, 260.7, 278.2, 283.3, 456.8, 437.9, 190.5, 65.65],
     "glu_treat":  [74, 381, 338, 299, 258, 169, 73, 40],
     "ins_treat":  [50.28, 230.3, 291.3, 346.8, 596.1, 770.5, 216, 63.66]},
    {"group": "Sema", "id": "184155", "cage": "204-07",
     "glu_base":   [74, 408, 357, 322, 274, 225, 131, 71],
     "ins_base":   [57.48, 269.8, 246.3, 231.7, 309.6, 357.5, 359.9, 103.3],
     "glu_treat":  [65, 419, 328, 298, 257, 188, 89, 40],
     "ins_treat":  [22.57, 189, 224.4, 200.3, 342.6, 463.3, 246.1, 55.28]},
    {"group": "Sema", "id": "130437", "cage": "207-10",
     "glu_base":   [71, 482, 323, 297, 252, 176, 60, 19],
     "ins_base":   [19.8, 258.4, 302.4, 286.5, 545.2, 756, 152, 61.14],
     "glu_treat":  [71, 489, 334, 290, 240, 132, 46, 14],
     "ins_treat":  [35.66, 639.5, 624.1, 636.5, 958.1, 1235, 264.6, 93.39]},
    {"group": "Sema", "id": "150643", "cage": "207-12",
     "glu_base":   [72, 390, 341, 296, 261, 204, 125, 82],
     "ins_base":   [38.33, 126.9, 157.7, 131.5, 173.8, 126.3, 128.1, 59.59],
     "glu_treat":  [65, 411, 329, 286, 228, 153, 58, 49],
     "ins_treat":  [22.72, 180.2, 215.1, 193.6, 232.6, 255.2, 41.89, 28.67]},
    {"group": "Sema", "id": "200059", "cage": "207-15",
     "glu_base":   [59, 374, 334, 295, 266, 209, 138, 99],
     "ins_base":   [38.19, 179.4, 127.6, 102.8, 154.2, 165.3, 125.1, 67.88],
     "glu_treat":  [57, 406, 320, 286, 247, 175, 92, 48],
     "ins_treat":  [34.55, 242.6, 204.8, 157.3, 233.1, 304.5, 148.8, 42.39]},

    # Group 3: JKL-010 (rows 13-17)
    {"group": "JKL-010", "id": "112107", "cage": "204-02",
     "glu_base":   [59, 417, 308, 264, 213, 153, 68, 48],
     "ins_base":   [43.97, 333.7, 284.4, 232, 378.7, 374.1, 96.94, 46.46],
     "glu_treat":  [61, 480, 373, 321, 263, 195, 110, 67],
     "ins_treat":  [86.91, 137.6, 275.2, 185.1, 305.8, 305, 254.3, 66.89]},
    {"group": "JKL-010", "id": "184219", "cage": "204-08",
     "glu_base":   [76, 342, 340, 298, 255, 209, 124, 86],
     "ins_base":   [36.11, 152.6, 127.5, 121.5, 146.6, 120.7, 126.3, 68.72],
     "glu_treat":  [67, 376, 332, 280, 246, 191, 112, 69],
     "ins_treat":  [38.02, 152.5, 126.9, 109.5, 151.9, 152, 106.4, 56.29]},
    {"group": "JKL-010", "id": "110079", "cage": "207-03",
     "glu_base":   [85, 421, 335, 302, 269, 214, 152, 107],
     "ins_base":   [53.05, 92.08, 180.4, 137.3, 197.8, 140.6, 107.3, 80.64],
     "glu_treat":  [73, 492, 351, 306, 274, 222, 159, 106],
     "ins_treat":  [35.34, 32.94, 117.9, 78.47, 97.91, 89.13, 61.17, 62.85]},
    {"group": "JKL-010", "id": "180043", "cage": "207-04",
     "glu_base":   [88, 392, 316, 305, 255, 179, 78, 54],
     "ins_base":   [75.52, 158.8, 318.6, 334, 468, 426.9, 85.97, 50.92],
     "glu_treat":  [70, 302, 274, 256, 224, 150, 63, 55],
     "ins_treat":  [37.1, 152.6, 159.4, 145.1, 257.2, 207.6, 32.51, 24.06]},
    {"group": "JKL-010", "id": "150201", "cage": "207-11",
     "glu_base":   [74, 476, 391, 345, 307, 252, 170, 109],
     "ins_base":   [77.77, 335.5, 379.9, 326.5, 532.9, 799.9, 782.4, 481.4],
     "glu_treat":  [78, 456, 394, 354, 313, 249, 156, 97],
     "ins_treat":  [95.79, 373.7, 421.9, 320.7, 491.4, 595.8, 503.7, 270.9]},

    # Group 4: JKL-010+Sema (rows 18-22)
    {"group": "JKL-010+Sema", "id": "182015", "cage": "204-04",
     "glu_base":   [66, 373, 300, 266, 249, 188, 98, 58],
     "ins_base":   [24.11, 41, 109.4, 79.07, 224.1, 212.6, 77.15, 41.38],
     "glu_treat":  [55, 340, 279, 254, 221, 147, 54, 26],
     "ins_treat":  [14.95, 166.4, 123.1, 120, 239, 325.3, 54.22, 21.09]},
    {"group": "JKL-010+Sema", "id": "140257", "cage": "207-06",
     "glu_base":   [64, 376, 323, 270, 242, 193, 122, 81],
     "ins_base":   [117.1, 532, 902.8, 697.8, 612.2, 636.4, 444.3, 392.9],
     "glu_treat":  [61, 453, 344, 316, 253, 155, 69, 35],
     "ins_treat":  [84.88, 671.6, 1203, 992.2, 1242, 1362, 342.7, 117.4]},
    {"group": "JKL-010+Sema", "id": "170331", "cage": "207-07",
     "glu_base":   [69, 427, 339, 310, 269, 212, 133, 84],
     "ins_base":   [61.63, 239.8, 245.3, 201.4, 300.9, 263.5, 167.3, 72.35],
     "glu_treat":  [73, 392, 303, 270, 230, 174, 87, 65],
     "ins_treat":  [43.92, 122, 172, 153.5, 178.3, 168.6, 76.41, 33.07]},
    {"group": "JKL-010+Sema", "id": "180103", "cage": "207-08",
     "glu_base":   [69, 422, 339, 311, 261, 178, 77, 79],
     "ins_base":   [18.44, 162.3, 147.6, 114.7, 193.1, 165.6, 26.77, 50.44],
     "glu_treat":  [58, 319, 250, 226, 222, 126, 56, 60],
     "ins_treat":  [11.11, 74.34, 114, 99.7, 139.1, 125.1, 13.55, 6.09]},
    {"group": "JKL-010+Sema", "id": "200109", "cage": "207-16",
     "glu_base":   [71, 424, 368, 337, 304, 248, 150, 80],
     "ins_base":   [32.71, 202.2, 151.3, 121.4, 162.6, 213.9, 279.2, 91.94],
     "glu_treat":  [74, 448, 390, 352, 293, 216, 92, 41],
     "ins_treat":  [80.33, 225.7, 311.9, 232.9, 367.6, 497, 245.8, 67.89]},
]

GROUP_ORDER = ["Vehicle", "Sema", "JKL-010", "JKL-010+Sema"]
GROUP_COLORS = {
    "Vehicle": "#377eb8",
    "Sema": "#ff7f00",
    "JKL-010": "#4daf4a",
    "JKL-010+Sema": "#984ea3",
}
BASELINE_COLOR = "#888888"
TREATMENT_COLOR = "#e41a1c"

# ── Calculations ───────────────────────────────────────────────────────────
def calc_auc(values):
    """Trapezoidal AUC per the study formula."""
    t = TIME_POINTS
    intervals = [t[1]-t[0], t[2]-t[1], t[3]-t[2], t[4]-t[3],
                 t[5]-t[4], t[6]-t[5], t[7]-t[6]]
    auc = sum((values[i] + values[i+1]) * intervals[i] for i in range(7)) / 2
    return auc

def calc_kg(glu_values):
    """Kg = (Glu_10min - Glu_40min) / (30 * Glu_10min)"""
    return (glu_values[4] - glu_values[6]) / (30 * glu_values[4])

# Compute derived metrics for each animal
for d in raw_data:
    d["glu_auc_base"] = calc_auc(d["glu_base"])
    d["kg_base"] = calc_kg(d["glu_base"])
    d["ins_auc_base"] = calc_auc(d["ins_base"])
    d["glu_auc_treat"] = calc_auc(d["glu_treat"])
    d["kg_treat"] = calc_kg(d["glu_treat"])
    d["ins_auc_treat"] = calc_auc(d["ins_treat"])
    # % changes
    d["pct_t0"] = (d["glu_treat"][0] - d["glu_base"][0]) / d["glu_base"][0] * 100
    d["pct_t60"] = (d["glu_treat"][7] - d["glu_base"][7]) / d["glu_base"][7] * 100
    d["pct_glu_auc"] = (d["glu_auc_treat"] - d["glu_auc_base"]) / d["glu_auc_base"] * 100
    d["pct_kg"] = (d["kg_treat"] - d["kg_base"]) / d["kg_base"] * 100
    d["pct_ins_auc"] = (d["ins_auc_treat"] - d["ins_auc_base"]) / d["ins_auc_base"] * 100

# Group-level summaries
def group_summary(data_list, key):
    vals = [d[key] for d in data_list]
    return {
        "mean": np.mean(vals),
        "sd": np.std(vals, ddof=1),
        "sem": np.std(vals, ddof=1) / np.sqrt(len(vals)),
        "n": len(vals),
    }

groups_data = {}
for g in GROUP_ORDER:
    members = [d for d in raw_data if d["group"] == g]
    groups_data[g] = {"members": members, "n": len(members)}
    for metric in ["glu_auc_base", "kg_base", "ins_auc_base",
                   "glu_auc_treat", "kg_treat", "ins_auc_treat",
                   "pct_t0", "pct_t60", "pct_glu_auc", "pct_kg", "pct_ins_auc"]:
        groups_data[g][metric] = group_summary(members, metric)

# ── Plotting helpers ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Statistical helpers ────────────────────────────────────────────────────
def pvalue_stars(p):
    """Convert p-value to asterisk notation."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def add_significance_bracket(ax, x1, x2, y, h, p_value, fontsize=7):
    """Draw a significance bracket with p-value asterisks between two bars.

    Args:
        ax: matplotlib axis
        x1, x2: x-positions of the two bars
        y: top of the higher bar (data coords)
        h: step height for the bracket
        p_value: p-value from statistical test
        fontsize: asterisk font size
    """
    stars = pvalue_stars(p_value)
    if stars == "ns":
        return  # skip non-significant brackets

    # Get axis transform for consistent offsets
    ylim = ax.get_ylim()
    y_top = y + h

    # Draw bracket: two vertical ticks + horizontal line
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h * 0.3, y_top + h * 0.3, y_top],
            lw=1.0, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y_top + h * 0.35, stars,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def compute_stats():
    """Compute all pairwise statistical comparisons and return structured results."""
    results = {}

    # Within-group: Baseline vs Treatment (paired t-test)
    for g in GROUP_ORDER:
        members = groups_data[g]["members"]
        results[g] = {}
        for metric_key, label in [("glu_auc", "Glucose AUC"), ("ins_auc", "Insulin AUC"), ("kg", "Kg")]:
            base_vals = [d[f"{metric_key}_base"] for d in members]
            treat_vals = [d[f"{metric_key}_treat"] for d in members]
            t_stat, p_val = stats.ttest_rel(treat_vals, base_vals)
            results[g][f"{metric_key}_bt"] = {"p": p_val, "label": f"{g} BvsT {label}"}

    # Between-group: Treatment vs Vehicle (unpaired t-test)
    vehicle_members = groups_data["Vehicle"]["members"]
    for g in ["Sema", "JKL-010", "JKL-010+Sema"]:
        members = groups_data[g]["members"]
        for metric_key, label in [("glu_auc", "Glucose AUC"), ("ins_auc", "Insulin AUC"), ("kg", "Kg")]:
            veh_vals = [d[f"{metric_key}_treat"] for d in vehicle_members]
            grp_vals = [d[f"{metric_key}_treat"] for d in members]
            t_stat, p_val = stats.ttest_ind(grp_vals, veh_vals, equal_var=False)
            results[g][f"{metric_key}_vv"] = {"p": p_val, "label": f"{g} vs Veh {label}"}

    # % Change vs 0 (one-sample t-test)
    for g in GROUP_ORDER:
        members = groups_data[g]["members"]
        for pct_key, label in [("pct_t0", "T0 GLU %"), ("pct_t60", "T60 GLU %"),
                                ("pct_glu_auc", "Glu AUC %"), ("pct_ins_auc", "Ins AUC %"),
                                ("pct_kg", "Kg %")]:
            pct_vals = [d[pct_key] for d in members]
            t_stat, p_val = stats.ttest_1samp(pct_vals, 0)
            results[g][f"{pct_key}_1s"] = {"p": p_val, "label": f"{g} %Change {label}"}

    # Between-group: % Change vs Vehicle % Change (unpaired t-test)
    for g in ["Sema", "JKL-010", "JKL-010+Sema"]:
        members = groups_data[g]["members"]
        for pct_key in ["pct_t0", "pct_t60", "pct_glu_auc", "pct_kg", "pct_ins_auc"]:
            veh_vals = [d[pct_key] for d in vehicle_members]
            grp_vals = [d[pct_key] for d in members]
            t_stat, p_val = stats.ttest_ind(grp_vals, veh_vals, equal_var=False)
            results[g][f"{pct_key}_vv"] = {"p": p_val, "label": f"{g} vs Veh %Change {pct_key}"}

    return results


# Compute all statistics once
STAT_RESULTS = compute_stats()

# Inject % Change vs 0 p-values into groups_data for easy access
for g in GROUP_ORDER:
    for key in ["pct_t0", "pct_t60", "pct_glu_auc", "pct_kg", "pct_ins_auc"]:
        groups_data[g][f"{key}_pval"] = STAT_RESULTS[g][f"{key}_1s"]["p"]

# Inject between-group % change p-values into groups_data
for g in ["Sema", "JKL-010", "JKL-010+Sema"]:
    for key in ["pct_t0", "pct_t60", "pct_glu_auc", "pct_kg", "pct_ins_auc"]:
        groups_data[g][f"{key}_pval_vv"] = STAT_RESULTS[g][f"{key}_vv"]["p"]

# ── Figure 1: Glucose time curves (4 panels: Baseline + Treatment per group) ──
def plot_glucose_curves():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, g in enumerate(GROUP_ORDER):
        members = groups_data[g]["members"]
        color = GROUP_COLORS[g]

        # Baseline glucose
        ax = axes[0, i]
        for d in members:
            ax.plot(TIME_POINTS, d["glu_base"], "o-", color=BASELINE_COLOR,
                    alpha=0.5, markersize=4, linewidth=1)
        mean_vals = np.mean([d["glu_base"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_vals, "o-", color=BASELINE_COLOR,
                linewidth=2.5, markersize=6, label="Mean±SEM")
        sem_vals = np.std([d["glu_base"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        ax.fill_between(TIME_POINTS, mean_vals - sem_vals, mean_vals + sem_vals,
                        color=BASELINE_COLOR, alpha=0.15)
        ax.set_title(f"{g} — Baseline\n{BASELINE_DATE}", fontsize=9)
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_xlabel("Time (min)")
        ax.axhline(y=mean_vals[0], color="gray", linestyle=":", alpha=0.5)

        # Treatment glucose
        ax = axes[1, i]
        for d in members:
            ax.plot(TIME_POINTS, d["glu_treat"], "o-", color=TREATMENT_COLOR,
                    alpha=0.5, markersize=4, linewidth=1)
        mean_vals = np.mean([d["glu_treat"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_vals, "o-", color=TREATMENT_COLOR,
                linewidth=2.5, markersize=6, label="Mean±SEM")
        sem_vals = np.std([d["glu_treat"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        ax.fill_between(TIME_POINTS, mean_vals - sem_vals, mean_vals + sem_vals,
                        color=TREATMENT_COLOR, alpha=0.15)
        ax.set_title(f"{g} — Treatment\n{TREATMENT_DATE}", fontsize=9)
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_xlabel("Time (min)")
        ax.axhline(y=mean_vals[0], color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(f"{STUDY_ID}: IVGTT Plasma Glucose Curves", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig1_glucose_curves.png")

# ── Figure 2: Insulin time curves ──────────────────────────────────────────
def plot_insulin_curves():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, g in enumerate(GROUP_ORDER):
        members = groups_data[g]["members"]
        color = GROUP_COLORS[g]

        ax = axes[0, i]
        for d in members:
            ax.plot(TIME_POINTS, d["ins_base"], "o-", color=BASELINE_COLOR,
                    alpha=0.5, markersize=4, linewidth=1)
        mean_vals = np.mean([d["ins_base"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_vals, "o-", color=BASELINE_COLOR,
                linewidth=2.5, markersize=6)
        sem_vals = np.std([d["ins_base"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        ax.fill_between(TIME_POINTS, mean_vals - sem_vals, mean_vals + sem_vals,
                        color=BASELINE_COLOR, alpha=0.15)
        ax.set_title(f"{g} — Baseline\n{BASELINE_DATE}", fontsize=9)
        ax.set_ylabel("Insulin (μU/mL)")
        ax.set_xlabel("Time (min)")

        ax = axes[1, i]
        for d in members:
            ax.plot(TIME_POINTS, d["ins_treat"], "o-", color=TREATMENT_COLOR,
                    alpha=0.5, markersize=4, linewidth=1)
        mean_vals = np.mean([d["ins_treat"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_vals, "o-", color=TREATMENT_COLOR,
                linewidth=2.5, markersize=6)
        sem_vals = np.std([d["ins_treat"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        ax.fill_between(TIME_POINTS, mean_vals - sem_vals, mean_vals + sem_vals,
                        color=TREATMENT_COLOR, alpha=0.15)
        ax.set_title(f"{g} — Treatment\n{TREATMENT_DATE}", fontsize=9)
        ax.set_ylabel("Insulin (μU/mL)")
        ax.set_xlabel("Time (min)")

    fig.suptitle(f"{STUDY_ID}: IVGTT Plasma Insulin Curves", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig2_insulin_curves.png")

# ── Figure 3: Baseline vs Treatment overlay per group (Glucose) ────────────
def plot_glucose_overlay():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, g in enumerate(GROUP_ORDER):
        ax = axes[i]
        members = groups_data[g]["members"]

        mean_base = np.mean([d["glu_base"] for d in members], axis=0)
        sem_base = np.std([d["glu_base"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        mean_treat = np.mean([d["glu_treat"] for d in members], axis=0)
        sem_treat = np.std([d["glu_treat"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))

        ax.fill_between(TIME_POINTS, mean_base - sem_base, mean_base + sem_base,
                        color=BASELINE_COLOR, alpha=0.15)
        ax.fill_between(TIME_POINTS, mean_treat - sem_treat, mean_treat + sem_treat,
                        color=TREATMENT_COLOR, alpha=0.15)
        ax.plot(TIME_POINTS, mean_base, "o-", color=BASELINE_COLOR, linewidth=2,
                markersize=6, label="Baseline")
        ax.plot(TIME_POINTS, mean_treat, "s--", color=TREATMENT_COLOR, linewidth=2,
                markersize=6, label="Treatment")
        ax.set_title(g, fontweight="bold")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=8)

    fig.suptitle(f"{STUDY_ID}: Baseline vs Treatment Glucose (Mean±SEM)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig3_glucose_overlay.png")

# ── Figure 4: Baseline vs Treatment overlay per group (Insulin) ────────────
def plot_insulin_overlay():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, g in enumerate(GROUP_ORDER):
        ax = axes[i]
        members = groups_data[g]["members"]

        mean_base = np.mean([d["ins_base"] for d in members], axis=0)
        sem_base = np.std([d["ins_base"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))
        mean_treat = np.mean([d["ins_treat"] for d in members], axis=0)
        sem_treat = np.std([d["ins_treat"] for d in members], axis=0, ddof=1) / np.sqrt(len(members))

        ax.fill_between(TIME_POINTS, mean_base - sem_base, mean_base + sem_base,
                        color=BASELINE_COLOR, alpha=0.15)
        ax.fill_between(TIME_POINTS, mean_treat - sem_treat, mean_treat + sem_treat,
                        color=TREATMENT_COLOR, alpha=0.15)
        ax.plot(TIME_POINTS, mean_base, "o-", color=BASELINE_COLOR, linewidth=2,
                markersize=6, label="Baseline")
        ax.plot(TIME_POINTS, mean_treat, "s--", color=TREATMENT_COLOR, linewidth=2,
                markersize=6, label="Treatment")
        ax.set_title(g, fontweight="bold")
        ax.set_ylabel("Insulin (μU/mL)")
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=8)

    fig.suptitle(f"{STUDY_ID}: Baseline vs Treatment Insulin (Mean±SEM)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig4_insulin_overlay.png")

# ── Figure 5: AUC & Kg bar charts with significance ───────────────────────
def plot_auc_kg_bars():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metric_labels = {
        "glu_auc": "Glucose AUC",
        "ins_auc": "Insulin AUC",
        "kg": "Kg",
    }
    metric_keys = ["glu_auc", "ins_auc", "kg"]
    bar_width = 0.45
    x = np.arange(len(GROUP_ORDER))

    for col, key in enumerate(metric_keys):
        ax = axes[col]
        pct_key = f"pct_{key}"
        pct_means = np.array([groups_data[g][pct_key]["mean"] for g in GROUP_ORDER])
        pct_sems = np.array([groups_data[g][pct_key]["sem"] for g in GROUP_ORDER])

        bar_colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
        ax.bar(x, pct_means, bar_width, yerr=pct_sems, capsize=6,
               color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.axhline(y=0, color="black", linewidth=0.8)

        data_max = np.max(pct_means + pct_sems)
        data_min = np.min(pct_means - pct_sems)
        y_span = max(abs(data_max), abs(data_min)) * 2 + 0.01
        y_buffer = y_span * 0.06
        tracked_max = data_max
        tracked_min = data_min

        # Vs-0 significance stars
        for i, g in enumerate(GROUP_ORDER):
            p_val = groups_data[g][f"{pct_key}_pval"]
            stars = pvalue_stars(p_val)
            if stars == "ns":
                continue
            sign = 1 if pct_means[i] > 0 else -1
            y_pos = pct_means[i] + sign * (pct_sems[i] + y_buffer * 1.5)
            ax.text(x[i], y_pos, stars, ha="center",
                    va="bottom" if sign > 0 else "top",
                    fontsize=8, fontweight="bold")
            if y_pos > 0:
                tracked_max = max(tracked_max, y_pos + y_buffer)
            else:
                tracked_min = min(tracked_min, y_pos - y_buffer)

        # Between-group significance brackets (vs Vehicle on % change)
        bracket_level = 0
        for i, g in enumerate(GROUP_ORDER):
            if g == "Vehicle":
                continue
            p_val = groups_data[g].get(f"{pct_key}_pval_vv", 1.0)
            stars = pvalue_stars(p_val)
            if stars == "ns":
                continue
            top_y = max(pct_means[0] + pct_sems[0],
                       pct_means[i] + pct_sems[i])
            h = y_buffer * (4 + bracket_level * 1.8)
            bracket_y = top_y + h
            ax.plot([x[0], x[0], x[i], x[i]],
                    [bracket_y, bracket_y + y_buffer * 0.3,
                     bracket_y + y_buffer * 0.3, bracket_y],
                    lw=1.0, color="gray", clip_on=False)
            ax.text((x[0] + x[i]) / 2, bracket_y + y_buffer * 0.35,
                    stars, ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="gray")
            tracked_max = max(tracked_max, bracket_y + y_buffer * 0.8)
            bracket_level += 1

        # Auto-adjust ylim to include all annotations
        margin = y_span * 0.12
        ax.set_ylim(tracked_min - margin, tracked_max + margin)

        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER, fontsize=8)
        ax.set_ylabel(f"{metric_labels[key]} % Change from Baseline  (±SEM)")
        ax.set_title(f"{metric_labels[key]} — % Change", fontsize=10)

    fig.suptitle(
        f"{STUDY_ID}: IVGTT AUC and Kg — % Change from Baseline\n"
        "* p<0.05, ** p<0.01, *** p<0.001 vs 0 (black) | vs Vehicle (gray brackets)",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    save_fig(fig, "fig5_auc_kg_bars.png")

# ── Figure 6: % Change summary with significance ───────────────────────────
def plot_pct_change():
    fig, ax = plt.subplots(figsize=(16, 7))
    pct_metrics = ["pct_t0", "pct_t60", "pct_glu_auc", "pct_kg", "pct_ins_auc"]
    pct_labels = ["T0min GLU", "T60min GLU", "Glucose AUC", "Kg", "Insulin AUC"]
    bar_width = 0.18
    x = np.arange(len(pct_metrics))

    tracked_max = 0
    tracked_min = 0

    for i, g in enumerate(GROUP_ORDER):
        means = [groups_data[g][m]["mean"] for m in pct_metrics]
        sems = [groups_data[g][m]["sem"] for m in pct_metrics]
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=sems, capsize=3,
               color=GROUP_COLORS[g], alpha=0.85, label=g, edgecolor="white")

        for j, m in enumerate(pct_metrics):
            val = means[j]
            tracked_max = max(tracked_max, val + sems[j])
            tracked_min = min(tracked_min, val - sems[j])

    # Now place stars, tracking extremes
    y_span = max(abs(tracked_max), abs(tracked_min)) * 2 + 0.01
    y_buffer = y_span * 0.05

    for i, g in enumerate(GROUP_ORDER):
        means = [groups_data[g][m]["mean"] for m in pct_metrics]
        sems = [groups_data[g][m]["sem"] for m in pct_metrics]
        offset = (i - 1.5) * bar_width

        for j, m in enumerate(pct_metrics):
            p_val = groups_data[g][f"{m}_pval"]
            stars = pvalue_stars(p_val)
            if stars == "ns":
                continue
            x_pos = x[j] + offset
            val = means[j]
            sem = sems[j]
            sign = 1 if val > 0 else -1
            y_pos = val + sign * (sem + y_buffer * 1.5)
            ax.text(x_pos, y_pos, stars, ha="center",
                    va="bottom" if sign > 0 else "top",
                    fontsize=6, fontweight="bold")
            if y_pos > 0:
                tracked_max = max(tracked_max, y_pos + y_buffer)
            else:
                tracked_min = min(tracked_min, y_pos - y_buffer)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pct_labels, fontsize=9)
    ax.set_ylabel("% Change from Baseline  (±SEM)")

    # Auto-adjust ylim so stars stay inside
    margin = y_span * 0.15
    ax.set_ylim(tracked_min - margin, tracked_max + margin)

    ax.set_title(f"{STUDY_ID}: % Change in IVGTT Parameters (Treatment vs Baseline)\n* p<0.05, ** p<0.01, *** p<0.001 vs 0",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, ncol=4, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig6_pct_change.png")

# ── Figure 7: Treatment-only between-group comparison ──────────────────────
def plot_treatment_between_groups():
    """Bar chart of treatment values only, with significance vs Vehicle."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [
        ("glu_auc", "Glucose AUC (mg/dL·min)", "treat"),
        ("ins_auc", "Insulin AUC (μU/mL·min)", "treat"),
        ("kg", "Kg (glucose clearance rate)", "treat"),
    ]
    bar_width = 0.45
    x = np.arange(len(GROUP_ORDER))

    for col, (key, ylabel, phase) in enumerate(metrics):
        ax = axes[col]
        treat_key = f"{key}_treat"
        means = np.array([groups_data[g][treat_key]["mean"] for g in GROUP_ORDER])
        sems = np.array([groups_data[g][treat_key]["sem"] for g in GROUP_ORDER])
        colors = [GROUP_COLORS[g] for g in GROUP_ORDER]

        ax.bar(x, means, bar_width, yerr=sems, capsize=6,
               color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

        # Between-group: each treatment vs Vehicle
        y_top = max(means + sems)
        y_range = y_top - min(0, min(means - sems))
        h = y_range * 0.05

        for i, g in enumerate(GROUP_ORDER):
            if g == "Vehicle":
                continue
            p_val = STAT_RESULTS[g].get(f"{key}_vv", {}).get("p", 1.0)
            stars = pvalue_stars(p_val)
            if stars == "ns":
                continue
            bracket_y = y_top + h * (2 + 0.8 * i)
            ax.plot([0, 0, x[i], x[i]],
                    [bracket_y, bracket_y + h * 0.3,
                     bracket_y + h * 0.3, bracket_y],
                    lw=1.0, color="black", clip_on=False)
            ax.text((0 + x[i]) / 2, bracket_y + h * 0.35, stars,
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\nTreatment vs Vehicle", fontsize=10)

    fig.suptitle(f"{STUDY_ID}: Treatment Phase — Between-Group Comparison\nSignificance vs Vehicle: * p<0.05, ** p<0.01, *** p<0.001",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig9_treatment_vs_vehicle.png")


# ── Figure 7 (renumbered): Individual animal spaghetti plots ───────────────
def plot_spaghetti():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, g in enumerate(GROUP_ORDER):
        ax = axes[i // 2, i % 2]
        members = groups_data[g]["members"]
        for d in members:
            ax.plot(TIME_POINTS, d["glu_base"], "o-", color=BASELINE_COLOR,
                    alpha=0.3, markersize=3, linewidth=0.8)
            ax.plot(TIME_POINTS, d["glu_treat"], "s--", color=TREATMENT_COLOR,
                    alpha=0.3, markersize=3, linewidth=0.8)
        # Mean lines
        mean_base = np.mean([d["glu_base"] for d in members], axis=0)
        mean_treat = np.mean([d["glu_treat"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_base, "o-", color=BASELINE_COLOR,
                linewidth=3, markersize=7, label="Baseline Mean")
        ax.plot(TIME_POINTS, mean_treat, "s--", color=TREATMENT_COLOR,
                linewidth=3, markersize=7, label="Treatment Mean")
        ax.set_title(f"{g} (n={len(members)})", fontweight="bold")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=7)

    fig.suptitle(f"{STUDY_ID}: Individual Glucose Curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig7_spaghetti_glucose.png")

# ── Figure 8: Insulin spaghetti ────────────────────────────────────────────
def plot_insulin_spaghetti():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, g in enumerate(GROUP_ORDER):
        ax = axes[i // 2, i % 2]
        members = groups_data[g]["members"]
        for d in members:
            ax.plot(TIME_POINTS, d["ins_base"], "o-", color=BASELINE_COLOR,
                    alpha=0.3, markersize=3, linewidth=0.8)
            ax.plot(TIME_POINTS, d["ins_treat"], "s--", color=TREATMENT_COLOR,
                    alpha=0.3, markersize=3, linewidth=0.8)
        mean_base = np.mean([d["ins_base"] for d in members], axis=0)
        mean_treat = np.mean([d["ins_treat"] for d in members], axis=0)
        ax.plot(TIME_POINTS, mean_base, "o-", color=BASELINE_COLOR,
                linewidth=3, markersize=7, label="Baseline Mean")
        ax.plot(TIME_POINTS, mean_treat, "s--", color=TREATMENT_COLOR,
                linewidth=3, markersize=7, label="Treatment Mean")
        ax.set_title(f"{g} (n={len(members)})", fontweight="bold")
        ax.set_ylabel("Insulin (μU/mL)")
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=7)

    fig.suptitle(f"{STUDY_ID}: Individual Insulin Curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig8_spaghetti_insulin.png")

# ── Report generation ──────────────────────────────────────────────────────
def generate_report():
    lines = []
    lines.append(f"# IVGTT Analysis Report — {STUDY_ID}")
    lines.append(f"## Jikang Study — KBI202601003")
    lines.append("")
    lines.append(f"- **Baseline IVGTT:** {BASELINE_DATE}")
    lines.append(f"- **Treatment IVGTT:** {TREATMENT_DATE}")
    lines.append("")
    lines.append("## Groups")
    for g in GROUP_ORDER:
        members = groups_data[g]["members"]
        ids = ", ".join(d["id"] for d in members)
        lines.append(f"- **{g}** (n={len(members)}): {ids}")
    lines.append("")

    # Summary table: Glucose AUC
    lines.append("## Glucose AUC (mg/dL·min)")
    lines.append("| Group | Baseline Mean ± SD | Treatment Mean ± SD | % Change | B vs T P-value |")
    lines.append("|-------|--------------------|----------------------|----------|----------------|")
    for g in GROUP_ORDER:
        b = groups_data[g]["glu_auc_base"]
        t = groups_data[g]["glu_auc_treat"]
        p = groups_data[g]["pct_glu_auc"]
        pv = STAT_RESULTS[g]["glu_auc_bt"]["p"]
        pv_str = f"p={pv:.4f}" + (" *" if pv < 0.05 else "")
        lines.append(f"| {g} | {b['mean']:.1f} ± {b['sd']:.1f} | {t['mean']:.1f} ± {t['sd']:.1f} | {p['mean']:.1f}% | {pv_str} |")
    lines.append("")

    # Insulin AUC
    lines.append("## Insulin AUC (μU/mL·min)")
    lines.append("| Group | Baseline Mean ± SD | Treatment Mean ± SD | % Change | B vs T P-value |")
    lines.append("|-------|--------------------|----------------------|----------|----------------|")
    for g in GROUP_ORDER:
        b = groups_data[g]["ins_auc_base"]
        t = groups_data[g]["ins_auc_treat"]
        p = groups_data[g]["pct_ins_auc"]
        pv = STAT_RESULTS[g]["ins_auc_bt"]["p"]
        pv_str = f"p={pv:.4f}" + (" *" if pv < 0.05 else "")
        lines.append(f"| {g} | {b['mean']:.1f} ± {b['sd']:.1f} | {t['mean']:.1f} ± {t['sd']:.1f} | {p['mean']:.1f}% | {pv_str} |")
    lines.append("")

    # Kg
    lines.append("## Kg (Glucose Clearance Rate)")
    lines.append("| Group | Baseline Mean ± SD | Treatment Mean ± SD | % Change | B vs T P-value |")
    lines.append("|-------|--------------------|----------------------|----------|----------------|")
    for g in GROUP_ORDER:
        b = groups_data[g]["kg_base"]
        t = groups_data[g]["kg_treat"]
        p = groups_data[g]["pct_kg"]
        pv = STAT_RESULTS[g]["kg_bt"]["p"]
        pv_str = f"p={pv:.4f}" + (" *" if pv < 0.05 else "")
        lines.append(f"| {g} | {b['mean']:.4f} ± {b['sd']:.4f} | {t['mean']:.4f} ± {t['sd']:.4f} | {p['mean']:.1f}% | {pv_str} |")
    lines.append("")

    # Between-group treatment comparisons
    lines.append("## Treatment Phase: Between-Group Comparisons (vs Vehicle)")
    lines.append("| Group | Glucose AUC p | Insulin AUC p | Kg p |")
    lines.append("|-------|--------------|---------------|------|")
    for g in ["Sema", "JKL-010", "JKL-010+Sema"]:
        ga = STAT_RESULTS[g]["glu_auc_vv"]["p"]
        ia = STAT_RESULTS[g]["ins_auc_vv"]["p"]
        kg = STAT_RESULTS[g]["kg_vv"]["p"]
        lines.append(f"| {g} | {ga:.4f}" + (" *" if ga < 0.05 else "") + f" | {ia:.4f}" + (" *" if ia < 0.05 else "") + f" | {kg:.4f}" + (" *" if kg < 0.05 else "") + " |")
    lines.append("")

    # % Change summary
    lines.append("## % Change Summary (Treatment vs Baseline) with Significance")
    lines.append("| Group | T0 GLU | T60 GLU | Glucose AUC | Kg | Insulin AUC |")
    lines.append("|-------|--------|---------|-------------|-----|-------------|")
    for g in GROUP_ORDER:
        t0 = groups_data[g]["pct_t0"]
        t60 = groups_data[g]["pct_t60"]
        ga = groups_data[g]["pct_glu_auc"]
        kg = groups_data[g]["pct_kg"]
        ia = groups_data[g]["pct_ins_auc"]

        def pct_sig(pv):
            s = pvalue_stars(pv)
            return f" {s}" if s != "ns" else ""

        t0s = pct_sig(STAT_RESULTS[g]["pct_t0_1s"]["p"])
        t60s = pct_sig(STAT_RESULTS[g]["pct_t60_1s"]["p"])
        gas = pct_sig(STAT_RESULTS[g]["pct_glu_auc_1s"]["p"])
        kgs = pct_sig(STAT_RESULTS[g]["pct_kg_1s"]["p"])
        ias = pct_sig(STAT_RESULTS[g]["pct_ins_auc_1s"]["p"])
        lines.append(f"| {g} | {t0['mean']:.1f}%{t0s} | {t60['mean']:.1f}%{t60s} | {ga['mean']:.1f}%{gas} | {kg['mean']:.1f}%{kgs} | {ia['mean']:.1f}%{ias} |")
    lines.append("")
    lines.append("*注: * p<0.05, ** p<0.01, *** p<0.001 vs 0 (单样本t检验)*")
    lines.append("")

    # Individual data
    lines.append("## Individual Animal Data")
    for g in GROUP_ORDER:
        lines.append(f"### {g}")
        lines.append("| ID | Cage | Glu AUC Base | Glu AUC Treat | Ins AUC Base | Ins AUC Treat | Kg Base | Kg Treat |")
        lines.append("|----|------|-------------|---------------|-------------|---------------|---------|----------|")
        for d in groups_data[g]["members"]:
            lines.append(f"| {d['id']} | {d['cage']} | {d['glu_auc_base']:.1f} | {d['glu_auc_treat']:.1f} | {d['ins_auc_base']:.1f} | {d['ins_auc_treat']:.1f} | {d['kg_base']:.4f} | {d['kg_treat']:.4f} |")
        lines.append("")

    # Figures generated
    lines.append("## Figures")
    for fname in [
        "fig1_glucose_curves.png — Glucose time curves (baseline & treatment per group)",
        "fig2_insulin_curves.png — Insulin time curves (baseline & treatment per group)",
        "fig3_glucose_overlay.png — Baseline vs Treatment glucose overlay",
        "fig4_insulin_overlay.png — Baseline vs Treatment insulin overlay",
        "fig5_auc_kg_bars.png — AUC and Kg % change bar charts with vs-0 & between-group significance",
        "fig6_pct_change.png — % Change summary with vs-0 significance",
        "fig7_spaghetti_glucose.png — Individual glucose curves per group",
        "fig8_spaghetti_insulin.png — Individual insulin curves per group",
        "fig9_treatment_vs_vehicle.png — Treatment phase between-group comparison",
    ]:
        lines.append(f"- {fname}")

    report_path = os.path.join(OUTPUT_DIR, "IVGTT_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: IVGTT_Report.md")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"IVGTT Analysis — {STUDY_ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print("Generating figures...")
    plot_glucose_curves()
    plot_insulin_curves()
    plot_glucose_overlay()
    plot_insulin_overlay()
    plot_auc_kg_bars()
    plot_pct_change()
    plot_treatment_between_groups()
    plot_spaghetti()
    plot_insulin_spaghetti()

    print()
    print("Generating report...")
    generate_report()

    print()
    print("Done! All outputs in:", OUTPUT_DIR)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (Mean ± SEM, p-value: * p<0.05, ** p<0.01, *** p<0.001)")
    print("=" * 60)
    for g in GROUP_ORDER:
        ga = groups_data[g]["pct_glu_auc"]
        kg = groups_data[g]["pct_kg"]
        ia = groups_data[g]["pct_ins_auc"]
        ga_p = STAT_RESULTS[g]["glu_auc_bt"]["p"]
        kg_p = STAT_RESULTS[g]["kg_bt"]["p"]
        ia_p = STAT_RESULTS[g]["ins_auc_bt"]["p"]
        ga_s = pvalue_stars(ga_p)
        kg_s = pvalue_stars(kg_p)
        ia_s = pvalue_stars(ia_p)
        print(f"  {g}:")
        print(f"    Glucose AUC change: {ga['mean']:.1f}% ± {ga['sem']:.1f}%  (p={ga_p:.4f} {ga_s})")
        print(f"    Kg change:          {kg['mean']:.1f}% ± {kg['sem']:.1f}%  (p={kg_p:.4f} {kg_s})")
        print(f"    Insulin AUC change: {ia['mean']:.1f}% ± {ia['sem']:.1f}%  (p={ia_p:.4f} {ia_s})")

    print("\n" + "-" * 60)
    print("BETWEEN-GROUP (Treatment vs Vehicle):")
    for g in ["Sema", "JKL-010", "JKL-010+Sema"]:
        ga_p = STAT_RESULTS[g]["glu_auc_vv"]["p"]
        kg_p = STAT_RESULTS[g]["kg_vv"]["p"]
        ia_p = STAT_RESULTS[g]["ins_auc_vv"]["p"]
        print(f"  {g} vs Vehicle:")
        print(f"    Glucose AUC p={ga_p:.4f} {pvalue_stars(ga_p)}")
        print(f"    Kg p={kg_p:.4f} {pvalue_stars(kg_p)}")
        print(f"    Insulin AUC p={ia_p:.4f} {pvalue_stars(ia_p)}")
