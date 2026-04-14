"""
visualize_coords.py
Batch experiment results map visualization - single HTML, Reward / Time / PDR button toggle

stat.txt structure:
  320 rows = 64 rules × 5 blocks order: Reward → Time → PDR → RewardWOG → PDRWOG
  Each row: rule_name  mean  std  95%CI_half  (2+ space delimited)
  Visualization value = mean of 64 means per block (mean of means)

Metric direction: Reward higher is better / Time·PDR lower is better
Coordinates with failed scenario generation are shown as black markers.

Usage:
    python experiment_1/visualize_coords.py
    python experiment_1/visualize_coords.py --out my_map.html
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch experiment results map visualization")
    p.add_argument("--progress",    default="experiment_1/progress.json")
    p.add_argument("--coords",      default="experiment_1/coords_korea.csv")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--out",         default="",
                   help="Output HTML path (default: experiment_1/coords_map.html)")
    p.add_argument("--clip-pct",    type=float, default=5.0,
                   help="Colormap clipping percentile (default: 5 → 5th~95th percentile range, 0 to disable)")
    p.add_argument("--outlier-n",   type=int,   default=3,
                   help="Number of outliers on each side (default: 3 → highlight top/bottom 3, 0 to disable)")
    p.add_argument("--hist-format", choices=["png", "pdf"], default="pdf",
                   help="Histogram output format (default: pdf — vector, ideal for papers)")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_coords(path: Path) -> dict:
    coords = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(int(row["coord_id"]))
            coords[cid] = (float(row["latitude"]), float(row["longitude"]))
    return coords


def load_progress(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_results_dir(progress: dict) -> Path | None:
    for v in progress.get("statuses", {}).values():
        if v.get("status") == "done" and v.get("config_path"):
            cp = Path(v["config_path"])
            parts = cp.parts
            try:
                sce_idx = next(i for i, p in enumerate(parts) if p == "scenarios")
                exp_id  = parts[sce_idx + 1]
                rd = Path(*parts[:sce_idx]) / "results" / exp_id
                if rd.exists():
                    return rd
            except (StopIteration, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# Parse stat file
# ---------------------------------------------------------------------------

def parse_stat_means(stat_path: Path) -> dict | None:
    """
    stat.txt -> {"reward": mean, "time": mean, "pdr": mean}

    Row format: rule_name  mean  std  95%CI_half  (2+ space delimited)
    320 rows = 64 rules x 5 blocks (Reward, Time, PDR, RewardWOG, PDRWOG)
    -> Average of 64 mean values per block; only first 3 blocks used
    """
    rows = []
    with open(stat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 4:
                try:
                    rows.append(float(parts[1]))
                except ValueError:
                    continue

    n = len(rows)
    if n < 3:
        return None

    # Block size: total rows / 5 (rounded)
    n_rules = round(n / 5)
    if n_rules < 1:
        return None

    def block_mean(start):
        vals = rows[start: start + n_rules]
        return sum(vals) / len(vals) if vals else None

    rew = block_mean(0)
    tim = block_mean(n_rules)
    pdr = block_mean(2 * n_rules)

    if rew is None or tim is None or pdr is None:
        return None

    return {"reward": rew, "time": tim, "pdr": pdr}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(coords: dict, progress: dict, results_dir: Path) -> dict:
    statuses = progress.get("statuses", {})
    data = {}

    for cid, (lat, lon) in coords.items():
        entry = {"lat": lat, "lon": lon, "reward": None, "time": None, "pdr": None}
        status_info = statuses.get(cid, {})

        if status_info.get("status") != "done":
            data[cid] = entry
            continue

        config_path = status_info.get("config_path", "")
        coord_str = None
        if config_path:
            m = re.search(r'\([\d.]+,[\d.]+\)', config_path)
            if m:
                coord_str = m.group(0)
        if coord_str is None:
            coord_str = f"({lat},{lon})"

        stat_path = results_dir / coord_str / f"results_{coord_str}_stat.txt"
        if not stat_path.exists():
            data[cid] = entry
            continue

        means = parse_stat_means(stat_path)
        if means:
            entry.update(means)
        data[cid] = entry

    return data


# ---------------------------------------------------------------------------
# Colormap range computation (percentile clipping)
# ---------------------------------------------------------------------------

def compute_ranges(data: dict, clip_pct: float) -> dict:
    """
    Return colormap vmin/vmax per metric.
    If clip_pct > 0, use P(clip_pct) ~ P(100-clip_pct) range.
    Values outside range are clamped to the extreme colors.
    """
    import numpy as np
    ranges = {}
    for metric, label, _ in MODES:
        vals = np.array([v[metric] for v in data.values() if v[metric] is not None])
        if len(vals) == 0:
            ranges[metric] = (0.0, 1.0, 0.0, 1.0)
            continue
        true_min, true_max = float(vals.min()), float(vals.max())
        if clip_pct > 0:
            vmin = float(np.percentile(vals, clip_pct))
            vmax = float(np.percentile(vals, 100 - clip_pct))
            clip_str = f"P{clip_pct:.0f}~P{100-clip_pct:.0f}"
        else:
            vmin, vmax = true_min, true_max
            clip_str = "min~max"
        ranges[metric] = (vmin, vmax, true_min, true_max)
        print(f"  {label}: color range {clip_str} = [{vmin:.4f}, {vmax:.4f}]  "
              f"(full [{true_min:.4f}, {true_max:.4f}])")
    return ranges


# ---------------------------------------------------------------------------
# Outlier ID computation
# ---------------------------------------------------------------------------

def get_outlier_ids(data: dict, metric: str, outlier_n: int, high_is_good: bool):
    """
    Return top N / bottom N coord_ids based on metric value.
    high_is_good=True : top N -> good, bottom N -> bad
    high_is_good=False: bottom N -> good, top N -> bad
    Returns: (good_ids: set, bad_ids: set)
    """
    if outlier_n <= 0:
        return set(), set()
    val_cid = sorted(
        ((v[metric], cid) for cid, v in data.items() if v[metric] is not None),
        key=lambda x: x[0]
    )
    if len(val_cid) == 0:
        return set(), set()
    n = min(outlier_n, len(val_cid) // 2)
    bottom_ids = {cid for _, cid in val_cid[:n]}
    top_ids    = {cid for _, cid in val_cid[-n:]}
    if high_is_good:
        return top_ids, bottom_ids    # high=good
    else:
        return bottom_ids, top_ids    # low=good


# ---------------------------------------------------------------------------
# Color computation
# ---------------------------------------------------------------------------

def compute_color(val: float, vmin: float, vmax: float, high_is_good: bool) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib required: pip install matplotlib")
        sys.exit(1)

    t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
    if not high_is_good:
        t = 1.0 - t
    t = max(0.0, min(1.0, t))
    r, g, b, _ = plt.get_cmap("RdYlGn")(t)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Histogram generation
# ---------------------------------------------------------------------------

HIST_CONFIG = [
    ("reward", "The expected number of survivors",     True,  "RdYlGn",   "Higher is better"),
    ("time",   "The response completion time (min)", False, "RdYlGn_r",  "Lower is better"),
    ("pdr",    "PDR",        False, "RdYlGn_r",  "Lower is better"),
]


def build_histograms(data: dict, out_path: Path, ranges: dict, clip_pct: float,
                     outlier_n: int, hist_fmt: str = "pdf"):
    """Generate histogram subplot image for each metric."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[ERROR] matplotlib / numpy required: pip install matplotlib numpy")
        sys.exit(1)

    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("MCI Simulation - Metric Distributions across Coordinates",
                 fontsize=13, fontweight="bold")

    for ax, (metric, label, high_is_good, cmap_name, direction) in zip(axes, HIST_CONFIG):
        vals = [v[metric] for v in data.values() if v[metric] is not None]
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        vals = np.array(vals)
        cmap_vmin, cmap_vmax, true_min, true_max = ranges[metric]
        mean_val = vals.mean()
        std_val  = vals.std()
        n = len(vals)

        # Actual outlier value lists
        good_ids, bad_ids = get_outlier_ids(data, metric, outlier_n, high_is_good)
        good_vals = sorted([v[metric] for cid, v in data.items()
                            if cid in good_ids and v[metric] is not None])
        bad_vals  = sorted([v[metric] for cid, v in data.items()
                            if cid in bad_ids  and v[metric] is not None])
        outlier_vals = set(good_vals + bad_vals)

        # Bin count: fine enough so outliers fall into separate bins
        # Freedman-Diaconis IQR-based + minimum 60
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        if iqr > 0:
            fd_width = 2 * iqr / (n ** (1/3))
            n_bins = max(60, int(np.ceil((true_max - true_min) / fd_width)))
        else:
            n_bins = 60
        n_bins = min(n_bins, 120)  # upper limit

        # Bar color: bins containing outlier values -> outlier color, rest -> colormap
        counts, bin_edges = np.histogram(vals, bins=n_bins)
        cmap = plt.get_cmap(cmap_name)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if cmap_vmax > cmap_vmin:
            norm_centers = np.clip(
                (bin_centers - cmap_vmin) / (cmap_vmax - cmap_vmin), 0.0, 1.0)
        else:
            norm_centers = np.full_like(bin_centers, 0.5)

        for count, left, right, nc in zip(
                counts, bin_edges[:-1], bin_edges[1:], norm_centers):
            # Check if this bin contains any outlier values
            has_good = any(left <= v <= right for v in good_vals)
            has_bad  = any(left <= v <= right for v in bad_vals)
            if has_good:
                bar_color = OUTLIER_GOOD
            elif has_bad:
                bar_color = OUTLIER_BAD
            else:
                bar_color = cmap(nc)
            ax.bar(left, count, width=(right - left),
                   color=bar_color, edgecolor="white", linewidth=0.4, align="edge")

        # Mean vertical line + +/-1 std shading
        ax.axvspan(mean_val - std_val, mean_val + std_val,
                   alpha=0.13, color="gray")
        ax.axvline(mean_val, color="#222222", linewidth=1.8, linestyle="--")

        # Clipping boundary lines
        if clip_pct > 0:
            for xval in (cmap_vmin, cmap_vmax):
                ax.axvline(xval, color="#555555", linewidth=1.1,
                           linestyle=":", alpha=0.8)

        # Outlier rug plot: individual tick marks just above x-axis
        ymax = counts.max()
        if good_vals:
            ax.plot(good_vals, [-ymax * 0.04] * len(good_vals),
                    marker='|', color=OUTLIER_GOOD, markersize=10,
                    linewidth=2, clip_on=False, zorder=5)
        if bad_vals:
            ax.plot(bad_vals, [-ymax * 0.04] * len(bad_vals),
                    marker='|', color=OUTLIER_BAD, markersize=10,
                    linewidth=2, clip_on=False, zorder=5)

        # Title: label + direction guide
        ax.set_title(f"{label}  ({direction})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Statistics summary box (always top-right)
        if clip_pct > 0:
            clip_str = (f"P{clip_pct:.0f} = {cmap_vmin:.3f}\n"
                        f"P{100-clip_pct:.0f} = {cmap_vmax:.3f}\n")
        else:
            clip_str = ""
        stats_text = (f"n = {n}\n"
                      f"min = {true_min:.3f}\n"
                      f"max = {true_max:.3f}\n"
                      f"mean = {mean_val:.3f}\n"
                      f"std  = {std_val:.3f}\n"
                      + clip_str.rstrip())
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))

        # Manually construct legend handles
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch as MPatch
        legend_handles = [
            Line2D([], [], color="#222222", linewidth=1.8, linestyle="--",
                   label=f"mean = {mean_val:.3f}"),
            MPatch(facecolor="gray", alpha=0.3, label=f"±1 std ({std_val:.3f})"),
        ]
        if clip_pct > 0:
            legend_handles.append(
                Line2D([], [], color="#555555", linewidth=1.1, linestyle=":",
                       label=f"P{clip_pct:.0f}/P{100-clip_pct:.0f} clipping"))
        if outlier_n > 0:
            legend_handles += [
                Line2D([], [], color=OUTLIER_GOOD, linewidth=1.5, linestyle="--",
                       label=f"★ Top Outliers (top {outlier_n})"),
                Line2D([], [], color=OUTLIER_BAD,  linewidth=1.5, linestyle="--",
                       label=f"▼ Bottom Outliers (bottom {outlier_n})"),
            ]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower center",
                  bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=True)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    hist_path_pdf = out_path.with_name(out_path.stem + "_hist.pdf")
    hist_path_png = out_path.with_name(out_path.stem + "_hist.png")
    fig.savefig(str(hist_path_pdf), bbox_inches="tight")
    fig.savefig(str(hist_path_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved: {hist_path_pdf}")
    print(f"  Histogram saved: {hist_path_png}")


# ---------------------------------------------------------------------------
# Per-rule data collection
# ---------------------------------------------------------------------------

RULE_METRICS = [
    ("reward", "The expected number\nof survivors", True),
    ("time",   "The response completion\ntime (min)", False),
    ("pdr",    "PDR", False),
]

FACTOR_NAMES = ["priority", "hos_select", "red_mode", "yellow_mode"]
FACTOR_LABELS = {
    "priority":    "Patient Prioritization",
    "hos_select":  "Hospital Selection",
    "red_mode":    "Transport Mode Selection (Red)",
    "yellow_mode": "Transport Mode Selection (Yellow)",
}
PRIORITY_LEVELS   = ["START", "ReSTART"]
HOS_SELECT_LEVELS = ["RedOnly", "YellowNearest"]
MODE_LEVELS       = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
MODE_DISPLAY      = ["UAV-only", "UAV-first", "AMB-first", "AMB-only"]


def _parse_rule_name(rule_str: str):
    """Parse 'START, RedOnly, Red OnlyUAV, Yellow Both_UAVFirst' into factor dict."""
    parts = [p.strip() for p in rule_str.split(",")]
    if len(parts) != 4:
        return None
    return {
        "priority":    parts[0],
        "hos_select":  parts[1],
        "red_mode":    parts[2].replace("Red ", ""),
        "yellow_mode": parts[3].replace("Yellow ", ""),
    }


def collect_rule_data(progress: dict, results_dir: Path) -> dict | None:
    """
    Collect per-rule mean values across all successful scenarios.
    Returns: {rule_name: {"reward": mean, "time": mean, "pdr": mean, "factors": {...},
                          "reward_vals": [...], "time_vals": [...], "pdr_vals": [...]}}
    """
    import numpy as np

    statuses = progress.get("statuses", {})

    # Find all possible result dirs (main + fail rerun)
    result_dirs = [results_dir]
    parent = results_dir.parent
    for d in parent.iterdir():
        if d.is_dir() and d != results_dir and "fail" in d.name:
            result_dirs.append(d)

    # Collect: rule_index -> list of values per scenario
    rule_rewards = {}  # idx -> list
    rule_times   = {}
    rule_pdrs    = {}
    rule_names   = {}
    n_scenarios  = 0

    for cid, v in statuses.items():
        if v.get("status") != "done" or not v.get("sim_ok"):
            continue
        config_path = v.get("config_path", "")
        m = re.search(r'\([\d.]+,[\d.]+\)', config_path)
        if not m:
            continue
        coord_str = m.group(0)

        stat_path = None
        for rd in result_dirs:
            candidate = rd / coord_str / f"results_{coord_str}_stat.txt"
            if candidate.exists():
                stat_path = candidate
                break
        if stat_path is None:
            continue

        rows = []
        names = []
        with open(stat_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 4:
                    try:
                        names.append(parts[0])
                        rows.append(float(parts[1]))
                    except ValueError:
                        continue

        n = len(rows)
        if n < 3:
            continue
        n_rules = round(n / 5)
        if n_rules < 1:
            continue

        for i in range(n_rules):
            if i not in rule_rewards:
                rule_rewards[i] = []
                rule_times[i]   = []
                rule_pdrs[i]    = []
                rule_names[i]   = names[i]
            rule_rewards[i].append(rows[i])
            rule_times[i].append(rows[n_rules + i])
            rule_pdrs[i].append(rows[2 * n_rules + i])

        n_scenarios += 1

    if n_scenarios == 0:
        return None

    result = {}
    for i in sorted(rule_names.keys()):
        name = rule_names[i]
        factors = _parse_rule_name(name)
        if factors is None:
            continue
        result[name] = {
            "reward": float(np.mean(rule_rewards[i])),
            "time":   float(np.mean(rule_times[i])),
            "pdr":    float(np.mean(rule_pdrs[i])),
            "reward_std": float(np.std(rule_rewards[i])),
            "time_std":   float(np.std(rule_times[i])),
            "pdr_std":    float(np.std(rule_pdrs[i])),
            "reward_vals": rule_rewards[i],
            "time_vals":   rule_times[i],
            "pdr_vals":    rule_pdrs[i],
            "factors": factors,
            "n": len(rule_rewards[i]),
        }

    print(f"  Rule data: {len(result)} rules × {n_scenarios} scenarios")
    return result


# ---------------------------------------------------------------------------
# Per-rule heatmap visualization
# ---------------------------------------------------------------------------

def build_rule_heatmaps(rule_data: dict, out_path: Path):
    """
    Generate heatmap matrix: 3 metrics (rows) × 4 panels (priority × hos_select).
    Each panel is a 4×4 heatmap (red_mode rows × yellow_mode cols).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    metrics = RULE_METRICS
    panel_combos = [
        (p, h) for p in PRIORITY_LEVELS for h in HOS_SELECT_LEVELS
    ]

    fig = plt.figure(figsize=(22, 14))
    outer = gridspec.GridSpec(len(metrics), len(panel_combos),
                              wspace=0.30, hspace=0.45,
                              left=0.08, right=0.91, top=0.86, bottom=0.08)

    # Precompute global min/max per metric for consistent colorbar
    metric_ranges = {}
    for mk, mlabel, _ in metrics:
        vals = [r[mk] for r in rule_data.values()]
        metric_ranges[mk] = (min(vals), max(vals))

    for row, (mk, mlabel, high_is_good) in enumerate(metrics):
        cmap_name = "GnBu" if high_is_good else "GnBu_r"
        vmin, vmax = metric_ranges[mk]

        for col, (pri, hos) in enumerate(panel_combos):
            ax = fig.add_subplot(outer[row, col])

            # Build 4×4 grid
            grid = np.full((4, 4), np.nan)
            for rname, rdata in rule_data.items():
                f = rdata["factors"]
                if f["priority"] != pri or f["hos_select"] != hos:
                    continue
                r_idx = MODE_LEVELS.index(f["red_mode"])
                y_idx = MODE_LEVELS.index(f["yellow_mode"])
                grid[r_idx, y_idx] = rdata[mk]

            im = ax.imshow(grid, cmap=cmap_name, vmin=vmin, vmax=vmax,
                           aspect="auto")

            # Annotate cells
            for r_i in range(4):
                for y_i in range(4):
                    v = grid[r_i, y_i]
                    if np.isnan(v):
                        continue
                    # Text color: black on light cells, white on dark
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    if high_is_good:
                        text_color = "white" if norm_v > 0.55 else "black"
                    else:
                        text_color = "white" if norm_v < 0.45 else "black"
                    fmt = f"{v:.1f}" if mk == "time" else f"{v:.2f}" if mk == "pdr" else f"{v:.2f}"
                    ax.text(y_i, r_i, fmt, ha="center", va="center",
                            fontsize=15, fontweight="bold", color=text_color)

            # Axis labels
            ax.set_xticks(range(4))
            ax.set_xticklabels(MODE_DISPLAY, fontsize=14)
            ax.set_yticks(range(4))
            ax.set_yticklabels(MODE_DISPLAY, fontsize=14)

            if row == len(metrics) - 1:
                ax.set_xlabel("Transport Mode Selection (Yellow)", fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Transport Mode Selection (Red)", fontsize=16, fontweight="bold")

            # Panel title
            ax.set_title(f"{pri} × {hos}", fontsize=16, fontweight="bold", pad=8)

        # Colorbar for this metric row
        cbar_ax = fig.add_axes([0.935, 0.06 + (len(metrics) - 1 - row) * 0.30,
                                0.012, 0.24])
        sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.set_ylabel(mlabel, fontsize=16, fontweight="bold")
        cbar.ax.tick_params(labelsize=14)

    fig.suptitle("Rule Performance Heatmap — Mean across Scenarios\n"
                 "(rows: Transport Mode Selection for Red, cols: Transport Mode Selection for Yellow)",
                 fontsize=20, fontweight="bold", y=0.97)

    for fmt, dpi in [("pdf", None), ("png", 300)]:
        p = out_path.with_name(out_path.stem + f"_rule_heatmap.{fmt}")
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        print(f"  Rule heatmap saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Factor main effects visualization
# ---------------------------------------------------------------------------

def _compute_eta_squared(rule_data: dict, factors_info, metrics):
    """
    Balanced full factorial ANOVA: SS decomposition for η² per factor × metric.
    η² = SS_factor / SS_total  (proportion of total variance explained by the factor)
    """
    import numpy as np

    eta_sq = {}  # (fname, metric) -> float
    for mk, _, _ in metrics:
        all_means = np.array([r[mk] for r in rule_data.values()])
        grand_mean = all_means.mean()
        ss_total = float(np.sum((all_means - grand_mean) ** 2))

        for fname, _, levels in factors_info:
            if ss_total == 0:
                eta_sq[(fname, mk)] = 0.0
                continue
            ss_factor = 0.0
            for lev in levels:
                group = [r[mk] for r in rule_data.values()
                         if r["factors"][fname] == lev]
                n_k = len(group)
                lev_mean = np.mean(group)
                ss_factor += n_k * (lev_mean - grand_mean) ** 2
            eta_sq[(fname, mk)] = float(ss_factor / ss_total)

    return eta_sq


def build_main_effects(rule_data: dict, out_path: Path):
    """
    Main effects plot: for each factor, show marginal mean per level across metrics.
    Effect size: ANOVA η² (eta-squared) from balanced full factorial SS decomposition.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = RULE_METRICS
    factors_info = [
        ("priority",    "Patient Prioritization",           PRIORITY_LEVELS),
        ("hos_select",  "Hospital Selection",               HOS_SELECT_LEVELS),
        ("red_mode",    "Transport Mode Selection (Red)",    MODE_LEVELS),
        ("yellow_mode", "Transport Mode Selection (Yellow)", MODE_LEVELS),
    ]

    # η² via ANOVA SS decomposition
    eta_sq = _compute_eta_squared(rule_data, factors_info, metrics)

    # Marginal means for plotting
    marginal = {}
    for fname, flabel, levels in factors_info:
        marginal[fname] = {}
        for lev in levels:
            matching = [r for r in rule_data.values()
                        if r["factors"][fname] == lev]
            marginal[fname][lev] = {}
            for mk, _, _ in metrics:
                all_vals = [r[mk] for r in matching]
                marginal[fname][lev][mk] = {
                    "mean": float(np.mean(all_vals)),
                    "std":  float(np.std(all_vals)),
                }

    # Print η² summary table
    print("\n  ANOVA η² (eta-squared) — proportion of variance explained:")
    header = f"  {'Factor':<22}"
    for mk, ml, _ in metrics:
        header += f"  {ml:>12}"
    print(header)
    for fname, flabel, _ in factors_info:
        row = f"  {flabel:<22}"
        for mk, _, _ in metrics:
            row += f"  {eta_sq[(fname, mk)]:>11.4f}"
        print(row)
    # Total
    row_total = f"  {'Σ main effects':<22}"
    for mk, _, _ in metrics:
        s = sum(eta_sq[(fn, mk)] for fn, _, _ in factors_info)
        row_total += f"  {s:>11.4f}"
    print(row_total)

    fig, axes = plt.subplots(len(metrics), len(factors_info),
                              figsize=(22, 12), sharey="row")

    colors = {"priority": ["#00695C", "#80CBC4"],
              "hos_select": ["#00695C", "#80CBC4"],
              "red_mode": ["#004D40", "#00695C", "#4DB6AC", "#B2DFDB"],
              "yellow_mode": ["#004D40", "#00695C", "#4DB6AC", "#B2DFDB"]}

    for row, (mk, mlabel, high_is_good) in enumerate(metrics):
        for col, (fname, flabel, levels) in enumerate(factors_info):
            ax = axes[row, col]
            means = [marginal[fname][lev][mk]["mean"] for lev in levels]
            stds  = [marginal[fname][lev][mk]["std"]  for lev in levels]

            x = np.arange(len(levels))
            bars = ax.bar(x, means, width=0.6, color=colors[fname][:len(levels)],
                          edgecolor="white", linewidth=0.8, zorder=3)

            # Value annotation on bars
            for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
                fmt = f"{m:.1f}" if mk == "time" else f"{m:.4f}" if mk == "pdr" else f"{m:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=15, fontweight="bold")

            # Highlight best level — bold red border + star marker
            if high_is_good:
                best_idx = int(np.argmax(means))
            else:
                best_idx = int(np.argmin(means))
            bars[best_idx].set_edgecolor("#D32F2F")
            bars[best_idx].set_linewidth(3.5)
            bars[best_idx].set_linestyle("solid")
            bx = bars[best_idx].get_x() + bars[best_idx].get_width() / 2
            by = bars[best_idx].get_height()
            ax.annotate("★ Best", xy=(bx, by), fontsize=15, fontweight="bold",
                        color="#D32F2F", ha="center", va="bottom",
                        xytext=(0, 16), textcoords="offset points")

            # η² annotation (ANOVA-based)
            eta = eta_sq[(fname, mk)]
            ax.text(0.98, 0.95,
                    f"η² = {eta:.4f}\n({eta*100:.1f}%)",
                    transform=ax.transAxes, fontsize=15, fontweight="bold",
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E0F2F1",
                              edgecolor="#00695C", linewidth=1.5, alpha=0.95))

            if levels is MODE_LEVELS:
                display_levels = MODE_DISPLAY
            else:
                display_levels = levels
            ax.set_xticks(x)
            ax.set_xticklabels(display_levels, fontsize=15, rotation=0)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

            ax.tick_params(axis="y", labelsize=14)
            if row == 0:
                ax.set_title(flabel, fontsize=17, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=16, fontweight="bold")

            # Tight y-axis range for readability
            val_range = max(means) - min(means)
            pad = max(val_range * 0.5, np.mean(means) * 0.005)
            ax.set_ylim(min(means) - pad, max(means) + pad * 1.5)

    fig.suptitle("Factor Main Effects — Marginal Mean across Scenarios\n"
                 "(★ best level, η² = ANOVA eta-squared)",
                 fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.04, 0, 1, 0.94])

    for fmt, dpi in [("pdf", None), ("png", 300)]:
        p = out_path.with_name(out_path.stem + f"_rule_effects.{fmt}")
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        print(f"  Main effects saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unified rule analysis invocation
# ---------------------------------------------------------------------------

def build_rule_analysis(progress: dict, results_dir: Path, out_path: Path):
    """Collect rule-level data and generate heatmap + main effects figures."""
    rule_data = collect_rule_data(progress, results_dir)
    if rule_data is None:
        print("  [WARN] No rule data found — skipping rule analysis.")
        return None
    build_rule_heatmaps(rule_data, out_path)
    build_main_effects(rule_data, out_path)
    return rule_data


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

MODES = [
    ("reward", "Reward",     True),   # True  = higher is better
    ("time",   "Time (min)", False),  # False = lower is better
    ("pdr",    "PDR",        False),
]

GRAD_HIGH = "linear-gradient(to right, #d7191c, #fdae61, #ffffbf, #a6d96a, #1a9641)"
GRAD_LOW  = "linear-gradient(to right, #1a9641, #a6d96a, #ffffbf, #fdae61, #d7191c)"

# Dedicated outlier colors (top/bottom N)
OUTLIER_GOOD = "#1565C0"   # dark blue - good outliers (top N)
OUTLIER_BAD  = "#7B1FA2"   # dark purple - bad outliers (bottom N)


def build_map(data: dict, out_path: Path, ranges: dict, clip_pct: float, outlier_n: int):
    try:
        import folium
    except ImportError:
        print("[ERROR] folium required: pip install folium")
        sys.exit(1)

    # Map center
    all_lats = [v["lat"] for v in data.values()]
    all_lons = [v["lon"] for v in data.values()]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap",
                   zoom_delta=0.5, zoom_snap=0.5)

    clip_note = f"" if clip_pct > 0 else ""

    # -----------------------------------------------------------------------
    # Precompute per-mode marker data + colorbar info
    # -----------------------------------------------------------------------
    js_data = {}

    for metric, label, high_is_good in MODES:
        vmin, vmax, _, _ = ranges[metric]
        valid_vals = [v[metric] for v in data.values() if v[metric] is not None]
        good_ids, bad_ids = get_outlier_ids(data, metric, outlier_n, high_is_good)

        markers = []
        good_list, bad_list = [], []   # Outlier detail lists (for legend)
        for cid, v in sorted(data.items(), key=lambda x: int(x[0])):
            val = v[metric]
            if val is None:
                markers.append({
                    "lat": v["lat"], "lon": v["lon"],
                    "color": "#222222",
                    "radius": 5, "opacity": 0.7, "weight": 1,
                    "popup": f"coord_id={cid}<br>({v['lat']:.6f}, {v['lon']:.6f})<br>No data",
                    "tooltip": f"ID:{cid} | No data",
                })
            else:
                if cid in good_ids:
                    color, tag = OUTLIER_GOOD, " ★Top"
                    good_list.append({"cid": cid, "lat": v["lat"],
                                      "lon": v["lon"], "val": val})
                elif cid in bad_ids:
                    color, tag = OUTLIER_BAD, " ▼Bottom"
                    bad_list.append({"cid": cid, "lat": v["lat"],
                                     "lon": v["lon"], "val": val})
                else:
                    color, tag = compute_color(val, vmin, vmax, high_is_good), ""

                markers.append({
                    "lat": v["lat"], "lon": v["lon"],
                    "color": color,
                    "radius": 5, "opacity": 0.9, "weight": 1,
                    "popup": (f"coord_id={cid}<br>"
                              f"({v['lat']:.6f}, {v['lon']:.6f})<br>"
                              f"{label}: {val:.4f}{tag}"),
                    "tooltip": f"ID:{cid} | {label}: {val:.4f}{tag}",
                })

        # Outlier lists: good in descending order, bad in ascending order (most extreme first)
        good_list.sort(key=lambda x: x["val"],
                       reverse=high_is_good)
        bad_list.sort(key=lambda x: x["val"],
                      reverse=not high_is_good)

        no_data = sum(1 for mk in markers if mk["color"] == "#222222")
        print(f"  {label}: valid={len(valid_vals)}, no_data={no_data}, "
              f"top_outliers={len(good_ids)}, bottom_outliers={len(bad_ids)}")

        if high_is_good:
            gradient    = GRAD_HIGH
            left_label  = f"{vmin:.3f} (Bad)"
            right_label = f"{vmax:.3f} (Good)"
        else:
            gradient    = GRAD_LOW
            left_label  = f"{vmin:.3f} (Good)"
            right_label = f"{vmax:.3f} (Bad)"

        js_data[metric] = {
            "markers":      markers,
            "label":        label,
            "gradient":     gradient,
            "left_label":   left_label,
            "right_label":  right_label,
            "good_outlier": len(good_ids),
            "bad_outlier":  len(bad_ids),
            "good_list":    good_list,
            "bad_list":     bad_list,
            "outlier_good_color": OUTLIER_GOOD,
            "outlier_bad_color":  OUTLIER_BAD,
        }

    # -----------------------------------------------------------------------
    # Inject custom HTML/JS (LayerControl not used)
    # -----------------------------------------------------------------------
    map_var   = m.get_name()
    data_json = json.dumps(js_data, ensure_ascii=False)

    custom_html = f"""
<!-- ===== MCI Custom Visualization UI ===== -->
<style>
  #mci-buttons {{
    position: fixed; top: 80px; right: 10px; z-index: 9999;
    background: white; padding: 10px 14px; border-radius: 8px;
    border: 2px solid #aaa; font-family: sans-serif; font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,.3);
  }}
  #mci-buttons b {{ display:block; margin-bottom:6px; }}
  .mci-btn {{
    margin: 2px; padding: 5px 14px; cursor: pointer;
    border: none; border-radius: 4px; font-size: 13px;
    background: #ddd; color: #333; transition: background .2s;
  }}
  .mci-btn.active {{ background: #2c7bb6; color: white; }}

  #mci-colorbar {{
    position: fixed; top: 50%; left: 10px; transform: translateY(-50%); z-index: 9999;
    background: white; padding: 10px 14px; border-radius: 6px;
    border: 2px solid #aaa; font-family: sans-serif; font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,.3); width: 240px;
    max-height: 80vh; overflow-y: auto;
  }}
  #mci-colorbar b {{ display:block; margin-bottom:6px; font-size:15px; font-weight:700; color:#222; }}
  #cb-gradient {{ height: 14px; border-radius: 3px; margin-bottom: 4px; }}
  #cb-labels {{ display:flex; justify-content:space-between; font-size:12px; font-weight:600; color:#444; }}
  #cb-outliers {{ margin-top: 8px; }}
  .cb-dot {{
    display: inline-block; width: 11px; height: 11px;
    border-radius: 50%; margin-right: 5px;
    border: 1.5px solid rgba(0,0,0,0.3); vertical-align: middle;
  }}
  .cb-details {{
    margin-top: 5px; border-radius: 4px; overflow: hidden;
    border: 1px solid #ccc;
  }}
  .cb-details summary {{
    cursor: pointer; padding: 5px 8px;
    font-size: 12.5px; font-weight: 700; color: #333;
    list-style: none; user-select: none;
    display: flex; align-items: center; gap: 5px;
  }}
  .cb-details summary::-webkit-details-marker {{ display:none; }}
  .cb-details summary::before {{
    content: '▶'; font-size: 9px; transition: transform .2s;
  }}
  .cb-details[open] summary::before {{ transform: rotate(90deg); }}
  .cb-detail-body {{
    padding: 5px 7px 7px 7px;
    font-size: 11.5px; line-height: 1.9;
    border-top: 1px solid #eee;
    max-height: 160px; overflow-y: auto;
    background: #fafafa;
  }}
  .cb-detail-row {{
    display: flex; justify-content: space-between;
    padding: 2px 0; border-bottom: 1px solid #eee;
  }}
  .cb-detail-row:last-child {{ border-bottom: none; }}
  .cb-idx {{ color: #444; min-width: 30px; font-weight: 600; }}
  .cb-coord {{ color: #555; flex:1; text-align:center; font-size:10.5px; }}
  .cb-val {{ font-weight: 700; color: #222; min-width: 55px; text-align:right; }}
  .cb-nodata {{
    margin-top: 6px; font-size: 12px; font-weight: 500;
    display: flex; align-items: center; gap: 5px;
  }}
</style>

<div id="mci-buttons">
  <b>Visualization Mode</b>
  <button class="mci-btn active" id="btn-reward" onclick="mciSwitch('reward')">Reward</button>
  <button class="mci-btn"        id="btn-time"   onclick="mciSwitch('time')">Time</button>
  <button class="mci-btn"        id="btn-pdr"    onclick="mciSwitch('pdr')">PDR</button>
  <hr style="margin:8px 0 6px 0;border:none;border-top:1px solid #ccc;">
  <b>Map Tile</b>
  <button class="mci-btn active" id="btn-tile-osm"    onclick="mciTile('osm')">OpenStreetMap</button>
  <button class="mci-btn"        id="btn-tile-carto"   onclick="mciTile('carto')">CartoDB</button>
</div>

<div id="mci-colorbar">
  <b id="cb-title">Reward</b>
  <div id="cb-gradient"></div>
  <div id="cb-labels">
    <span id="cb-left"></span>
    <span id="cb-right"></span>
  </div>
  <div id="cb-outliers">
    <details class="cb-details" id="details-good">
      <summary id="sum-good">
        <span class="cb-dot" id="dot-good"></span>
        <span id="cb-good-label"></span>
      </summary>
      <div class="cb-detail-body" id="list-good"></div>
    </details>
    <details class="cb-details" id="details-bad" style="margin-top:4px;">
      <summary id="sum-bad">
        <span class="cb-dot" id="dot-bad"></span>
        <span id="cb-bad-label"></span>
      </summary>
      <div class="cb-detail-body" id="list-bad"></div>
    </details>
    <div class="cb-nodata" id="cb-nodata-row">
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none;">
        <input type="checkbox" id="nodata-toggle" checked onchange="mciToggleNoData()">
        <span class="cb-dot" style="background:#222222;"></span>No Data
      </label>
    </div>
  </div>
</div>

<script>
(function() {{
  var modesData = {data_json};
  var currentLayer = null;
  var currentMode  = 'reward';
  var currentTileLayer = null;
  var tileUrls = {{
    'osm':   'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    'carto':  'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png'
  }};
  var tileAttrs = {{
    'osm':   '&copy; OpenStreetMap contributors',
    'carto': '&copy; OpenStreetMap contributors &copy; CARTO'
  }};

  function getMap() {{
    return window["{map_var}"];
  }}

  function buildRows(items, valLabel) {{
    if (!items || items.length === 0)
      return '<div style="color:#999;padding:2px 0;">None</div>';
    return items.map(function(it) {{
      return '<div class="cb-detail-row">'
        + '<span class="cb-idx">#' + it.cid + '</span>'
        + '<span class="cb-coord">(' + it.lat.toFixed(4) + ', ' + it.lon.toFixed(4) + ')</span>'
        + '<span class="cb-val">' + it.val.toFixed(4) + '</span>'
        + '</div>';
    }}).join('');
  }}

  window.mciSwitch = function(mode) {{
    var mapObj = getMap();
    if (!mapObj) return;

    currentMode = mode;

    if (currentLayer) {{
      mapObj.removeLayer(currentLayer);
      currentLayer = null;
    }}

    var d = modesData[mode];
    var showNoData = document.getElementById('nodata-toggle').checked;
    var lg = L.layerGroup();
    d.markers.forEach(function(mk) {{
      if (!showNoData && mk.color === '#222222') return;
      L.circleMarker([mk.lat, mk.lon], {{
        radius:      mk.radius,
        color:       mk.color,
        fillColor:   mk.color,
        fillOpacity: mk.opacity,
        weight:      mk.weight || 1,
      }}).bindPopup(mk.popup).bindTooltip(mk.tooltip).addTo(lg);
    }});
    lg.addTo(mapObj);
    currentLayer = lg;

    // colorbar
    document.getElementById('cb-title').textContent = d.label;
    document.getElementById('cb-gradient').style.background = d.gradient;
    document.getElementById('cb-left').textContent  = d.left_label;
    document.getElementById('cb-right').textContent = d.right_label;

    // outlier legend + collapsible list
    document.getElementById('dot-good').style.background = d.outlier_good_color;
    document.getElementById('dot-bad').style.background  = d.outlier_bad_color;
    document.getElementById('cb-good-label').textContent =
      '★ Top Outliers (' + d.good_outlier + ')';
    document.getElementById('cb-bad-label').textContent =
      '▼ Bottom Outliers (' + d.bad_outlier + ')';
    document.getElementById('list-good').innerHTML = buildRows(d.good_list, d.label);
    document.getElementById('list-bad').innerHTML  = buildRows(d.bad_list,  d.label);

    // button style
    ['reward','time','pdr'].forEach(function(m) {{
      document.getElementById('btn-' + m).className =
        'mci-btn' + (m === mode ? ' active' : '');
    }});
  }};

  window.mciToggleNoData = function() {{
    window.mciSwitch(currentMode);
  }};

  window.mciTile = function(tileKey) {{
    var mapObj = getMap();
    if (!mapObj) return;
    if (currentTileLayer) {{
      mapObj.removeLayer(currentTileLayer);
    }}
    currentTileLayer = L.tileLayer(tileUrls[tileKey], {{
      attribution: tileAttrs[tileKey],
      maxZoom: 19
    }}).addTo(mapObj);
    ['osm','carto'].forEach(function(k) {{
      document.getElementById('btn-tile-' + k).className =
        'mci-btn' + (k === tileKey ? ' active' : '');
    }});
  }};

  function init() {{
    if (window["{map_var}"]) {{
      // Remove default tile layer and track our own
      var mapObj = getMap();
      mapObj.eachLayer(function(layer) {{
        if (layer instanceof L.TileLayer) {{
          mapObj.removeLayer(layer);
        }}
      }});
      currentTileLayer = L.tileLayer(tileUrls['osm'], {{
        attribution: tileAttrs['osm'],
        maxZoom: 19
      }}).addTo(mapObj);
      window.mciSwitch('reward');
    }} else {{
      setTimeout(init, 100);
    }}
  }}
  init();
}})();
</script>
"""

    m.get_root().html.add_child(folium.Element(custom_html))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{'='*50}")
    print(f"  MCI Results Map Visualization (single HTML)")
    print(f"{'='*50}")

    coords_path   = resolve(args.coords)
    progress_path = resolve(args.progress)

    coords   = load_coords(coords_path)
    progress = load_progress(progress_path)

    if args.results_dir:
        results_dir = resolve(args.results_dir)
    else:
        results_dir = find_results_dir(progress)
        if results_dir is None:
            print("[ERROR] Auto-detection of results folder failed. Use --results-dir option.")
            sys.exit(1)
    print(f"  results dir: {results_dir}")

    out_path = resolve(args.out) if args.out else _SCRIPT_DIR / "coords_map.html"

    print("  Collecting data...")
    data = collect_data(coords, progress, results_dir)

    print("  Computing colormap ranges...")
    ranges = compute_ranges(data, args.clip_pct)

    print("  Building map...")
    build_map(data, out_path, ranges, args.clip_pct, args.outlier_n)

    print("  Building histograms...")
    build_histograms(data, out_path, ranges, args.clip_pct, args.outlier_n,
                     hist_fmt=args.hist_format)

    print("  Building rule analysis...")
    build_rule_analysis(progress, results_dir, out_path)

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
