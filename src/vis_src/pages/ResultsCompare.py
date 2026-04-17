"""
Results comparison page.
- Loads multiple experiment/coordinate results (results_*.txt) and provides
  multi-dimensional visualization.
- Batch generation features have been moved to the Generate tab.
"""
from pathlib import Path
from datetime import timedelta, timezone
import math
import re
import sys

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import plotly.colors as pc


def _detect_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "src" / "sce_src" / "orchestrator.py").is_file():
            return parent
    return here


def _detect_cloud_base_path() -> str:
    for cand in ("/mount/src/mci_adv", "/mount/src/mci_adv/Simul_team"):
        p = Path(cand)
        if p.is_dir() and (p / "scenarios").is_dir():
            return cand
    return ""


REPO_ROOT = _detect_repo_root()
ORCHESTRATOR_DIR = REPO_ROOT / "src" / "sce_src"
if ORCHESTRATOR_DIR.is_dir():
    orch_path = str(ORCHESTRATOR_DIR)
    if orch_path not in sys.path:
        sys.path.insert(0, orch_path)

KST = timezone(timedelta(hours=9))
CLOUD_BASE_PATH = _detect_cloud_base_path()
IS_CLOUD = bool(CLOUD_BASE_PATH)
DEFAULT_LOCAL_BASE_PATH = str(REPO_ROOT) if (REPO_ROOT / "scenarios").is_dir() else ""

RAW_BLOCK_NAMES = ["Reward", "Time", "PDR", "Reward_woG", "PDR_woG"]
RAW_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def load_label_map_file(base_path: str) -> tuple[dict, dict]:
    """
    scenarios/label_map.csv -> {(exp_id, coord): label}, {coord: label}
    """
    path = Path(base_path) / "scenarios" / "label_map.csv"
    legacy_path = Path(base_path) / "results" / "label_map.csv"
    frames = []
    if path.exists():
        try:
            frames.append(pd.read_csv(path, encoding="utf-8"))
        except Exception:
            pass
    if legacy_path.exists():
        try:
            frames.append(pd.read_csv(legacy_path, encoding="utf-8"))
        except Exception:
            pass
    if not frames:
        return {}, {}
    df = pd.concat(frames, ignore_index=True)
    if {"exp_id", "coord"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["exp_id", "coord"], keep="first")
    exp_coord = {}
    coord_only = {}
    for _, row in df.iterrows():
        coord = str(row.get("coord", "")).strip()
        label = str(row.get("label", "")).strip()
        exp_id = str(row.get("exp_id", "")).strip()
        if coord and label:
            if exp_id:
                exp_coord[(exp_id, coord)] = label
            coord_only[coord] = label
    if legacy_path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    return exp_coord, coord_only


def _split_key_vals(line: str):
    m = re.search(RAW_FLOAT, line)
    if not m:
        return line.strip(), []
    key = line[: m.start()].strip().rstrip(",")
    vals = [float(x) for x in re.findall(RAW_FLOAT, line[m.start() :])]
    return key, vals


def _split_factors(rule_label: str):
    parts = [p.strip() for p in rule_label.split(",")]
    phase = parts[0] if len(parts) > 0 else ""
    red_policy = parts[1] if len(parts) > 1 else ""

    def pick_mode(p: str, color: str):
        if not p:
            return ""
        for k in ("OnlyUAV", "OnlyAMB", "Both_UAVFirst", "Both_AMBFirst"):
            if k in p:
                return k
        toks = p.replace(",", " ").split()
        try:
            cidx = toks.index(color)
            return "_".join(toks[cidx + 1 :]) if cidx < len(toks) - 1 else ""
        except ValueError:
            return "_".join(toks[1:]) if len(toks) > 1 else ""

    red_action = pick_mode(parts[2] if len(parts) > 2 else "", "Red")
    yellow_action = pick_mode(parts[3] if len(parts) > 3 else "", "Yellow")
    return phase, red_policy, red_action, yellow_action


def _cuboid_mesh(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float):
    x1 = x0 + dx
    y1 = y0 + dy
    z1 = z0 + dz
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 3, 7]
    j = [1, 2, 3, 2, 5, 3, 5, 6, 6, 7, 7, 4]
    k = [2, 3, 1, 5, 6, 7, 6, 7, 4, 4, 0, 0]
    return x, y, z, i, j, k


@st.cache_data(ttl=600, show_spinner=False)
def parse_raw_results(raw_path: str) -> pd.DataFrame:
    """
    results_(lat,lon).txt -> long DF
    columns: ['exp_id','coord','rule','Phase','RedPolicy','RedAction','YellowAction','run','metric','value']
    """
    if not (raw_path and Path(raw_path).is_file()):
        return pd.DataFrame(columns=["rule","Phase","RedPolicy","RedAction","YellowAction","run","metric","value"])

    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            key, vals = _split_key_vals(raw)
            if not vals:
                continue
            rows.append((key, vals))
    if not rows:
        return pd.DataFrame(columns=["rule","Phase","RedPolicy","RedAction","YellowAction","run","metric","value"])

    first_key = rows[0][0]
    L = None
    for i in range(1, len(rows)):
        if rows[i][0] == first_key:
            L = i
            break
    if L is None:
        L = len(rows)

    n_blocks = int(math.ceil(len(rows) / L))
    out_recs = []
    for b in range(n_blocks):
        block = rows[b * L : (b + 1) * L]
        if not block:
            continue
        metric_name = RAW_BLOCK_NAMES[b] if b < len(RAW_BLOCK_NAMES) else f"Metric_{b+1}"
        R = len(block[0][1])
        for (label, vals) in block:
            Phase, RedPolicy, RedAction, YellowAction = _split_factors(label)
            rule = f"{Phase}, {RedPolicy}, Red {RedAction}, Yellow {YellowAction}".strip().replace("  ", " ")
            for r_idx, v in enumerate(vals, start=1):
                out_recs.append(
                    {
                        "rule": rule,
                        "Phase": Phase,
                        "RedPolicy": RedPolicy,
                        "RedAction": RedAction,
                        "YellowAction": YellowAction,
                        "run": r_idx,
                        "metric": metric_name,
                        "value": v,
                    }
                )
    df = pd.DataFrame.from_records(out_recs)
    if not df.empty:
        df["run"] = df["run"].astype(int)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def collect_results(base_path: str, selected_exps: list[str]) -> pd.DataFrame:
    root = Path(base_path) / "results"
    if not root.exists():
        return pd.DataFrame()
    frames = []
    for exp in selected_exps:
        exp_dir = root / exp
        if not exp_dir.is_dir():
            continue
        for coord_dir in exp_dir.iterdir():
            if not coord_dir.is_dir():
                continue
            raw_path = coord_dir / f"results_{coord_dir.name}.txt"
            if not raw_path.exists():
                continue
            df = parse_raw_results(str(raw_path))
            if df.empty:
                continue
            df["exp_id"] = exp
            df["coord"] = coord_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def list_results_experiments(base_path: str) -> list[str]:
    root = Path(base_path) / "results"
    if not root.is_dir():
        return []
    items = [p.name for p in root.iterdir() if p.is_dir()]
    items.sort(key=lambda n: (root / n).stat().st_mtime, reverse=True)
    return items


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Results Compare", page_icon="📈", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
.stApp {
    background: #141417;
}
[data-testid="stSidebar"] {
    background: #1c1c21 !important;
    border-right: 1px solid rgba(226, 160, 74, 0.08) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(28, 28, 33, 0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(226, 160, 74, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    color: #94a3b8 !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(226, 160, 74, 0.15) !important;
    color: #e2a04a !important;
    box-shadow: 0 1px 4px rgba(226, 160, 74, 0.1);
}
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid rgba(226, 160, 74, 0.3) !important;
    background: rgba(226, 160, 74, 0.12) !important;
    color: #e2a04a !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(226, 160, 74, 0.22) !important;
    border-color: rgba(226, 160, 74, 0.5) !important;
}
[data-baseweb="input"], [data-baseweb="select"] > div {
    background: rgba(20, 20, 23, 0.8) !important;
    border: 1px solid rgba(226, 160, 74, 0.1) !important;
    border-radius: 8px !important;
}
[data-testid="stMetric"] {
    background: rgba(28, 28, 33, 0.6);
    border: 1px solid rgba(226, 160, 74, 0.08);
    border-left: 3px solid #e2a04a;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stExpander"] {
    background: rgba(28, 28, 33, 0.5) !important;
    border: 1px solid rgba(226, 160, 74, 0.08) !important;
    border-radius: 10px !important;
}
hr { border-color: rgba(226, 160, 74, 0.08) !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(226, 160, 74, 0.25); border-radius: 4px; }
</style>""", unsafe_allow_html=True)

st.title("Simulation Results Comparison")
st.caption("PDR is shown as percent; PDR/Time axes are reversed so lower is better.")

# base_path
if "base_path_compare" not in st.session_state:
    st.session_state.base_path_compare = CLOUD_BASE_PATH if IS_CLOUD else DEFAULT_LOCAL_BASE_PATH

col_bp, = st.columns(1)
with col_bp:
    bp_input = st.text_input(
        "base_path (MCI_ADV root)",
        value=st.session_state.base_path_compare,
        placeholder="e.g. C:\\Users\\USER\\MCI_ADV",
        disabled=IS_CLOUD,
    )
    if st.button("Apply", key="btn_set_bp"):
        st.session_state.base_path_compare = bp_input.strip()
    if IS_CLOUD:
        st.info(f"Cloud mode detected: fixed to `{CLOUD_BASE_PATH}`")
bp = st.session_state.base_path_compare
if not bp or not Path(bp).is_dir():
    st.stop()

saved_label_exp, saved_label_coord = load_label_map_file(bp)
if saved_label_coord:
    st.caption(f"label_map.csv loaded: {len(saved_label_coord)} labels")

# Load saved label mapping file

# Label mapping input

# Experiment selection
experiments = list_results_experiments(bp)
sel_exps = st.multiselect("Select Result Folders (results/<exp_id>)", options=experiments, default=experiments[:5])

if not sel_exps:
    st.info("Set base_path and select result folders.")
    st.stop()

metric_choices = ["Reward", "PDR", "Time", "Reward_woG", "PDR_woG"]
metric_sel = st.selectbox("Primary Metric", metric_choices, index=0)

if st.button("Load Results", type="primary"):
    df_raw = collect_results(bp, sel_exps)
    st.session_state.df_compare_raw = df_raw

df_raw = st.session_state.get("df_compare_raw", pd.DataFrame())
if df_raw.empty:
    st.warning("No results loaded.")
    st.stop()

# Apply labels
def _resolve_label(row):
    coord = row["coord"]
    exp_id = row["exp_id"]
    if (exp_id, coord) in saved_label_exp:
        return saved_label_exp[(exp_id, coord)]
    if coord in saved_label_coord:
        return saved_label_coord[coord]
    return coord

df_raw["label"] = df_raw.apply(_resolve_label, axis=1)

df_m = df_raw[df_raw["metric"] == metric_sel].copy()
if df_m.empty:
    st.warning(f"{metric_sel}: No data available.")
    st.stop()

agg = (
    df_m.groupby(["label", "coord", "exp_id", "rule"])
    .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
    .reset_index()
)
st.success(f"{len(agg)} rule summaries loaded")

st.markdown("#### Metric Summary (Table)")
st.dataframe(agg, width='stretch', hide_index=True)

st.download_button(
    "Download Summary CSV",
    agg.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"results_compare_{metric_sel}.csv",
    mime="text/csv",
)

# Top-N bar with ±1 SD error bars
st.markdown("#### Top-N Bar Chart (Mean ± 1 SD)")
topN = st.slider("Top N", min_value=5, max_value=50, value=15, step=1)
lower_better = metric_sel in ("PDR", "Time", "PDR_woG")
top_rules = agg.sort_values("mean", ascending=lower_better).head(topN)
_tr = top_rules.sort_values("mean", ascending=not lower_better)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=_tr["rule"], x=_tr["mean"], orientation="h",
    error_x=dict(type="data", array=_tr["std"].fillna(0), visible=True),
    marker_color=[pc.qualitative.Set2[i % len(pc.qualitative.Set2)] for i in range(len(_tr))],
    hovertemplate="%{y}<br>mean=%{x:.4f} ± %{error_x.array:.4f}<extra></extra>",
))
fig_bar.update_layout(height=max(400, topN * 28), yaxis_title="Rule", xaxis_title=f"{metric_sel} (mean)",
                       margin=dict(l=0, r=20, t=30, b=40))
st.plotly_chart(fig_bar, width='stretch')

# Reward vs Time scatter plot
st.markdown("#### Reward vs Time (Mean) Scatter Plot")
if {"Reward", "Time"}.issubset(set(df_raw["metric"].unique())):
    pivot_rt = (
        df_raw[df_raw["metric"].isin(["Reward", "Time"])]
        .groupby(["exp_id", "coord", "label", "rule", "metric"])
        .agg(mean=("value", "mean"))
        .reset_index()
        .pivot_table(index=["exp_id", "coord", "label", "rule"], columns="metric", values="mean")
        .reset_index()
    )
    # Pareto frontier: minimize Time, maximize Reward
    _prt = pivot_rt.dropna(subset=["Time", "Reward"]).sort_values("Time").reset_index(drop=True)
    pareto_idx = []
    best_reward = -np.inf
    for _i, _r in _prt.iterrows():
        if _r["Reward"] > best_reward:
            best_reward = _r["Reward"]
            pareto_idx.append(_i)
    pareto_pts = _prt.loc[pareto_idx].sort_values("Time")

    scat = (
        alt.Chart(pivot_rt)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("Time:Q", title="Time (mean)"),
            y=alt.Y("Reward:Q", title="Reward (mean)"),
            color="label:N",
            shape="exp_id:N",
            tooltip=["exp_id", "label", "coord", "rule", "Reward", "Time"],
        )
        .properties(height=420)
        .interactive()
    )
    pareto_line = (
        alt.Chart(pareto_pts)
        .mark_line(color="red", strokeDash=[6, 3], strokeWidth=2)
        .encode(x="Time:Q", y="Reward:Q")
    )
    pareto_dots = (
        alt.Chart(pareto_pts)
        .mark_point(color="red", size=120, filled=True, shape="diamond")
        .encode(x="Time:Q", y="Reward:Q",
                tooltip=["rule", "Reward", "Time"])
    )
    st.altair_chart(scat + pareto_line + pareto_dots, width='stretch')
    st.caption("Red dashed line = Pareto frontier (Reward↑, Time↓).")
else:
    st.info("Reward/Time data not available; skipping scatter plot.")

# Heatmap (rule x label)
st.markdown("#### Label x Rule Heatmap")
hm_base = agg.copy()
hm_base["rule_short"] = hm_base["rule"].str.slice(0, 40)
heat = (
    alt.Chart(hm_base)
    .mark_rect()
    .encode(
        x=alt.X("rule_short:N", title="Rule", sort=None),
        y=alt.Y("label:N", title="Label"),
        color=alt.Color("mean:Q", title=f"{metric_sel} (mean)", scale=alt.Scale(scheme="blueorange")),
        tooltip=["label", "rule", "mean", "std", "n", "exp_id", "coord"],
    )
    .properties(height=400)
)
st.altair_chart(heat, width='stretch')

# Box plot with jitter overlay (Plotly)
st.markdown("#### Distribution by Label (Box Plot + Jitter)")
fig_box = go.Figure()
_labels_unique = sorted(df_m["label"].unique())
_box_colors = pc.qualitative.Set2
for _li, _lbl in enumerate(_labels_unique):
    _sub = df_m[df_m["label"] == _lbl]
    _clr = _box_colors[_li % len(_box_colors)]
    fig_box.add_trace(go.Box(
        y=_sub["value"], name=_lbl, marker_color=_clr,
        boxpoints="all", jitter=0.4, pointpos=-1.5,
        marker=dict(size=3, opacity=0.5),
        hovertext=_sub["rule"],
    ))
fig_box.update_layout(height=450, yaxis_title=metric_sel, showlegend=True,
                       margin=dict(l=50, r=20, t=30, b=40))
st.plotly_chart(fig_box, width='stretch')
# 3D compare (mean over all rules; x=PDR, y=Time, z=Reward)
st.markdown("#### Label 3D comparison (PDR/Time/Reward)")
metric_3d = ["PDR", "Time", "Reward"]
base_3d = df_raw[df_raw["metric"].isin(metric_3d)].copy()
if base_3d.empty:
    st.info("PDR/Time/Reward data not found; 3D comparison is unavailable.")
else:
    agg_rule = (
        base_3d.groupby(["label", "rule", "metric"]).agg(mean=("value", "mean")).reset_index()
    )
    pv_rule = (
        agg_rule.pivot_table(index=["label", "rule"], columns="metric", values="mean")
        .reindex(columns=metric_3d)
        .reset_index()
    )
    pv_rule = pv_rule.dropna(subset=metric_3d)
    if pv_rule.empty:
        st.info("PDR/Time/Reward data not found; 3D comparison is unavailable.")
    else:
        st.caption("Each label is split into rule-level cuboids. PDR is shown as percent; PDR/Time axes are reversed so lower is better.")
        label_ranges = (
            pv_rule.groupby("label")
            .agg(pdr_min=("PDR", "min"), pdr_max=("PDR", "max"),
                 time_min=("Time", "min"), time_max=("Time", "max"))
            .reset_index()
        )
        range_map = {
            r["label"]: {
                "pdr_min": float(r["pdr_min"]),
                "pdr_max": float(r["pdr_max"]),
                "time_min": float(r["time_min"]),
                "time_max": float(r["time_max"]),
            }
            for _, r in label_ranges.iterrows()
        }

        colors = pc.qualitative.Set3
        labels = list(pv_rule["label"].unique())
        color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(labels)}
        fig = go.Figure()
        seen_labels = set()
        for idx, row in pv_rule.iterrows():
            label = row["label"]
            rule = row["rule"]
            raw_pdr = float(row["PDR"])
            raw_time = float(row["Time"])
            raw_reward = float(row["Reward"])

            lr = range_map.get(label)
            pdr_range = (lr["pdr_max"] - lr["pdr_min"]) * 100.0 if lr else 0.0
            time_range = (lr["time_max"] - lr["time_min"]) if lr else 0.0
            dx = max(pdr_range * 0.06, 0.4)
            dy = max(time_range * 0.05, 1.0)

            x_center = raw_pdr * 100.0
            y_center = raw_time
            z_val = raw_reward
            z0 = 0.0 if z_val >= 0 else z_val
            dz = abs(z_val)
            x0 = x_center - dx / 2
            y0 = y_center - dy / 2
            x, y, z, i, j, k = _cuboid_mesh(x0, y0, z0, dx, dy, dz)
            color = color_map.get(label, colors[0])
            show_legend = label not in seen_labels
            if show_legend:
                seen_labels.add(label)

            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=0.85,
                    name=label,
                    showlegend=show_legend,
                    hovertext=(
                        f"label={label}<br>rule={rule}"
                        f"<br>PDR={raw_pdr:.4f} ({x_center:.1f}%)"
                        f"<br>Time={raw_time:.3f}<br>Reward={raw_reward:.3f}"
                    ),
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            height=650,
            legend_title_text="label",
            scene=dict(
                xaxis_title="PDR (%) (lower is better)",
                yaxis_title="Time (mean, lower is better)",
                zaxis_title="Reward (mean, higher is better)",
                xaxis=dict(autorange="reversed"),
                yaxis=dict(autorange="reversed"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, width='stretch')

# ========== Multi-metric pivot for new charts ==========
_multi_metrics = ["Reward", "Time", "PDR"]
_avail_metrics = set(df_raw["metric"].unique())
if set(_multi_metrics).issubset(_avail_metrics):
    _pv_multi = (
        df_raw[df_raw["metric"].isin(_multi_metrics)]
        .groupby(["label", "rule", "metric"])
        .agg(mean=("value", "mean"))
        .reset_index()
        .pivot_table(index=["label", "rule"], columns="metric", values="mean")
        .reindex(columns=_multi_metrics)
        .dropna()
        .reset_index()
    )
else:
    _pv_multi = pd.DataFrame()

# --- Parallel Coordinates ---
st.markdown("#### Parallel Coordinates (Reward / Time / PDR)")
if _pv_multi.empty:
    st.info("Reward/Time/PDR data not all available; skipping parallel coordinates.")
else:
    _pc_df = _pv_multi.copy()
    _pc_labels = sorted(_pc_df["label"].unique())
    _pc_label_num = {lbl: i for i, lbl in enumerate(_pc_labels)}
    _pc_df["_label_num"] = _pc_df["label"].map(_pc_label_num)
    _pc_colorscale = pc.qualitative.Set2
    _pc_cs = [[i / max(len(_pc_labels) - 1, 1), _pc_colorscale[i % len(_pc_colorscale)]] for i in range(len(_pc_labels))]

    fig_pc = go.Figure(data=go.Parcoords(
        line=dict(
            color=_pc_df["_label_num"],
            colorscale=_pc_cs,
            showscale=False,
        ),
        dimensions=[
            dict(label="Reward", values=_pc_df["Reward"]),
            dict(label="Time", values=_pc_df["Time"]),
            dict(label="PDR", values=_pc_df["PDR"]),
        ],
    ))
    fig_pc.update_layout(height=420, margin=dict(l=80, r=80, t=40, b=30))
    st.plotly_chart(fig_pc, width='stretch')
    st.caption("Color = label. Each line = one rule. "
               + ", ".join(f"{lbl} = color {i}" for i, lbl in enumerate(_pc_labels)))

# --- Radar Chart (Top-N rules) ---
st.markdown("#### Radar Chart (Top-N Rules)")
if _pv_multi.empty:
    st.info("Reward/Time/PDR data not all available; skipping radar chart.")
else:
    radar_n = st.slider("Radar Top N", min_value=3, max_value=20, value=8, step=1, key="radar_n")
    # Composite score for ranking: normalize each metric to [0,1], Reward↑ Time↓ PDR↓
    _rdr = _pv_multi.copy()
    for _col, _asc in [("Reward", False), ("Time", True), ("PDR", True)]:
        _mn, _mx = _rdr[_col].min(), _rdr[_col].max()
        if _mx - _mn > 1e-12:
            _norm = (_rdr[_col] - _mn) / (_mx - _mn)
            _rdr[f"{_col}_n"] = _norm if _asc else (1 - _norm)  # higher = better after normalization
        else:
            _rdr[f"{_col}_n"] = 0.5
    _rdr["_composite"] = (_rdr["Reward_n"] + _rdr["Time_n"] + _rdr["PDR_n"]) / 3
    _rdr_top = _rdr.sort_values("_composite", ascending=False).head(radar_n)

    _radar_axes = ["Reward_n", "Time_n", "PDR_n"]
    _radar_labels = ["Reward ↑", "Time ↓", "PDR ↓"]
    _radar_colors = pc.qualitative.Plotly
    fig_radar = go.Figure()
    for _ri, (_, _row) in enumerate(_rdr_top.iterrows()):
        _vals = [_row[a] for a in _radar_axes] + [_row[_radar_axes[0]]]  # close polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=_vals,
            theta=_radar_labels + [_radar_labels[0]],
            fill="toself",
            name=_row["rule"][:50],
            opacity=0.55,
            line_color=_radar_colors[_ri % len(_radar_colors)],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_radar, width='stretch')
    st.caption("Axes normalized to [0,1]; higher = better for all axes after direction adjustment.")

# --- Ranked Summary Table with Composite Score & Tier ---
st.markdown("#### Ranked Summary Table")
if _pv_multi.empty:
    st.info("Reward/Time/PDR data not all available; skipping ranked table.")
else:
    _rank = _pv_multi.copy()
    for _col, _asc in [("Reward", False), ("Time", True), ("PDR", True)]:
        _mn, _mx = _rank[_col].min(), _rank[_col].max()
        if _mx - _mn > 1e-12:
            _norm = (_rank[_col] - _mn) / (_mx - _mn)
            _rank[f"{_col}_n"] = _norm if _asc else (1 - _norm)
        else:
            _rank[f"{_col}_n"] = 0.5
    _rank["Composite"] = (_rank["Reward_n"] + _rank["Time_n"] + _rank["PDR_n"]) / 3
    _rank = _rank.sort_values("Composite", ascending=False).reset_index(drop=True)
    _rank["Rank"] = range(1, len(_rank) + 1)

    # Tier assignment: top 25% = A, next 25% = B, rest = C
    _n = len(_rank)
    _rank["Tier"] = "C"
    _rank.loc[_rank["Rank"] <= max(1, int(_n * 0.25)), "Tier"] = "A"
    _rank.loc[(_rank["Rank"] > max(1, int(_n * 0.25))) & (_rank["Rank"] <= max(1, int(_n * 0.50))), "Tier"] = "B"

    _disp_cols = ["Rank", "Tier", "label", "rule", "Reward", "Time", "PDR", "Composite"]
    st.dataframe(
        _rank[_disp_cols].style.format({"Reward": "{:.4f}", "Time": "{:.3f}", "PDR": "{:.4f}", "Composite": "{:.4f}"}),
        width='stretch', hide_index=True,
    )
    st.caption("Composite = mean of min-max normalized scores (Reward↑, Time↓, PDR↓). "
               "Tier: A = top 25%, B = 25–50%, C = bottom 50%.")

# ══════════════════════════════════════════════════════════════════════════════
# Cross-Scenario Meta-Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Cross-Scenario Meta-Analysis")
st.caption("Assess whether rule rankings generalize across locations/experiments.")

if df_raw.empty or "label" not in df_raw.columns:
    st.info("Multi-experiment data required for meta-analysis.")
else:
    _meta_labels = sorted(df_raw["label"].unique())
    if len(_meta_labels) < 2:
        st.info("At least 2 distinct labels (locations/experiments) are needed for meta-analysis.")
    else:
        _meta_metric = st.selectbox("Metric for meta-analysis",
                                     ["Reward", "Time", "PDR"], key="meta_metric")
        _meta_df = df_raw[df_raw["metric"] == _meta_metric].copy()

        if _meta_df.empty:
            st.warning(f"No {_meta_metric} data available.")
        else:
            # Per-label rule means
            _meta_pivot = _meta_df.groupby(["label", "rule"])["value"].mean().reset_index()
            _meta_wide = _meta_pivot.pivot_table(index="rule", columns="label", values="value")
            _meta_wide = _meta_wide.dropna()

            if _meta_wide.shape[0] < 2 or _meta_wide.shape[1] < 2:
                st.warning("Insufficient data for meta-analysis (need 2+ rules across 2+ locations).")
            else:
                st.markdown("### Kendall's W (Concordance of Rankings)")
                st.caption("Measures agreement in rule rankings across locations. W=1: perfect agreement, W=0: no agreement.")

                from scipy.stats import rankdata as _meta_rankdata
                # Rank rules within each location
                _meta_ranks = _meta_wide.copy()
                _is_asc = _meta_metric in ("Time", "PDR")
                for col in _meta_ranks.columns:
                    _meta_ranks[col] = _meta_rankdata(
                        _meta_ranks[col].values * (1 if _is_asc else -1)
                    )

                _m_k = _meta_ranks.shape[1]  # number of raters (locations)
                _m_n = _meta_ranks.shape[0]  # number of items (rules)
                _m_rank_sums = _meta_ranks.sum(axis=1)
                _m_S = ((_m_rank_sums - _m_rank_sums.mean()) ** 2).sum()
                _m_W = 12 * _m_S / (_m_k**2 * (_m_n**3 - _m_n)) if (_m_k**2 * (_m_n**3 - _m_n)) > 0 else 0

                # Chi-square test for significance
                _m_chi2 = _m_k * (_m_n - 1) * _m_W
                _m_df_chi = _m_n - 1
                from scipy.stats import chi2 as _chi2_dist
                _m_p = 1 - _chi2_dist.cdf(_m_chi2, _m_df_chi)

                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                col_w1.metric("Kendall's W", f"{_m_W:.4f}")
                col_w2.metric("Chi-square", f"{_m_chi2:.2f}")
                col_w3.metric("p-value", f"{_m_p:.3g}")
                col_w4.metric("Interpretation",
                              "Strong" if _m_W > 0.7 else "Moderate" if _m_W > 0.4 else "Weak")

                if _m_p < 0.05:
                    st.success(f"Significant concordance (p={_m_p:.3g}): Rule rankings are consistent across locations.")
                else:
                    st.warning(f"No significant concordance (p={_m_p:.3g}): Rule performance may be location-dependent.")

                # Forest Plot
                st.markdown("### Forest Plot (Rule Performance Across Locations)")
                _forest_n = st.slider("Show top N rules", 5, 64, 15, 1, key="forest_n")

                # Compute per-rule grand mean and CI across locations
                _forest_stats = []
                for _rule in _meta_wide.index:
                    _vals = _meta_wide.loc[_rule].values
                    _f_mean = np.mean(_vals)
                    _f_std = np.std(_vals, ddof=1) if len(_vals) > 1 else 0
                    _f_n = len(_vals)
                    _f_se = _f_std / np.sqrt(_f_n) if _f_n > 0 else 0
                    from scipy.stats import t as _t_dist
                    if _f_n > 1:
                        _f_ci = _t_dist.interval(0.95, df=_f_n-1, loc=_f_mean, scale=_f_se)
                    else:
                        _f_ci = (_f_mean, _f_mean)
                    _forest_stats.append({
                        "rule": _rule, "grand_mean": _f_mean,
                        "CI_low": _f_ci[0], "CI_high": _f_ci[1],
                        "std": _f_std, "n_locations": _f_n,
                    })
                _forest_df = pd.DataFrame(_forest_stats)
                _forest_df = _forest_df.sort_values("grand_mean",
                                                     ascending=_is_asc).head(_forest_n).reset_index(drop=True)

                fig_forest = go.Figure()
                fig_forest.add_trace(go.Scatter(
                    x=_forest_df["grand_mean"],
                    y=list(range(len(_forest_df))),
                    error_x=dict(
                        type="data", symmetric=False,
                        array=(_forest_df["CI_high"] - _forest_df["grand_mean"]).tolist(),
                        arrayminus=(_forest_df["grand_mean"] - _forest_df["CI_low"]).tolist(),
                    ),
                    mode="markers",
                    marker=dict(size=8, color="#1abc9c"),
                    text=_forest_df["rule"],
                    hovertemplate="%{text}<br>Mean: %{x:.4f}<extra></extra>",
                ))
                fig_forest.update_layout(
                    yaxis=dict(tickmode="array", tickvals=list(range(len(_forest_df))),
                               ticktext=[r[:35] for r in _forest_df["rule"]], autorange="reversed"),
                    xaxis_title=f"{_meta_metric} (grand mean across locations)",
                    height=max(400, len(_forest_df) * 25),
                    margin=dict(l=280, r=30, t=30, b=30),
                    title=f"Forest Plot: {_meta_metric} ({'lower=better' if _is_asc else 'higher=better'})",
                )
                st.plotly_chart(fig_forest, width='stretch')

                # Rule x Location interaction
                st.markdown("### Rule x Location Interaction Test")
                st.caption("Tests whether rule performance depends on location (significant interaction = location-dependent).")
                try:
                    import statsmodels.api as _sm_meta
                    import statsmodels.formula.api as _smf_meta
                    _inter_df = _meta_df.copy()
                    _inter_df["label"] = _inter_df["label"].astype(str)
                    _inter_df["rule"] = _inter_df["rule"].astype(str)
                    _inter_model = _smf_meta.ols("value ~ C(rule) + C(label) + C(rule):C(label)", data=_inter_df).fit()
                    _inter_anova = _sm_meta.stats.anova_lm(_inter_model, typ=2)

                    _inter_display = _inter_anova.copy()
                    _inter_display.index = _inter_display.index.str.replace("C(rule):C(label)", "Rule x Location", regex=False)
                    _inter_display.index = _inter_display.index.str.replace("C(rule)", "Rule", regex=False)
                    _inter_display.index = _inter_display.index.str.replace("C(label)", "Location", regex=False)
                    st.dataframe(_inter_display.style.format("{:.4f}"), width='stretch')

                    _inter_p = _inter_anova.loc["C(rule):C(label)", "PR(>F)"] if "C(rule):C(label)" in _inter_anova.index else 1.0
                    if _inter_p < 0.05:
                        st.warning(f"Significant interaction (p={_inter_p:.3g}): Rule recommendations are location-dependent. "
                                   "Caution when generalizing across different geographic areas.")
                    else:
                        st.success(f"No significant interaction (p={_inter_p:.3g}): Rule rankings are generalizable across locations.")
                except ImportError:
                    st.info("statsmodels required for interaction test. Install with: pip install statsmodels")
                except Exception as e_inter:
                    st.warning(f"Interaction test failed: {e_inter}")

                # Universally-good vs location-dependent rules
                st.markdown("### Rule Classification: Universal vs Location-Dependent")
                _class_df = _meta_wide.copy()
                _class_ranks = _meta_ranks.copy()
                _class_mean_rank = _class_ranks.mean(axis=1)
                _class_rank_std = _class_ranks.std(axis=1, ddof=1) if _meta_ranks.shape[1] > 1 else pd.Series(0, index=_class_ranks.index)
                _class_cv = _class_rank_std / _class_mean_rank.replace(0, np.nan)

                _class_summary = pd.DataFrame({
                    "rule": _class_df.index,
                    "mean_rank": _class_mean_rank.values,
                    "rank_std": _class_rank_std.values,
                    "rank_CV": _class_cv.values,
                    "grand_mean": _class_df.mean(axis=1).values,
                }).sort_values("mean_rank").reset_index(drop=True)

                _class_summary["classification"] = "Moderate"
                _good_thresh = _m_n * 0.25
                _class_summary.loc[
                    (_class_summary["mean_rank"] <= _good_thresh) & (_class_summary["rank_CV"] < 0.3),
                    "classification"
                ] = "Universally Good"
                _class_summary.loc[_class_summary["rank_CV"] > 0.5, "classification"] = "Location-Dependent"

                st.dataframe(
                    _class_summary.style.format({
                        "mean_rank": "{:.1f}", "rank_std": "{:.2f}",
                        "rank_CV": "{:.3f}", "grand_mean": "{:.4f}"
                    }),
                    width='stretch', hide_index=True,
                )
                _n_univ = len(_class_summary[_class_summary["classification"] == "Universally Good"])
                _n_dep = len(_class_summary[_class_summary["classification"] == "Location-Dependent"])
                st.caption(f"Universally Good: {_n_univ} rules (top 25% mean rank, CV < 0.3). "
                           f"Location-Dependent: {_n_dep} rules (rank CV > 0.5).")

                # Tornado Diagram (Rank Variation)
                st.markdown("### Tornado Diagram (Rank Variation)")
                _tornado_n = st.slider("Show top N rules", 5, 64, 20, 1, key="tornado_n")

                _stability_t = _class_summary.head(_tornado_n).copy()
                _stability_t = _stability_t.sort_values("mean_rank", ascending=False)

                fig_tornado = go.Figure()
                _t_range = _stability_t["mean_rank"].max() - _stability_t["mean_rank"].min()
                fig_tornado.add_trace(go.Bar(
                    y=[r[:35] for r in _stability_t["rule"]],
                    x=(_stability_t["rank_std"] * 2).tolist(),
                    orientation='h',
                    marker=dict(
                        color=_stability_t["rank_CV"].tolist(),
                        colorscale="RdYlGn_r", cmin=0, cmax=1,
                        colorbar=dict(title="CV"),
                    ),
                    text=[f"Rank {r:.1f}" for r in _stability_t["mean_rank"]],
                    textposition="outside",
                    hovertemplate="%{y}<br>Mean rank: %{text}<br>Rank spread: %{x:.1f}<extra></extra>",
                ))
                fig_tornado.update_layout(
                    xaxis_title="Rank Spread (2 x std)",
                    height=max(400, _tornado_n * 25),
                    margin=dict(l=300, r=80, t=30, b=30),
                    title=f"Rule Rank Variation Across Locations ({_meta_metric})",
                )
                st.plotly_chart(fig_tornado, width='stretch')

                # Factor-Level Sensitivity
                st.markdown("### Factor-Level Sensitivity")
                st.caption("Which factors have the most influence on performance across locations?")

                _factor_cols = ["Phase", "RedPolicy", "RedAction", "YellowAction"]
                _avail_f = [f for f in _factor_cols if f in _meta_df.columns and _meta_df[f].nunique() > 1]

                if _avail_f:
                    _f_effects = []
                    for _f in _avail_f:
                        _f_means = _meta_df.groupby(_f)["value"].mean()
                        _f_range = _f_means.max() - _f_means.min()
                        _f_effects.append({
                            "factor": _f, "range": _f_range,
                            "levels": _f_means.to_dict(), "n_levels": len(_f_means),
                        })
                    _f_df = pd.DataFrame(_f_effects).sort_values("range", ascending=False)

                    fig_fmain = go.Figure()
                    fig_fmain.add_trace(go.Bar(
                        y=_f_df["factor"], x=_f_df["range"],
                        orientation='h', marker=dict(color="#1abc9c"),
                        text=[f"{r:.4f}" for r in _f_df["range"]],
                        textposition="outside",
                    ))
                    fig_fmain.update_layout(
                        xaxis_title=f"Range of Level Means ({_meta_metric})",
                        height=250, margin=dict(l=150, r=80, t=30, b=30),
                        title="Factor Main Effect Sizes",
                    )
                    st.plotly_chart(fig_fmain, width='stretch')

                    for _fe in _f_effects:
                        with st.expander(f"Factor: {_fe['factor']} ({_fe['n_levels']} levels)"):
                            _ldf = pd.DataFrame([
                                {"level": k, f"{_meta_metric}_mean": v}
                                for k, v in _fe["levels"].items()
                            ]).sort_values(f"{_meta_metric}_mean", ascending=_is_asc)
                            st.dataframe(_ldf, width='stretch', hide_index=True)
                            st.info(f"Best level: **{_ldf.iloc[0]['level']}** "
                                    f"({_meta_metric} = {_ldf.iloc[0][f'{_meta_metric}_mean']:.4f})")
                else:
                    st.info("Factor columns not found in data.")

# ══════════════════════════════════════════════════════════════════════════════
# Scenario Config Diff
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Scenario Config Diff")
st.caption("Compare YAML configurations between two experiments/coordinates.")

if "rc_base_path" in st.session_state and st.session_state.rc_base_path:
    _diff_bp = st.session_state.rc_base_path
else:
    _diff_bp = DEFAULT_LOCAL_BASE_PATH

import yaml as _yaml_diff
import os as _os_diff

_diff_scenarios_dir = Path(_diff_bp) / "scenarios" if _diff_bp else None
if _diff_scenarios_dir and _diff_scenarios_dir.is_dir():
    _diff_exps = sorted([d.name for d in _diff_scenarios_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if len(_diff_exps) >= 1:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Config A**")
            _diff_exp_a = st.selectbox("Experiment A", _diff_exps, key="diff_exp_a")
            _diff_coords_a_dir = _diff_scenarios_dir / _diff_exp_a
            _diff_coords_a = sorted([d.name for d in _diff_coords_a_dir.iterdir()
                                      if d.is_dir() and not d.name.startswith(".")]) if _diff_coords_a_dir.is_dir() else []
            _diff_coord_a = st.selectbox("Coordinate A", _diff_coords_a, key="diff_coord_a") if _diff_coords_a else None

        with col_d2:
            st.markdown("**Config B**")
            _diff_exp_b = st.selectbox("Experiment B", _diff_exps, key="diff_exp_b")
            _diff_coords_b_dir = _diff_scenarios_dir / _diff_exp_b
            _diff_coords_b = sorted([d.name for d in _diff_coords_b_dir.iterdir()
                                      if d.is_dir() and not d.name.startswith(".")]) if _diff_coords_b_dir.is_dir() else []
            _diff_coord_b = st.selectbox("Coordinate B", _diff_coords_b, key="diff_coord_b") if _diff_coords_b else None

        if _diff_coord_a and _diff_coord_b and st.button("Compare Configs", key="btn_diff"):
            _yaml_a_path = _diff_scenarios_dir / _diff_exp_a / _diff_coord_a / f"config_{_diff_coord_a}.yaml"
            _yaml_b_path = _diff_scenarios_dir / _diff_exp_b / _diff_coord_b / f"config_{_diff_coord_b}.yaml"

            def _flatten_dict(d, prefix=""):
                items = {}
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.update(_flatten_dict(v, key))
                    elif isinstance(v, list):
                        items[key] = str(v)
                    else:
                        items[key] = v
                return items

            if not _yaml_a_path.is_file():
                st.error(f"Config A not found: {_yaml_a_path}")
            elif not _yaml_b_path.is_file():
                st.error(f"Config B not found: {_yaml_b_path}")
            else:
                with open(_yaml_a_path, "r", encoding="utf-8") as f:
                    _cfg_a = _yaml_diff.safe_load(f)
                with open(_yaml_b_path, "r", encoding="utf-8") as f:
                    _cfg_b = _yaml_diff.safe_load(f)

                _flat_a = _flatten_dict(_cfg_a) if _cfg_a else {}
                _flat_b = _flatten_dict(_cfg_b) if _cfg_b else {}
                _all_keys = sorted(set(list(_flat_a.keys()) + list(_flat_b.keys())))

                _diff_rows = []
                for _dk in _all_keys:
                    _va = _flat_a.get(_dk, "(missing)")
                    _vb = _flat_b.get(_dk, "(missing)")
                    _status = "Same" if str(_va) == str(_vb) else "Different"
                    _diff_rows.append({
                        "Parameter": _dk,
                        "Config A": str(_va),
                        "Config B": str(_vb),
                        "Status": _status,
                    })
                _diff_result = pd.DataFrame(_diff_rows)

                _show_all = st.checkbox("Show all parameters (including identical)", value=False, key="diff_show_all")
                if not _show_all:
                    _diff_result = _diff_result[_diff_result["Status"] == "Different"]

                if _diff_result.empty:
                    st.success("Configurations are identical.")
                else:
                    n_diff = len(_diff_result[_diff_result["Status"] == "Different"])
                    st.info(f"{n_diff} parameter(s) differ between the two configs.")
                    st.dataframe(_diff_result, width='stretch', hide_index=True)

                # CSV file comparison
                with st.expander("CSV Input File Comparison"):
                    _csv_a_dir = _diff_scenarios_dir / _diff_exp_a / _diff_coord_a
                    _csv_b_dir = _diff_scenarios_dir / _diff_exp_b / _diff_coord_b
                    _csv_files_a = sorted([f.name for f in _csv_a_dir.glob("*.csv")]) if _csv_a_dir.is_dir() else []
                    _csv_files_b = sorted([f.name for f in _csv_b_dir.glob("*.csv")]) if _csv_b_dir.is_dir() else []
                    _csv_all_files = sorted(set(_csv_files_a + _csv_files_b))

                    _csv_comp = []
                    for _cf in _csv_all_files:
                        _fa = _csv_a_dir / _cf
                        _fb = _csv_b_dir / _cf
                        _row = {"file": _cf}
                        if _fa.is_file():
                            try:
                                _dfa = pd.read_csv(_fa, encoding="utf-8")
                            except Exception:
                                try:
                                    _dfa = pd.read_csv(_fa, encoding="cp949")
                                except Exception:
                                    _dfa = pd.DataFrame()
                            _row["A_rows"] = len(_dfa)
                            _row["A_cols"] = len(_dfa.columns)
                        else:
                            _row["A_rows"] = "(missing)"
                            _row["A_cols"] = "(missing)"
                        if _fb.is_file():
                            try:
                                _dfb = pd.read_csv(_fb, encoding="utf-8")
                            except Exception:
                                try:
                                    _dfb = pd.read_csv(_fb, encoding="cp949")
                                except Exception:
                                    _dfb = pd.DataFrame()
                            _row["B_rows"] = len(_dfb)
                            _row["B_cols"] = len(_dfb.columns)
                        else:
                            _row["B_rows"] = "(missing)"
                            _row["B_cols"] = "(missing)"
                        _csv_comp.append(_row)

                    if _csv_comp:
                        st.dataframe(pd.DataFrame(_csv_comp), width='stretch', hide_index=True)
    else:
        st.info("No experiment folders found for config diff.")
else:
    st.info("Set base path above to enable config diff.")
