# MCI_Streamlit.py — Patched (upgrade-in-place, v4)
# Incremental patch on top of the existing dashboard structure, incorporating all requested additions/enhancements.
# Python 3.9 / Streamlit 1.34+ / folium / streamlit-folium / altair / pandas / openpyxl /
# (optional) statsmodels, scipy
# -------------------------------------------------------------------------------------------------
# Patch summary
# 1) Settings: Mini-map at sidebar bottom when a coordinate is selected (quick overview of current position)
# 2) Scenarios: Log selection restricted to experiment_logs/<coord>; Rule block parser restored
#    - (Unlabeled) blocks hidden from dropdown + cause comment shown
#    - Patient-centric summary table (rescue time/transport mode/hospital/arrival time/care complete/remarks) + full event table retained
# 3) Maps: Multi-select (first 3-digit index), Select All/Deselect All buttons, amb_info limited to patient count
#    - UAV route display toggle (single toggle controls both dispatch/transport)
#    - UAV Dispatch: Tertiary hospital (grade code=1) -> incident site (purple dashed) / UAV Transport: incident -> all hospitals (teal dashed)
#    - When dispatch and transport overlap at the same hospital, slight offset applied to transport line (reduce overlap)
#    - Legend includes AMB congestion colors / UAV dashed line sample + (when available) AMB/UAV speed from YAML
#    - Route summary (distance km, time min) inserted into popup, duration(ms) -> min correction
# 4) Analytics: Original table kept at top + "Sort by" selector (Reward desc, PDR asc (lower=better), Time asc (shorter=better)) for full table sorting
#    - Heatmap/bar chart removed -> replaced with ANOVA suite (Full Factorial, M1=Reward default)
#      - raw(results_{coord}.txt) parsing -> Phase, RedPolicy, RedAction, YellowAction x Sample
#      - If statsmodels available: OLS + Type-II ANOVA, residual normality (Shapiro), QQ scatter, residual histogram
# 5) Data Tables: File selector shows filename only (path hidden), fire station CSV excluded from edit list
# -------------------------------------------------------------------------------------------------
import os, re, json, shutil, subprocess, math, ast, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium
from openpyxl import load_workbook  # dependency ensure
import yaml

# Suppress Tukey HSD convergence speed warnings
import warnings
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=IntegrationWarning)


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

# (optional) statistics packages
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False
try:
    from scipy import stats as scistats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

KST = timezone(timedelta(hours=9))

# ------------------------------
# Session defaults
# ------------------------------
# Session defaults
# ------------------------------
CLOUD_BASE_PATH = _detect_cloud_base_path()
IS_CLOUD = bool(CLOUD_BASE_PATH)
DEFAULT_LOCAL_BASE_PATH = str(REPO_ROOT) if (REPO_ROOT / "scenarios").is_dir() else ""

if "base_path" not in st.session_state:
    st.session_state.base_path = CLOUD_BASE_PATH if IS_CLOUD else DEFAULT_LOCAL_BASE_PATH
else:
    # In cloud mode, always fixed (immediately revert even if user changes it)
    if IS_CLOUD and st.session_state.base_path != CLOUD_BASE_PATH:
        st.session_state.base_path = CLOUD_BASE_PATH

if "selected_exp" not in st.session_state:
    st.session_state.selected_exp = ""
if "selected_coord" not in st.session_state:
    st.session_state.selected_coord = ""
if "ps_running" not in st.session_state:
    st.session_state.ps_running = False
if "py_running" not in st.session_state:
    st.session_state.py_running = False

# ==== RAW results utils (results_{coord}.txt) ====
import os, re
import numpy as np
import pandas as pd

# Regex for detecting numbers in RAW lines
_RAW_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# Metric block order stored in RAW file (same order as code saves them)
RAW_METRIC_NAMES = ["Reward", "Time", "PDR", "Reward w.o.G", "PDR w.o.G"]

def _split_key_and_vals(line: str):
    """
    Split a line into 'key string' and 'list of floats'.
    - Comma between key and values is optional; split at the position of the first number.
    """
    m = re.search(_RAW_FLOAT, line)
    if not m:
        return line.strip(), []
    key = line[:m.start()].strip().rstrip(",")
    vals = [float(x) for x in re.findall(_RAW_FLOAT, line[m.start():])]
    return key, vals

def _split_factors(rule_label: str):
    """
    'ReSTART, YellowHalf, Red Both_UAVFirst, Yellow OnlyUAV'
      -> (Phase, RedPolicy, RedAction, YellowAction)
    """
    parts = [p.strip() for p in rule_label.split(",")]
    phase = parts[0] if len(parts) > 0 else ""
    red_policy = parts[1] if len(parts) > 1 else ""

    def pick_mode(p: str, color: str):
        if not p:
            return ""
        for key in ("OnlyUAV", "OnlyAMB", "Both_UAVFirst", "Both_AMBFirst"):
            if key in p:
                return key
        toks = p.replace(",", " ").split()
        try:
            cidx = toks.index(color)
            return "_".join(toks[cidx+1:]) if cidx < len(toks)-1 else ""
        except ValueError:
            return "_".join(toks[1:]) if len(toks) > 1 else ""

    red_action = pick_mode(parts[2] if len(parts) > 2 else "", "Red")
    yellow_action = pick_mode(parts[3] if len(parts) > 3 else "", "Yellow")
    return phase, red_policy, red_action, yellow_action

def parse_raw_all_metrics(raw_path: str) -> dict:
    """
    Read the entire results_{coord}.txt and return wide tables per metric.
    Returns: {metric_name: DataFrame(64 rows x [ScenarioIdx, Phase, RedPolicy, RedAction, YellowAction, metric run1..runR])}
    - Block boundaries: delimited by 'one cycle of rule keys'
    - Number of run columns auto-detected from file
    """
    out = {}
    if not (raw_path and os.path.exists(raw_path)):
        return out

    # 1) Parse all lines into (key, values)
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            key, vals = _split_key_and_vals(raw)
            if not vals:
                continue
            rows.append((key, vals))
    if not rows:
        return out

    # 2) Detect cycle length (= number of rules): position where first key reappears
    first_key = rows[0][0]
    L = None
    for i in range(1, len(rows)):
        if rows[i][0] == first_key:
            L = i
            break
    if L is None:
        L = len(rows)

    # 3) Split into blocks (one per metric)
    n_blocks = int(np.ceil(len(rows) / L))
    for b in range(n_blocks):
        block = rows[b*L : (b+1)*L]
        if not block:
            continue
        metric_name = RAW_METRIC_NAMES[b] if b < len(RAW_METRIC_NAMES) else f"Metric {b+1}"

        # number of runs
        R = len(block[0][1])

        # create records
        recs = []
        for idx, (label, vals) in enumerate(block):
            phase, red_policy, red_action, yellow_action = _split_factors(label)
            rec = {
                "ScenarioIdx": idx,
                "Phase": phase,
                "RedPolicy": red_policy,
                "RedAction": red_action,
                "YellowAction": yellow_action,
            }
            for r in range(R):
                rec[f"{metric_name} run{r+1}"] = vals[r] if r < len(vals) else np.nan
            recs.append(rec)

        out[metric_name] = pd.DataFrame.from_records(recs)

    return out

# ==== RAW → long parser for ANOVA ====
import os, re
import numpy as np
import pandas as pd

_RAW_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# Block (metric) name mapping within RAW file (matches file order)
# Aligned with underscore-style keys used in ANOVA UI
RAW_BLOCK_NAMES = ["Reward", "Time", "PDR", "Reward_woG", "PDR_woG"]

def _split_key_vals(line: str):
    m = re.search(_RAW_FLOAT, line)
    if not m:  # no numbers found
        return line.strip(), []
    key = line[:m.start()].strip().rstrip(",")
    vals = [float(x) for x in re.findall(_RAW_FLOAT, line[m.start():])]
    return key, vals

def _split_factors(rule_label: str):
    parts = [p.strip() for p in rule_label.split(",")]
    phase = parts[0] if len(parts) > 0 else ""
    red_policy = parts[1] if len(parts) > 1 else ""
    def pick_mode(p: str, color: str):
        if not p: return ""
        for k in ("OnlyUAV","OnlyAMB","Both_UAVFirst","Both_AMBFirst"):
            if k in p: return k
        toks = p.replace(",", " ").split()
        try:
            cidx = toks.index(color)
            return "_".join(toks[cidx+1:]) if cidx < len(toks)-1 else ""
        except ValueError:
            return "_".join(toks[1:]) if len(toks) > 1 else ""
    red_action = pick_mode(parts[2] if len(parts)>2 else "", "Red")
    yellow_action = pick_mode(parts[3] if len(parts)>3 else "", "Yellow")
    return phase, red_policy, red_action, yellow_action

@st.cache_data(ttl=600)
def parse_raw_results(raw_path: str) -> pd.DataFrame:
    """
    results_(lat,lon).txt → long DF
    columns: ['rule','Phase','RedPolicy','RedAction','YellowAction','run','metric','value']
    """
    if not (raw_path and os.path.exists(raw_path)):
        return pd.DataFrame(columns=["rule","Phase","RedPolicy","RedAction","YellowAction","run","metric","value"])

    # Parse all lines into (key, values)
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

    # Detect cycle length (= number of rules): position where first key reappears
    first_key = rows[0][0]
    L = None
    for i in range(1, len(rows)):
        if rows[i][0] == first_key:
            L = i
            break
    if L is None:
        L = len(rows)

    n_blocks = int(np.ceil(len(rows) / L))
    out_recs = []
    for b in range(n_blocks):
        block = rows[b*L:(b+1)*L]
        if not block:
            continue
        metric_name = RAW_BLOCK_NAMES[b] if b < len(RAW_BLOCK_NAMES) else f"Metric_{b+1}"
        # number of runs (auto-determined from value count)
        R = len(block[0][1])
        for (label, vals) in block:
            Phase, RedPolicy, RedAction, YellowAction = _split_factors(label)
            rule = f"{Phase}, {RedPolicy}, Red {RedAction}, Yellow {YellowAction}".strip().replace("  "," ")
            for r_idx, v in enumerate(vals, start=1):
                out_recs.append({
                    "rule": rule,
                    "Phase": Phase,
                    "RedPolicy": RedPolicy,
                    "RedAction": RedAction,
                    "YellowAction": YellowAction,
                    "run": r_idx,           # 1..R
                    "metric": metric_name,  # 'Reward' / 'Time' / 'PDR' / 'Reward_woG' / 'PDR_woG'
                    "value": v
                })
    df = pd.DataFrame.from_records(out_recs)
    # Category sorting (optional)
    if not df.empty:
        # Fix rule order to that of the first block
        first_block_rules = [rows[i][0] for i in range(L)]
        # The above first_block_rules are raw labels. They may differ from the 'rule' strings we created, so skip.
        # If needed, a sort key can be generated from Phase/RedPolicy/RedAction/YellowAction combinations.
        df["run"] = df["run"].astype(int)
    return df

def _means_series(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """
    Always return group means as a Series (safe regardless of pandas version/context).
    Returns: index=group_col, values=mean(value_col)
    """
    tmp = df.groupby(group_col, as_index=False).agg({value_col: "mean"})
    s = tmp.set_index(group_col)[value_col]
    # Force to numeric if not already
    return pd.to_numeric(s, errors="coerce")




# ------------------------------
# Helpers: paths & IO
# ------------------------------

def norm(p: Union[str, Path]) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(p))))

def ts() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def exists_dir(p: Union[str, Path]) -> bool:
    return os.path.isdir(str(p))

def exists_file(p: Union[str, Path]) -> bool:
    return os.path.isfile(str(p))

def base_ok(base_path: str) -> bool:
    s = Path(base_path)
    return s.is_dir() and (s / "scenarios").is_dir()

# --- NEW: Combined experiment list from scenarios + results (scenarios preferred) ---
@st.cache_data(ttl=60)
def list_experiments_any(base_path: str) -> List[str]:
    res_root = Path(base_path) / "results"
    scn_root = Path(base_path) / "scenarios"
    names = set()
    if scn_root.is_dir():
        names |= {p.name for p in scn_root.iterdir() if p.is_dir()}
    if res_root.is_dir():
        names |= {p.name for p in res_root.iterdir() if p.is_dir()}
    items = list(names)

    # Sort by latest modification time (reflecting the more recent of scenarios and results)
    def mtime(name: str) -> float:
        t1 = (scn_root / name).stat().st_mtime if (scn_root / name).exists() else 0.0
        t2 = (res_root / name).stat().st_mtime if (res_root / name).exists() else 0.0
        return max(t1, t2)

    items.sort(key=mtime, reverse=True)
    return items

# --- NEW: Coordinate list for a specific experiment (based on scenarios) ---
@st.cache_data(ttl=60)
def list_coords_from_scenarios(base_path: str, exp_id: str) -> List[str]:
    root = Path(base_path) / "scenarios" / exp_id
    if not root.is_dir():
        return []
    items = [p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\(.*\)$", p.name)]
    items.sort()
    return items




# results/exp_* layout

def list_experiments(base_path: str) -> List[str]:
    root = Path(base_path) / "results"
    if not root.is_dir():
        return []
    items = [p.name for p in root.iterdir() if p.is_dir()]
    items.sort(key=lambda name: (root / name).stat().st_mtime, reverse=True)
    return items


def list_coords(base_path: str, exp_id: str) -> List[str]:
    root = Path(base_path) / "results" / exp_id
    if not root.is_dir():
        return []
    items = [p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\(.*\)$", p.name)]
    items.sort()
    return items


def find_yaml_in_coord(base_path: str, exp_id: str, coord: str) -> Optional[str]:
    folder = Path(base_path) / "scenarios" / exp_id / coord
    if not folder.is_dir():
        return None
    yamls = list(folder.glob("*.yaml"))
    return str(yamls[0]) if yamls else None


def coord_to_tuple(coord_name: str) -> Optional[Tuple[float,float]]:
    m = re.match(r"^\(([-\d\.]+),\s*([-\d\.]+)\)$", coord_name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))  # (lat, lon)


def coord_center(coord_name: str) -> Tuple[float,float]:
    t = coord_to_tuple(coord_name)
    if not t:
        return (37.5665, 126.9780)  # Seoul fallback
    return t


def get_routes_dirs(base_path: str, exp_id: str, coord: str) -> Dict[str, Path]:
    r = Path(base_path) / "scenarios" / exp_id / coord / "routes"
    return {
        "center2site": r / "center2site",
        "hos2site": r / "hos2site",
    }


@st.cache_data(ttl=300)
def load_json_files(folder: Path, limit: Optional[int]=None) -> List[dict]:
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    out = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            obj["_file"] = str(f)
            out.append(obj)
        except Exception as e:
            print(f"[WARN] JSON load failed: {f} ({e})")
    return out


def read_csv_smart(path: str) -> pd.DataFrame:
    if os.path.basename(path) == "fire_stations.csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8", errors="ignore")


def write_csv_smart(df: pd.DataFrame, path: str):
    b = f"{os.path.splitext(path)[0]}_backup_{ts()}.csv"
    shutil.copy2(path, b) if exists_file(path) else None
    if os.path.basename(path) == "fire_stations.csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")

# Hospital Excel: priority 1) base_path/hospital_master_data.xlsx  2) base_path/scenarios/hospital_master_data.xlsx

@st.cache_data(ttl=600)
def read_excel_hospital(base_path: str) -> Optional[pd.DataFrame]:
    cands = [
        Path(base_path) / "hospital_master_data.xlsx",
        Path(base_path) / "scenarios" / "hospital_master_data.xlsx",
    ]
    for excel_path in cands:
        if excel_path.is_file():
            try:
                df = pd.read_excel(excel_path, engine="openpyxl")
                return df
            except Exception as e:
                st.warning(f"Excel load failed: {excel_path} ({e})")
    return None

@st.cache_data(ttl=300)
def read_experiment_summary_csv(base_path: str, exp_id: str) -> Optional[pd.DataFrame]:
    folder = Path(base_path) / "scenarios" / exp_id
    if not folder.is_dir():
        return None
    cand1 = folder / "summary.csv"
    if cand1.is_file():
        for enc in ("utf-8-sig","cp949","utf-8"):
            try:
                return pd.read_csv(cand1, encoding=enc)
            except Exception:
                pass
    picks = sorted(folder.glob("*_summary.csv"))
    for p in picks:
        for enc in ("utf-8-sig","cp949","utf-8"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
    return None

def summarize_experiment(df: pd.DataFrame) -> Tuple[Dict[str,str], Optional[str]]:
    info, addr = {}, None
    if df is None or df.empty: return info, addr
    # Address candidates
    addr_cols = [c for c in df.columns if any(k in str(c) for k in ["address","site_addr","incident_site"])]
    if addr_cols:
        addr = str(df.iloc[0][addr_cols[0]])
    # Parameter candidates
    keys = ["seed","patients","amb","uav","hospital","policy","rule","speed","radius","duration","start","api","grid"]
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            v = df.iloc[0][c]
            if pd.notna(v): info[str(c)] = str(v)
    if len(df.columns)==2 and df.columns[0]!=df.columns[1]:
        for _,r in df.iterrows():
            k,v = str(r.iloc[0]), str(r.iloc[1])
            if any(t in k.lower() for t in keys) and v:
                info[k]=v
            if addr is None and ("address" in k.lower()):
                addr = v
    return info, addr

def summarize_experiment_extended(df: Optional[pd.DataFrame]) -> Tuple[Dict[str,str], Dict[str,str]]:
    site_info = {"Coordinate":"","Address":"","Road Address":""}
    sim_info: Dict[str,str] = {}
    if df is None or df.empty: return site_info, sim_info

    # Use the latest round of data (last row)
    latest_row_idx = -1

    addr_cols = [c for c in df.columns if ("address" in str(c).lower() and "road" not in str(c).lower())]  # match address columns (exclude road address)
    road_cols = [c for c in df.columns if "road_address" in str(c).lower()]  # match road address columns
    if addr_cols:
        v = df.iloc[latest_row_idx][addr_cols[0]]; site_info["Address"] = "" if pd.isna(v) else str(v)
    if road_cols:
        v = df.iloc[latest_row_idx][road_cols[0]]; site_info["Road Address"] = "" if pd.isna(v) else str(v)
    start_idx = None
    for i,c in enumerate(df.columns):
        if "scenario_gen_start" in str(c).lower():  # scenario_gen_start column marker
            start_idx = i; break
    if start_idx is not None:
        row = df.iloc[latest_row_idx, start_idx:]
        for k,v in row.items():
            if pd.notna(v): sim_info[str(k)] = str(v)
    return site_info, sim_info


def list_coord_csvs(base_path: str, exp_id: str, coord: str) -> List[str]:
    folder = Path(base_path) / "scenarios" / exp_id / coord
    return [str(p) for p in folder.glob("*.csv")]


def results_stat_path(base_path: str, exp_id: str, coord: str) -> Optional[str]:
    s = Path(base_path) / "results" / exp_id / coord / f"results_{coord}_stat.txt"
    return str(s) if s.is_file() else None


def results_raw_path(base_path: str, exp_id: str, coord: str) -> Optional[str]:
    s = Path(base_path) / "results" / exp_id / coord / f"results_{coord}.txt"
    return str(s) if s.is_file() else None


@st.cache_data(ttl=300)
def get_patient_count(base_path: str, exp_id: str, coord: str) -> Optional[int]:
    folder = Path(base_path) / "scenarios" / exp_id / coord
    for cand in ["amb_info.csv", "amb_info_road.csv", "patient_info.csv"]:
        fp = folder / cand
        if fp.is_file():
            try:
                df = read_csv_smart(str(fp))
                return len(df.index)
            except Exception:
                pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_scenario_csvs(bp: str, exp: str, coord: str):
    """Load & rename scenario CSVs (cached at top level)."""
    base = Path(bp) / "scenarios" / exp / coord
    _KR_HOSP = {"type_code":"Grade Code","institution_name":"Hospital Name","num_or":"ORs","num_beds":"Beds","helipad":"Helipad"}
    _KR_AMB  = {"fire_station_name":"Fire Station","num_vehicles_owned":"Fleet Size"}

    def _read(p, enc=None):
        if not p.is_file(): return pd.DataFrame()
        return pd.read_csv(p, encoding=enc) if enc else pd.read_csv(p)

    h = _read(base / "hospital_info_road.csv").rename(columns=_KR_HOSP)
    c = _read(Path(bp) / "scenarios" / "fire_stations.csv", "utf-8-sig")
    a = _read(base / "amb_info_road.csv").rename(columns=_KR_AMB)
    dr = _read(base / "distance_Hos2Site_road.csv")
    he = _read(base / "hospital_info_euc.csv").rename(columns=_KR_HOSP)
    de = _read(base / "distance_Hos2Site_euc.csv")
    return h, c, a, dr, he, de


# ------------------------------
# Logs (execution log discovery + parsing)
# ------------------------------
PHASES = ["START", "ReSTART"]
RED_POLICY = ["RedOnly", "YellowNearest"]
ACTIONS = ["OnlyUAV","Both_UAVFirst","Both_AMBFirst","OnlyAMB"]

RULE_HEADER_RE = re.compile(r"^(START|ReSTART)\s*,\s*(.+?)\s*$")
TUPLE_LINE_RE = re.compile(r"^\(\s*([^,]+)\s*,\s*(\d+)\s*,\s*'([A-Za-z_]+)'\s*,\s*\(([^)]*)\)\s*\)\s*$")
ACTION_RE = re.compile(r"^Action:\s*\[([^\]]+)\]")

# Iteration line parsing (e.g., "Iter : 3" or "Iteration: 3")
ITER_RE = re.compile(r'^\s*(?:Iter(?:ation)?\s*[:=]\s*)(\d+)\b', re.I)
NP_SCALAR_RE = re.compile(r"^np\.(?:float|int)\d+\((.+)\)$")
NP_WRAP_RE = re.compile(r"np\.(?:float|int)\d+\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)")


EV_ARG_PARSERS = {
    "p_rescue": lambda args: {"p": int(args[0])} if len(args)>=1 else {},
    "amb_arrival_site": lambda args: {"a": int(args[0])} if len(args)>=1 else {},
    "uav_arrival_site": lambda args: {"u": int(args[0])} if len(args)>=1 else {},
    "amb_arrival_hospital": lambda args: {"p": int(args[0]), "a": int(args[1]), "h": int(args[2])} if len(args)>=3 else {},
    "uav_arrival_hospital": lambda args: {"p": int(args[0]), "u": int(args[1]), "h": int(args[2])} if len(args)>=3 else {},
    "p_care_ready": lambda args: {"p": int(args[0]), "h": int(args[1])} if len(args)>=2 else {},
    "p_def_care": lambda args: {"p": int(args[0]), "h": int(args[1])} if len(args)>=2 else {},
}

ACTION_TOOLTIP_MD = (
    "**Action Index Reference**\n"
    "- `action[0] = p_class` → Patient severity (0=Red, 1=Yellow, 2=Green)\n"
    "- `action[1] = destination` → 0=On-site wait, 1…N → Hospital index+1\n"
    "- `action[2] = mode` → 0=AMB(Ambulance), 1=UAV"
)

def _strip_np_scalar(token: str) -> str:
    s = token.strip()
    m = NP_SCALAR_RE.match(s)
    return m.group(1).strip() if m else s

def _coerce_float(token: str) -> Optional[float]:
    if token is None:
        return None
    s = _strip_np_scalar(token)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _coerce_int(token: str) -> Optional[int]:
    if token is None:
        return None
    s = _strip_np_scalar(token)
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return int(v) if v.is_integer() else None

def _parse_event_tuple_line(line: str) -> Optional[Tuple[float, int, str, List[Union[int, float]]]]:
    if not line or line[0] != "(" or line[-1] != ")":
        return None
    clean = NP_WRAP_RE.sub(r"\1", line)
    try:
        t, eid, ev, args = ast.literal_eval(clean)
    except Exception:
        return None
    if not isinstance(ev, str):
        return None
    try:
        t_val = float(t)
        eid_val = int(eid)
    except Exception:
        return None
    if isinstance(args, (list, tuple)):
        args_list = list(args)
    else:
        args_list = [args] if args is not None else []
    parsed_args: List[Union[int, float]] = []
    for a in args_list:
        ival = _coerce_int(str(a))
        if ival is None:
            fval = _coerce_float(str(a))
            if fval is None:
                continue
            parsed_args.append(fval)
        else:
            parsed_args.append(ival)
    return t_val, eid_val, ev, parsed_args


def _read_text_any(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()

# Rule block parser (v3 reintroduced)
def parse_log_blocks(log_text: str) -> List[Dict]:
    blocks = []
    cur_iter = None                         # <- [added] store current Iter
    cur = {"rule": "(Unlabeled)", "iter": cur_iter, "events": [], "actions": []}  # <- [changed]

    for raw in log_text.splitlines():
        s = raw.strip()
        if not s:
            continue

        # [added] Check Iter line first
        miter = ITER_RE.match(s)
        if miter:
            cur_iter = int(miter.group(1))
            # Inject iter into in-progress block (update if missing/None)
            if "iter" not in cur or cur.get("iter") is None:
                cur["iter"] = cur_iter
            continue

        mhead = RULE_HEADER_RE.match(s)
        if mhead:
            if cur["events"] or cur["actions"] or cur["rule"] != "(Unlabeled)":
                blocks.append(cur)
            # [changed] Include current iter in new block too
            cur = {"rule": s, "iter": cur_iter, "events": [], "actions": []}
            continue

        ma = ACTION_RE.match(s)
        if ma:
            try:
                parts = [x.strip() for x in ma.group(1).split(',')]
                vec = []
                for part in parts:
                    if not part:
                        continue
                    ival = _coerce_int(part)
                    if ival is None:
                        fval = _coerce_float(part)
                        if fval is None:
                            continue
                        vec.append(fval)
                    else:
                        vec.append(ival)
                if vec:
                    cur["actions"].append(vec)
            except Exception:
                pass
            continue

        parsed = _parse_event_tuple_line(s)
        if parsed:
            t, eid, ev, args = parsed
            rec = {"t": t, "eid": eid, "ev": ev, "p": None, "a": None, "u": None, "h": None}
            if ev in EV_ARG_PARSERS:
                try:
                    rec.update(EV_ARG_PARSERS[ev](args))
                except Exception:
                    pass
            cur["events"].append(rec)
            continue

        mt = TUPLE_LINE_RE.match(s)
        if mt:
            t = _coerce_float(mt.group(1))
            if t is None:
                continue
            eid = int(mt.group(2))
            ev = mt.group(3)
            args_raw = [x.strip() for x in mt.group(4).split(',') if x.strip() != '']
            args = []
            for a in args_raw:
                ival = _coerce_int(a)
                if ival is None:
                    fval = _coerce_float(a)
                    if fval is None:
                        continue
                    args.append(fval)
                else:
                    args.append(ival)
            rec = {"t": t, "eid": eid, "ev": ev, "p": None, "a": None, "u": None, "h": None}
            if ev in EV_ARG_PARSERS:
                try:
                    rec.update(EV_ARG_PARSERS[ev](args))
                except Exception:
                    pass
            cur["events"].append(rec)
            continue

    if cur["events"] or cur["actions"] or cur["rule"] != "(Unlabeled)":
        blocks.append(cur)
    return blocks


# Patient-centric summary

def build_patient_summary(events: List[Dict]) -> pd.DataFrame:
    byp: Dict[int, Dict] = {}
    arrival_hist: Dict[int, List[Tuple[str,int,float]]] = {}
    for e in sorted(events, key=lambda r: r["t"]):
        ev = e.get("ev"); p = e.get("p"); h = e.get("h")
        if ev == "p_rescue" and p is not None:
            byp.setdefault(p, {}).setdefault("rescue_t", e["t"])
        elif ev in ("amb_arrival_hospital","uav_arrival_hospital") and p is not None:
            mode = "AMB" if ev.startswith("amb_") else "UAV"
            if p not in byp:
                byp[p] = {}
            if "arrive_t" not in byp[p]:
                byp[p]["arrive_t"] = e["t"]; byp[p]["mode"] = mode; byp[p]["hospital"] = h
            arrival_hist.setdefault(p, []).append((mode, h, e["t"]))
        elif ev == "p_care_ready" and p is not None:
            byp.setdefault(p, {}).setdefault("care_ready_t", e["t"])
        elif ev == "p_def_care" and p is not None:
            byp.setdefault(p, {}).setdefault("def_care_t", e["t"])
    rows = []
    for p, info in sorted(byp.items(), key=lambda kv: kv[0]):
        hist = arrival_hist.get(p, [])
        remark = "Normal"
        if len(hist) >= 2:
            hosp_set = {h for (_,h,_) in hist}; mode_set = {m for (m,_,_) in hist}
            if len(hosp_set) > 1 or len(mode_set) > 1:
                remark = "divert"
        rows.append({
            "PatientID": p,
            "Rescue Time": info.get("rescue_t"),
            "Transport Mode": info.get("mode"),
            "Dest. Hospital": info.get("hospital"),
            "Hospital Arrival": info.get("arrive_t"),
            "Care Ready": info.get("care_ready_t"),
            "Care Complete": info.get("def_care_t"),
            "Remarks": remark,
        })
    return pd.DataFrame(rows)

# Restrict to files within experiment_logs only

def _extract_ts(exp_id: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", exp_id or "")
    return m.group(1) if m else ""


def experiment_log_candidates(base_path: str, exp_id: str, coord: str) -> List[str]:
    """Collect only <coord>_*.log|.txt from experiment_logs folder (exp_id timestamp filter removed)."""
    logs_folder = Path(base_path) / "experiment_logs"
    if not logs_folder.is_dir():
        return []
    cprefix = f"{coord}_"
    files = [p for p in logs_folder.iterdir()
             if p.is_file() and p.name.startswith(cprefix)
             and p.suffix.lower() in (".log", ".txt")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


# ------------------------------
# Map helpers (congestion/legend/UAV/summary) - based on Kakao API
# ------------------------------
# Kakao traffic_state: 0=No info, 1=Congested, 2=Slow, 3=Moderate, 4=Clear, 6=Accident
CONG_COLORS = {0:"#888888", 1:"#FF0000", 2:"#FF6347", 3:"#FFD700", 4:"#7CFC00", 6:"#000000"}
CONG_LABELS = {0:"Unknown", 1:"Congested", 2:"Slow", 3:"Moderate", 4:"Clear", 6:"Accident"}
UAV_OUT_COLOR = "#8A2BE2"   # Dispatch (Hospital -> Incident)
UAV_BACK_COLOR = "#00CED1"  # Transport (Incident -> Hospital)

ROADTYPE_TO_KIND = {1:"expressway", 2:"urban_express", 3:"national_road"}
SPEED_THRESHOLDS = {
    "local_road":     {"clear":30, "moderate_low":15, "moderate_high":30, "congested":15},
    "national_road":  {"clear":40, "moderate_low":20, "moderate_high":40, "congested":20},
    "urban_express":  {"clear":60, "moderate_low":30, "moderate_high":60, "congested":30},
    "expressway":     {"clear":70, "moderate_low":40, "moderate_high":70, "congested":40},
}


def _classify_cong_by_speed(road_kind: str, speed_kmh: Optional[float]) -> Optional[int]:
    if speed_kmh is None:
        return None
    kind = road_kind if road_kind in SPEED_THRESHOLDS else "local_road"
    th = SPEED_THRESHOLDS[kind]
    if speed_kmh >= th["clear"]:  # clear
        return 1
    if th["moderate_low"] <= speed_kmh < th["moderate_high"]:  # moderate
        return 2
    if speed_kmh < th["congested"]:  # congested
        return 3
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    from math import radians, sin, cos, sqrt, atan2
    phi1,phi2=radians(lat1),radians(lat2)
    dphi=radians(lat2-lat1); dl=radians(lon2-lon1)
    a=sin(dphi/2)**2+cos(phi1)*cos(phi2)*sin(dl/2)**2
    return 2*R*atan2(math.sqrt(a), math.sqrt(1-a))


def _extract_summary_meta(obj: dict) -> Tuple[Optional[float], Optional[float], list]:
    meta = obj.get("meta", {})
    api_provider = meta.get("api_provider", "naver")

    if api_provider == "kakao":
        # Kakao API: Use meta fields directly (already extracted)
        dist_km = meta.get("distance_km")
        dur_min = meta.get("duration_min")
        # Kakao doesn't have guide in the same format as Naver
        return dist_km, dur_min, []
    else:
        # Naver API: Extract from payload
        payload = (obj.get("payload") or {}).get("naver_response", {})
        route_list = payload.get("route", {}).get("trafast", [])
        if not route_list:
            return None, None, []
        r0 = route_list[0]
        summ = r0.get("summary", {})
        dist_km = (float(summ.get("distance", 0)) / 1000.0) if summ else None
        dur_min = (float(summ.get("duration", 0)) / 60000.0) if summ else None
        guide = r0.get("guide", []) or []
        return dist_km, dur_min, guide


def _guide_html(guide: list) -> str:
    if not guide:
        return ""
    rows = ["<details><summary>🧭 Route Guide (click)</summary><ol style='padding-left:16px;'>"]
    for g in guide[:200]:
        # Handle both Naver "instructions" and Kakao "guidance" fields
        inst = str(g.get("instructions") or g.get("guidance", "")).replace("<", "&lt;").replace(">", "&gt;")
        gd = float(g.get("distance", 0))/1000.0 if g.get("distance") is not None else None

        # Handle different duration formats
        # Naver API: duration in milliseconds
        # Kakao API: duration in seconds
        dur_val = g.get("duration", 0)
        if dur_val and dur_val > 1000:
            # Likely Naver (milliseconds)
            gm = float(dur_val) / 60000.0
        else:
            # Likely Kakao (seconds) or zero
            gm = float(dur_val or 0) / 60.0

        tail = []
        if gd is not None and gd>0: tail.append(f"{gd:.2f} km")
        if gm is not None and gm>0: tail.append(f"{gm:.1f} min")
        rows.append(f"<li>{inst} {' · '.join(tail) if tail else ''}</li>")
    rows.append("</ol></details>")
    return "".join(rows)


def add_site_marker(m: folium.Map, site_latlon: Tuple[float,float]):
    
    folium.CircelMarker(
        location=[site_latlon[0], site_latlon[1]],
        icon=folium.Icon(color="purple", icon="map-pin", prefix="fa"),
        radius =  10,
        tooltip="Incident Site",
        popup=f"Incident Site<br>lat,lon={site_latlon[0]:.6f},{site_latlon[1]:.6f}"
    ).add_to(m)


def add_center_marker(m: folium.Map, name: str, latlon: Tuple[float,float], extra_lines: List[str]):
    body = [f"<b>{name}</b>"] + [x for x in extra_lines if x]
    folium.Marker(
        location=[latlon[0], latlon[1]],
        icon=folium.Icon(color="blue", icon="ambulance", prefix="fa"),
        tooltip=name,
        popup="<br>".join(body)
    ).add_to(m)


# Cleaned up add_hospital_marker(...) as follows
def add_hospital_marker(
    m: folium.Map,
    name: str,
    grade_code: Union[int, str],
    latlon: Tuple[float, float],
    op_rooms: Optional[Union[int, float]] = None,  # number of operating rooms
    beds: Optional[Union[int, float]] = None,      # number of beds
    extra_lines: Optional[List[str]] = None,
):
    try:
        g = int(float(str(grade_code)))
    except:
        g = -1
    color = "red" if g == 1 else ("orange" if g == 11 else "green")

    body = [f"<b>{name}</b>"]                    # title
    body.append(f"lat,lon={latlon[0]:.6f},{latlon[1]:.6f}")  # coordinates
    if op_rooms is not None and not (isinstance(op_rooms, float) and math.isnan(op_rooms)):
        body.append(f"OR={int(op_rooms)}")              # operating rooms
    if beds is not None and not (isinstance(beds, float) and math.isnan(beds)):
        body.append(f"Beds={int(beds)}")                    # beds
    body.append(f"GradeCode={grade_code}")                    # grade code
    if extra_lines:                                           # hospital grade, distance, duration, etc.
        body += [x for x in extra_lines if x]

    folium.Marker(
        location=[latlon[0], latlon[1]],
        icon=folium.Icon(color=color, icon="plus", prefix="fa"),
        tooltip=name,
        popup="<br>".join(body)
    ).add_to(m)



def draw_route_from_json(m: folium.Map, route_obj: dict, highlight: bool=False):
    meta = route_obj.get("meta", {})
    api_provider = meta.get("api_provider", "naver")

    if api_provider == "kakao":
        # Kakao API: Parse kakao_response structure
        payload = (route_obj.get("payload") or {}).get("kakao_response", {})
        routes = payload.get("routes", [])
        if not routes:
            return

        route = routes[0]
        sections = route.get("sections", [])
        if not sections:
            return

        # Kakao API structure: sections[0].roads[]
        roads = sections[0].get("roads", [])

        for road in roads:
            traffic_state = road.get("traffic_state", 0)
            traffic_speed = road.get("traffic_speed")
            road_name = road.get("name", "")
            vertexes = road.get("vertexes", [])

            # Convert vertexes from [lon, lat, lon, lat, ...] to [[lat, lon], ...]
            latlngs = []
            for i in range(0, len(vertexes), 2):
                if i + 1 < len(vertexes):
                    latlngs.append([vertexes[i+1], vertexes[i]])  # [lat, lon]

            if not latlngs:
                continue

            # Map Kakao traffic_state directly
            # Kakao: 0=No info, 1=Congested, 2=Slow, 3=Moderate, 4=Clear, 6=Accident
            cong = traffic_state

            # Build tooltip with road info
            tooltip_parts = []
            if road_name:
                tooltip_parts.append(road_name)
            tooltip_parts.append(CONG_LABELS.get(cong, "Unknown"))
            if traffic_speed is not None:
                tooltip_parts.append(f"{traffic_speed:.0f}km/h")
            tooltip_text = " · ".join(tooltip_parts)

            folium.PolyLine(
                locations=latlngs,
                color=CONG_COLORS.get(cong, "#888888"),
                weight=8 if highlight else 5,
                opacity=0.9 if highlight else 0.7,
                tooltip=tooltip_text
            ).add_to(m)

    elif api_provider == "osrm":
        # OSRM: GeoJSON LineString geometry, no congestion info -> single-color polyline
        payload = (route_obj.get("payload") or {}).get("osrm_response", {})
        routes = payload.get("routes", [])
        if not routes:
            return
        geom = routes[0].get("geometry", {}) or {}
        coords = geom.get("coordinates", []) or []
        # GeoJSON: [[lon, lat], ...] -> folium [[lat, lon], ...]
        latlngs = [[c[1], c[0]] for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        if not latlngs:
            return
        meta_loc = route_obj.get("meta") or {}
        dist_km = meta_loc.get("distance_km")
        dur_min = meta_loc.get("duration_min")
        try:
            tooltip_text = f"OSRM · {float(dist_km):.2f}km · {float(dur_min):.1f}min"
        except (TypeError, ValueError):
            tooltip_text = "OSRM"
        folium.PolyLine(
            locations=latlngs,
            color=CONG_COLORS.get(0, "#3388ff"),  # No-info color (single)
            weight=8 if highlight else 5,
            opacity=0.9 if highlight else 0.7,
            tooltip=tooltip_text,
        ).add_to(m)

    else:
        # Naver API: Original logic
        payload = (route_obj.get("payload") or {}).get("naver_response", {})
        route_list = payload.get("route", {}).get("trafast", [])
        if not route_list:
            return
        r0 = route_list[0]
        path = r0.get("path", [])
        latlngs = [[p[1], p[0]] for p in path if isinstance(p, (list,tuple)) and len(p) >= 2]
        if not latlngs:
            return
        sections = r0.get("section", [])
        if sections:
            try:
                idx = 0
                for sec in sections:
                    road_kind = ROADTYPE_TO_KIND.get(int(sec.get("roadType", -1)), "local_road") if isinstance(sec.get("roadType"), (int,float)) else "local_road"
                    spd = float(sec.get("speed", np.nan)) if sec.get("speed") is not None else np.nan
                    cong_by_speed = _classify_cong_by_speed(road_kind, (None if np.isnan(spd) else spd))
                    cong = cong_by_speed if cong_by_speed is not None else int(sec.get("congestion", 0))
                    cnt = int(sec.get("pointCount", 0))
                    if cnt > 0 and idx + cnt <= len(latlngs):
                        seg = latlngs[idx:idx+cnt]
                        idx += cnt
                    else:
                        seg = latlngs
                    folium.PolyLine(
                        locations=seg,
                        color=CONG_COLORS.get(cong, "#888888"),
                        weight=8 if highlight else 5,
                        opacity=0.9 if highlight else 0.7,
                    ).add_to(m)
            except Exception:
                folium.PolyLine(locations=latlngs, color="#3388ff", weight=8 if highlight else 5, opacity=0.7).add_to(m)
        else:
            folium.PolyLine(locations=latlngs, color="#3388ff", weight=8 if highlight else 5, opacity=0.7).add_to(m)


def _offset_line(start: Tuple[float,float], end: Tuple[float,float], meters: float=30.0) -> List[Tuple[float,float]]:
    # Simple lat/lon offset (approximate): 1deg lat ~ 111320m, 1deg lon ~ 111320*cos(lat)
    lat1, lon1 = start; lat2, lon2 = end
    latc = (lat1+lat2)/2
    import math as _m
    dx = lon2 - lon1; dy = lat2 - lat1
    # Perpendicular unit vector (rotated left)
    vx, vy = -dy, dx
    norm = _m.hypot(vx, vy) or 1.0
    vx/=norm; vy/=norm
    dlat = (meters/111320.0)*vy
    dlon = (meters/(111320.0*_m.cos(_m.radians(latc))))*vx
    return [(lat1+dlat, lon1+dlon), (lat2+dlat, lon2+dlon)]


def extract_polyline(route_obj: dict) -> List[Tuple[float, float]]:
    """Extract [lat, lon] polyline from route JSON (Kakao / OSRM)."""
    meta = route_obj.get("meta", {})
    api_provider = meta.get("api_provider", "")
    latlngs: List[Tuple[float, float]] = []

    if api_provider == "kakao":
        payload = (route_obj.get("payload") or {}).get("kakao_response", {})
        routes = payload.get("routes", [])
        if routes:
            for road in routes[0].get("sections", [{}])[0].get("roads", []):
                vx = road.get("vertexes", [])
                for i in range(0, len(vx), 2):
                    if i + 1 < len(vx):
                        latlngs.append((vx[i + 1], vx[i]))
    elif api_provider == "osrm":
        payload = (route_obj.get("payload") or {}).get("osrm_response", {})
        routes = payload.get("routes", [])
        if routes:
            coords = routes[0].get("geometry", {}).get("coordinates", [])
            latlngs = [(c[1], c[0]) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
    return latlngs


def _interpolate_along_polyline(polyline: List[Tuple[float, float]], frac: float) -> Tuple[float, float]:
    """Interpolate position along a polyline at fraction frac (0..1)."""
    if not polyline:
        return (0.0, 0.0)
    if frac <= 0:
        return polyline[0]
    if frac >= 1:
        return polyline[-1]

    # Compute cumulative distances
    dists = [0.0]
    for i in range(1, len(polyline)):
        d = math.hypot(polyline[i][0] - polyline[i - 1][0], polyline[i][1] - polyline[i - 1][1])
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total == 0:
        return polyline[0]

    target = frac * total
    for i in range(1, len(dists)):
        if dists[i] >= target:
            seg_frac = (target - dists[i - 1]) / (dists[i] - dists[i - 1]) if dists[i] != dists[i - 1] else 0
            lat = polyline[i - 1][0] + seg_frac * (polyline[i][0] - polyline[i - 1][0])
            lon = polyline[i - 1][1] + seg_frac * (polyline[i][1] - polyline[i - 1][1])
            return (lat, lon)
    return polyline[-1]


def draw_uav_dash(m: folium.Map, start_latlon: Tuple[float,float], end_latlon: Tuple[float,float], color: str, tooltip: str, offset_m: float=0.0):
    pts = [start_latlon, end_latlon]
    if offset_m != 0.0:
        pts = _offset_line(start_latlon, end_latlon, meters=offset_m)
    folium.PolyLine(
        locations=pts,
        color=color,
        weight=3,
        opacity=0.95,
        dash_array="6,6",
        tooltip=tooltip
    ).add_to(m)

# Parse speed from YAML (for legend)

def get_speed_from_yaml(yaml_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not yaml_path or not exists_file(yaml_path):
        return None, None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)

        # Read speed from exact path
        amb_speed = None
        uav_speed = None

        # Read from entity_info structure
        if isinstance(y, dict) and "entity_info" in y:
            entity = y["entity_info"]

            # ambulance.velocity
            if isinstance(entity, dict) and "ambulance" in entity:
                amb_data = entity["ambulance"]
                if isinstance(amb_data, dict) and "velocity" in amb_data:
                    amb_speed = float(amb_data["velocity"])

            # uav.velocity
            if isinstance(entity, dict) and "uav" in entity:
                uav_data = entity["uav"]
                if isinstance(uav_data, dict) and "velocity" in uav_data:
                    uav_speed = float(uav_data["velocity"])

        return amb_speed, uav_speed
    except Exception:
        return None, None


def get_total_samples_from_yaml(yaml_path: Optional[str]) -> int:
    """Read total_samples value from YAML (for performance optimization)"""
    if not yaml_path or not exists_file(yaml_path):
        return 0
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        return y.get('rule_info', {}).get('total_samples', 0)
    except Exception:
        return 0


# ------------------------------
# Execution/Re-run
# ------------------------------
# PowerShell-related functions are no longer used (using Orchestrator instead)

# ------------------------------
# Page common settings + CSS (multiselect ellipsis mitigation)
# ------------------------------
st.set_page_config(page_title="MCI Streamlit", page_icon="📊", layout="wide")


with st.sidebar:
    st.header("Settings")

    base_input = st.text_input(
        "base_path",
        st.session_state.base_path,
        placeholder="e.g. C:\\Users\\USER\\MCI",
        disabled=IS_CLOUD,
    )

    if IS_CLOUD:
        st.caption(f"☁️ Cloud mode: base_path is fixed to `{CLOUD_BASE_PATH}`.")

    # Button only works locally
    if (not IS_CLOUD) and st.button("Set base_path"):
        st.session_state.base_path = norm(base_input)
        if base_ok(norm(base_input)):
            st.success("✅ base_path set! Select a scenario or create a new one in the Generate tab.")
    if st.session_state.base_path and not base_ok(st.session_state.base_path):
        st.warning("Invalid base_path. (scenarios folder required)")
    st.text("※ Click the button above to start")
    if base_ok(st.session_state.base_path):
        exps = list_experiments_any(st.session_state.base_path)
        st.caption("📂 Select existing scenario (optional)")

        # Ensure stored value is still valid; reset if not
        if st.session_state.selected_exp not in exps:
            st.session_state.selected_exp = ""
        # Reset coord when exp changes
        def _on_exp_change():
            st.session_state.selected_coord = ""

        st.selectbox(
            "Experiment ID",
            options=[""] + exps,
            key="selected_exp",
            on_change=_on_exp_change,
        )
        coords = list_coords_from_scenarios(st.session_state.base_path, st.session_state.selected_exp) if st.session_state.selected_exp else []
        if st.session_state.selected_coord not in coords:
            st.session_state.selected_coord = ""
        st.selectbox(
            "Coordinate folder",
            options=[""] + coords,
            key="selected_coord",
        )

        # 1) Mini-map (when coordinate is selected)
        if st.session_state.selected_coord:
            lat, lon = coord_center(st.session_state.selected_coord)
            st.caption("Current Coordinate")

            # -- folium-based render (same display logic: single point) --
            try:
                import folium
                from streamlit_folium import st_folium

                chosen_tile = "CartoDB positron"

                m = folium.Map(location=(lat, lon), zoom_start=12, control_scale=True, tiles=chosen_tile)
                folium.CircleMarker(
                    location=(lat, lon), radius=5, weight=1, opacity=0.9,
                    fill=True, fill_opacity=0.8, tooltip=f"{lat:.6f}, {lon:.6f}"
                ).add_to(m)
                # Minimize license attribution display size
                from folium import Element
                m.get_root().html.add_child(Element("""
                <style>
                .leaflet-control-attribution {
                font-size: 1px !important;
                opacity: .55 !important;
                background: rgba(255,255,255,.6) !important;
                padding: 2px 6px !important;
                border-radius: 1px !important;
                }
                .leaflet-control-attribution a { color: inherit !important; text-decoration: none !important; }
                /* (optional) Enable to move attribution to top instead of bottom */
                .leaflet-bottom.leaflet-right { bottom: auto !important; top: 6px !important; right: 8px !important; }
                </style>
                """))
                st_folium(m, width='stretch', height=260)

            except Exception:
                # (fallback) Streamlit default map (no tile customization)
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), width='stretch')

        st.divider()



# CSS: Global UI theme + multiselect width expansion
st.markdown("""
<style>
/* -- Preserve original: multiselect width -- */
.stMultiSelect [data-baseweb="select"]{max-width:100%!important}

/* -- Font -- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', 'DM Sans', -apple-system, sans-serif !important;
}

/* -- Background -- */
.stApp {
    background: #141417;
}

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: #1c1c21 !important;
    border-right: 1px solid rgba(226, 160, 74, 0.1) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2a04a !important;
}
[data-testid="stSidebar"] label {
    color: #a1a1aa !important;
    font-weight: 500;
    font-size: 0.85rem;
}

/* -- Main title -- */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    padding-bottom: 4px;
    color: #e4e4e7 !important;
}
.gradient-text {
    background: linear-gradient(90deg, #e2a04a 0%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* -- Subheader -- */
h2, h3 {
    color: #e4e4e7 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(226, 160, 74, 0.18);
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}

/* -- Tab bar -- */
.stTabs [data-baseweb="tab-list"] {
    background: #1c1c21;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 500;
    color: #71717a !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(226, 160, 74, 0.15) !important;
    color: #e2a04a !important;
    box-shadow: none;
    border-bottom: 2px solid #e2a04a;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #d4d4d8 !important;
    background: rgba(255, 255, 255, 0.04);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* -- Button (primary: amber) -- */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid rgba(226, 160, 74, 0.4) !important;
    background: rgba(226, 160, 74, 0.12) !important;
    color: #e2a04a !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(226, 160, 74, 0.22) !important;
    border-color: #e2a04a !important;
}
.stButton > button:active { transform: translateY(0); }

/* -- Input fields -- */
[data-baseweb="input"],
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div > div {
    background: #1c1c21 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease;
}
[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(226, 160, 74, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(226, 160, 74, 0.08) !important;
}

/* -- Dropdown menu -- */
[data-baseweb="popover"] {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden;
}
[data-baseweb="menu"] { background: #1c1c21 !important; }

/* -- Expander -- */
[data-testid="stExpander"] {
    background: #1c1c21 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
}

/* -- Metric card -- */
[data-testid="stMetric"] {
    background: #1c1c21;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid #e2a04a;
    border-radius: 10px;
    padding: 18px 20px;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255, 255, 255, 0.1);
    border-left-color: #e2a04a;
}
[data-testid="stMetricLabel"] {
    color: #71717a !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: #e4e4e7 !important;
    font-weight: 700 !important;
}

/* -- DataFrame -- */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: 8px !important;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* -- Divider -- */
hr {
    border-color: rgba(255, 255, 255, 0.06) !important;
    margin: 24px 0 !important;
}

/* -- Alert messages -- */
.stAlert, [data-testid="stAlert"] { border-radius: 8px !important; }

/* -- Checkbox/radio hover -- */
.stCheckbox label:hover, .stRadio label:hover { color: #e2a04a !important; }

/* -- Scrollbar -- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #141417; }
::-webkit-scrollbar-thumb {
    background: #3f3f46;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #52525b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1><span>🚑</span> <span class="gradient-text">MCI Disaster Simulation Dashboard</span></h1>', unsafe_allow_html=True)

tabs = st.tabs(["Maps", "Scenarios", "Analytics", "Data Tables", "Rerun"])

# ------------------------------
# Scenarios tab
# ------------------------------
with tabs[1]:
    st.subheader("Selected Scenario")
    bp = st.session_state.base_path; exp = st.session_state.selected_exp; coord = st.session_state.selected_coord

    # Reset log load state when scenario changes
    current_scenario_key = f"{exp}_{coord}"
    if st.session_state.get("last_scenario_key") != current_scenario_key:
        st.session_state.logs_loaded = False
        st.session_state.last_scenario_key = current_scenario_key

    if bp and exp and coord:
        yaml_path = find_yaml_in_coord(bp, exp, coord)
        st.write("**YAML**:", yaml_path or "(none)")
        
        smdf = read_experiment_summary_csv(bp, exp)
        info, site_addr = summarize_experiment(smdf) if smdf is not None else ({}, None)
        lat, lon = coord_center(coord)

        site_info, sim_info = summarize_experiment_extended(smdf)
        site_info["Coordinate"] = f"{lat:.6f}, {lon:.6f}"

        with st.expander("View Experiment Summary", expanded=True):
            left, right = st.columns([0.42, 0.58])
            with left:
                st.markdown("**Incident Site**")
                st.dataframe(pd.DataFrame({"Field":list(site_info.keys()), "Value":list(site_info.values())}), width='stretch', height=170)
            with right:
                st.markdown("**Simulation Info** (summary.csv)")
                if sim_info:
                    st.dataframe(pd.DataFrame(sorted(sim_info.items(), key=lambda x: x[0]), columns=["Field","Value"]), width='stretch', height=170)
                else:
                    st.caption("No simulation info columns found in summary.csv.")


        # Performance optimization: check simulation iterations directly from summary CSV (latest attempt)
        total_samples = 0
        if smdf is not None and not smdf.empty and "sim_repeats" in smdf.columns:
            # Among rows matching the current coordinate, pick the most recent attempt (last row)
            coord_rows = smdf[smdf["coord"] == coord]
            if not coord_rows.empty:
                latest_row = coord_rows.iloc[-1]  # most recent attempt
                total_samples_val = latest_row.get("sim_repeats")
                if pd.notna(total_samples_val):
                    total_samples = int(total_samples_val)

        st.markdown("### 🧾 Execution Log")

        # Debug: verify total_samples value
        if total_samples > 0:
            st.caption(f"🔍 Detected simulation iterations: {total_samples} (latest run in summary CSV)")

        # Conditional loading based on total_samples (101+ iterations: button disabled too)
        if total_samples >= 101:
            st.warning(f"⚠️ Simulation iterations ({total_samples}) >= 101. Log viewer disabled.")
            st.info(f"📁 Check log files directly: `experiment_logs/{coord}_*.txt`")
            st.caption(f"💡 For performance, logs with 101+ iterations should be checked in the source folder.")

            # Show button in disabled state
            st.button("Load Log Files", key="load_logs_btn_disabled", disabled=True, help="Log viewer is disabled for 101+ iterations")
            logs = None
        else:
            # Performance optimization: load logs only on button click (improves other tab loading speed)
            if st.button("Load Log Files", key="load_logs_btn", help="Click to load logs"):
                st.session_state.logs_loaded = True

            if st.session_state.get("logs_loaded", False):
                logs = experiment_log_candidates(bp, exp, coord)
            else:
                st.info("💡 Click 'Load Log Files' above to view logs. (Disabled by default for faster tab loading)")
                logs = None

        if logs:
            log_sel = st.selectbox("Select Log File (experiment_logs/<coord>)", logs)
            log_text = _read_text_any(log_sel)
            blocks = parse_log_blocks(log_text)
            # Hide (Unlabeled) blocks
            rule_list = [b["rule"] for b in blocks if b.get("rule")!="(Unlabeled)"] if blocks else []
            if not rule_list:
                st.info("No event patterns found. (Check log format)")
            else:
                sel_rule = st.selectbox("Select Rule", options=rule_list, index=0)

                # [added] Collect available Iter list for the selected Rule
                rule_blocks = [b for b in blocks if b.get("rule") == sel_rule]
                iter_list = sorted({b.get("iter") for b in rule_blocks if b.get("iter") is not None})

                # [added] Iter selector (shown only when available)
                sel_iter = None
                if iter_list:
                    sel_iter = st.selectbox("Select Iteration", iter_list, index=0, help="Based on 'Iter : n' lines in the log")
                    # Filter by selected Rule & Iter combination
                    cand_blocks = [b for b in rule_blocks if b.get("iter") == sel_iter]
                else:
                    # If no Iter lines found, keep existing behavior (first block)
                    st.caption("No 'Iter' lines found — treating this log as a single run.")
                    cand_blocks = rule_blocks

                # Final block selection (if multiple, use first block)
                blk = cand_blocks[0] if cand_blocks else None

                # (optional) Show current selection status
                st.caption(f"Selected: Rule={sel_rule} / Iter={sel_iter if sel_iter is not None else 'N/A'}")


                st.markdown("#### 👤 Patient Story (Summary)")
                psum = build_patient_summary(blk["events"]) if blk else pd.DataFrame()
                if psum.empty:
                    st.caption("No patient events found.")
                else:
                    st.dataframe(psum, width='stretch', height=340)
                    _suffix = f"_iter{sel_iter}" if sel_iter is not None else ""
                    st.download_button(
                        "⬇️ Patient Timeline (csv)",
                        psum.to_csv(index=False).encode('utf-8-sig'),
                        file_name=f"patient_timeline{_suffix}.csv"
                    )

                    # ── Patient Story Animation ──
                    st.markdown("#### Patient Story Animation")
                    st.caption("Animated timeline showing patient state transitions over time.")

                    import plotly.graph_objects as go_anim

                    _sev_labels = {0: "Red", 1: "Yellow", 2: "Green", 3: "Black"}
                    _sev_colors_map = {"Red": "#e74c3c", "Yellow": "#f39c12", "Green": "#2ecc71", "Black": "#2c3e50", "?": "#95a5a6"}

                    # Build per-patient state at each event time
                    _anim_events = sorted(blk["events"], key=lambda r: r["t"]) if blk else []
                    if _anim_events:
                        # Collect all unique times and patient states
                        _all_patients = sorted(set(e.get("p") for e in _anim_events if e.get("p") is not None))
                        if _all_patients:
                            _state_names = {
                                "p_rescue": "Rescued",
                                "amb_arrival_site": "AMB Ready",
                                "uav_arrival_site": "UAV Ready",
                                "amb_arrival_hospital": "At Hospital",
                                "uav_arrival_hospital": "At Hospital",
                                "p_care_ready": "In Treatment",
                                "p_def_care": "Completed",
                            }
                            _state_order = {"Waiting": 0, "Rescued": 1, "AMB Ready": 1, "UAV Ready": 1,
                                            "In Transport": 2, "At Hospital": 3, "In Treatment": 4, "Completed": 5}

                            # Track each patient's current state over time
                            _p_current = {p: "Waiting" for p in _all_patients}
                            _p_severity = {}

                            # Determine severity for each patient from onset
                            _onset_ev = [e for e in _anim_events if e.get("ev") == "onset"]
                            # Get severity from p_rescue events
                            for e in _anim_events:
                                p = e.get("p")
                                if p is not None and e.get("ev") == "p_rescue":
                                    # Infer severity from patient ID ranges (or use severity field if available)
                                    _p_severity.setdefault(p, "?")

                            # Collect snapshots: use exact event times to avoid missing events
                            _exact_times = sorted(set(e["t"] for e in _anim_events))
                            # Sample up to 40 frames but always keep first + last + all state-change times
                            _max_frames = 40
                            if len(_exact_times) > _max_frames:
                                _step = max(1, len(_exact_times) // _max_frames)
                                _sampled = _exact_times[::_step]
                                # Always include the very last event time (all patients done)
                                if _sampled[-1] != _exact_times[-1]:
                                    _sampled.append(_exact_times[-1])
                                _time_points = sorted(set(_sampled))
                            else:
                                _time_points = _exact_times

                            _frames_data = []
                            _p_current = {p: "Waiting" for p in _all_patients}

                            _ev_idx = 0
                            for t_snap in _time_points:
                                # Advance state for all events up to this time
                                while _ev_idx < len(_anim_events) and _anim_events[_ev_idx]["t"] <= t_snap:
                                    e = _anim_events[_ev_idx]
                                    p = e.get("p")
                                    ev_name = e.get("ev", "")
                                    if p is not None and ev_name in _state_names:
                                        _p_current[p] = _state_names[ev_name]
                                    _ev_idx += 1

                                for p in _all_patients:
                                    _frames_data.append({
                                        "time": t_snap,
                                        "patient": f"P{p}",
                                        "state": _p_current[p],
                                        "state_num": _state_order.get(_p_current[p], 0),
                                    })

                            _adf = pd.DataFrame(_frames_data)

                            if not _adf.empty:
                                # Build animated figure with frames
                                _patients_sorted = sorted(_adf["patient"].unique(), key=lambda x: int(x[1:]))
                                _state_color_map = {
                                    "Waiting": "#7f8c8d", "Rescued": "#3498db",
                                    "AMB Ready": "#3498db", "UAV Ready": "#9b59b6",
                                    "In Transport": "#f39c12", "At Hospital": "#e67e22",
                                    "In Treatment": "#e74c3c", "Completed": "#2ecc71",
                                }

                                _first_t = _time_points[0]
                                _init = _adf[_adf["time"] == _first_t]

                                fig_anim = go_anim.Figure(
                                    data=[go_anim.Bar(
                                        y=_init["patient"],
                                        x=_init["state_num"],
                                        orientation='h',
                                        marker=dict(
                                            color=[_state_color_map.get(s, "#95a5a6") for s in _init["state"]],
                                        ),
                                        text=_init["state"],
                                        textposition="inside",
                                        hovertemplate="%{y}: %{text}<extra></extra>",
                                    )],
                                    layout=go_anim.Layout(
                                        xaxis=dict(
                                            range=[-0.5, 6],
                                            tickmode="array",
                                            tickvals=[0, 1, 2, 3, 4, 5],
                                            ticktext=["Waiting", "Rescued", "Transport", "Hospital", "Treatment", "Completed"],
                                        ),
                                        yaxis=dict(categoryorder="array", categoryarray=_patients_sorted[::-1]),
                                        height=max(400, len(_patients_sorted) * 20),
                                        margin=dict(l=60, r=20, t=50, b=30),
                                        title=f"Patient State at t = {_first_t:.1f} min",
                                        updatemenus=[dict(
                                            type="buttons",
                                            showactive=False,
                                            y=1.12, x=0.5, xanchor="center",
                                            buttons=[
                                                dict(label="Play", method="animate",
                                                     args=[None, {"frame": {"duration": 400, "redraw": True},
                                                                  "fromcurrent": True,
                                                                  "transition": {"duration": 200}}]),
                                                dict(label="Pause", method="animate",
                                                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                    "mode": "immediate",
                                                                    "transition": {"duration": 0}}]),
                                            ],
                                        )],
                                        sliders=[dict(
                                            active=0,
                                            steps=[
                                                dict(args=[[str(t)], {"frame": {"duration": 300, "redraw": True},
                                                                       "mode": "immediate",
                                                                       "transition": {"duration": 200}}],
                                                     label=f"{t:.0f}",
                                                     method="animate")
                                                for t in _time_points
                                            ],
                                            x=0.05, len=0.9,
                                            y=-0.05,
                                            currentvalue=dict(prefix="Time: ", suffix=" min", visible=True),
                                            transition=dict(duration=200),
                                        )],
                                    ),
                                    frames=[
                                        go_anim.Frame(
                                            data=[go_anim.Bar(
                                                y=_adf[_adf["time"] == t]["patient"],
                                                x=_adf[_adf["time"] == t]["state_num"],
                                                orientation='h',
                                                marker=dict(
                                                    color=[_state_color_map.get(s, "#95a5a6")
                                                           for s in _adf[_adf["time"] == t]["state"]],
                                                ),
                                                text=_adf[_adf["time"] == t]["state"],
                                                textposition="inside",
                                                hovertemplate="%{y}: %{text}<extra></extra>",
                                            )],
                                            layout=go_anim.Layout(title=f"Patient State at t = {t:.1f} min"),
                                            name=str(t),
                                        )
                                        for t in _time_points
                                    ],
                                )

                                st.plotly_chart(fig_anim, width='stretch')

                                # Legend
                                _legend_md = " | ".join(
                                    f"<span style='color:{c}'>{s}</span>"
                                    for s, c in _state_color_map.items()
                                )
                                st.markdown(f"**States:** {_legend_md}", unsafe_allow_html=True)
                                st.caption("Press **Play** to animate or drag the slider. Each frame shows all patients' states at that time point.")


                st.markdown("#### 🧰 Full Event Table")
                ev_df = pd.DataFrame(blk["events"]).rename(columns={"t":"Time","eid":"EventID","ev":"Event","p":"Patient","a":"Ambulance","u":"UAV","h":"Hospital"}) if blk else pd.DataFrame()
                st.dataframe(ev_df, width='stretch', height=320)
                _suffix = f"_iter{sel_iter}" if sel_iter is not None else ""
                st.download_button("⬇️ Full Events (csv)", ev_df.to_csv(index=False).encode('utf-8-sig'), file_name=f"events_all{_suffix}.csv")

            st.markdown("#### 🗂 View Raw Log")
            with st.expander("Expand Raw Text", expanded=False):
                st.code(log_text[:30000] + ("\n... (truncated)" if len(log_text) > 30000 else ""))
                st.download_button("Download Raw Log", log_text, file_name=os.path.basename(log_sel))

            st.markdown("---")
            st.markdown("### 🛈 Action/Rule Reference")
            st.markdown(ACTION_TOOLTIP_MD)
        else:
            st.info("No log files found for this combination.")

        # ── Trace Replay (Per-Patient Timeline) ──
        st.markdown("---")
        st.markdown("### Simulation Trace Replay")
        st.caption("Visualize per-patient event timeline from trace data. "
                   "Run simulation with `--trace` flag to generate trace_*.json files.")

        _trace_dir = Path(bp) / "results" / exp / coord
        _trace_files = sorted(_trace_dir.glob("trace_*.json")) if _trace_dir.is_dir() else []

        if not _trace_files:
            st.info("No trace files found. Run simulation with `--trace` flag to enable trace logging.\n\n"
                    "```bash\npython main.py --config_path <config.yaml> --trace\n```")
        else:
            _trace_sel = st.selectbox("Trace file", [f.name for f in _trace_files], key="trace_file_sel")
            _trace_path = _trace_dir / _trace_sel

            try:
                with open(_trace_path, "r", encoding="utf-8") as _tf:
                    _trace_data = json.load(_tf)

                _trace_keys = list(_trace_data.keys())
                if not _trace_keys:
                    st.warning("Trace file is empty.")
                else:
                    _trace_rule = st.selectbox("Rule / Iteration", _trace_keys, key="trace_rule_sel")
                    _events = _trace_data[_trace_rule]

                    if not _events:
                        st.info("No trace events for this rule/iteration.")
                    else:
                        import plotly.graph_objects as go_trace

                        _severity_names = {0: "Red", 1: "Yellow", 2: "Green", 3: "Black"}
                        _severity_colors = {0: "#e74c3c", 1: "#f1c40f", 2: "#2ecc71", 3: "#2c3e50"}

                        # Build per-patient timeline
                        _patient_events = {}
                        for _ev in _events:
                            pid = _ev.get("patient_id")
                            if pid is not None:
                                _patient_events.setdefault(pid, []).append(_ev)

                        if not _patient_events:
                            st.info("No patient-level events in trace.")
                        else:
                            # Gantt-style timeline
                            _gantt_data = []
                            for pid in sorted(_patient_events.keys()):
                                pevts = sorted(_patient_events[pid], key=lambda x: x["time"])
                                sev = pevts[0].get("severity", -1) if pevts else -1
                                for j in range(len(pevts) - 1):
                                    _gantt_data.append({
                                        "patient_id": pid,
                                        "severity": _severity_names.get(sev, "?"),
                                        "event_start": pevts[j]["event"],
                                        "event_end": pevts[j + 1]["event"],
                                        "t_start": pevts[j]["time"],
                                        "t_end": pevts[j + 1]["time"],
                                        "color": _severity_colors.get(sev, "#95a5a6"),
                                        "vehicle": pevts[j].get("vehicle", ""),
                                        "hospital": pevts[j].get("hospital_id", ""),
                                    })

                            if _gantt_data:
                                _gdf = pd.DataFrame(_gantt_data)

                                fig_gantt = go_trace.Figure()
                                for sev_name, sev_color in [("Red", "#e74c3c"), ("Yellow", "#f1c40f"),
                                                            ("Green", "#2ecc71"), ("Black", "#2c3e50")]:
                                    _sdf = _gdf[_gdf["severity"] == sev_name]
                                    if _sdf.empty:
                                        continue
                                    for _, row in _sdf.iterrows():
                                        fig_gantt.add_trace(go_trace.Bar(
                                            y=[f"P{row['patient_id']}"],
                                            x=[row["t_end"] - row["t_start"]],
                                            base=row["t_start"],
                                            orientation='h',
                                            marker=dict(color=sev_color, opacity=0.7),
                                            name=sev_name,
                                            showlegend=False,
                                            hovertemplate=(
                                                f"Patient {row['patient_id']} ({sev_name})<br>"
                                                f"{row['event_start']} -> {row['event_end']}<br>"
                                                f"t={row['t_start']:.1f} - {row['t_end']:.1f} min<br>"
                                                f"Vehicle: {row['vehicle']}<br>"
                                                f"Hospital: {row['hospital']}<extra></extra>"
                                            ),
                                        ))

                                n_patients = len(_patient_events)
                                fig_gantt.update_layout(
                                    barmode="overlay",
                                    xaxis_title="Time (minutes)",
                                    yaxis_title="Patient",
                                    height=max(400, n_patients * 22),
                                    margin=dict(l=60, r=20, t=30, b=30),
                                    title=f"Patient Timeline ({_trace_rule[:50]})",
                                )
                                st.plotly_chart(fig_gantt, width='stretch')

                            # Event summary table
                            with st.expander("Trace Event Log"):
                                st.dataframe(pd.DataFrame(_events), width='stretch', height=400)

                            # Summary stats
                            _n_rescue = sum(1 for e in _events if e["event"] == "rescue")
                            _n_transport = sum(1 for e in _events if e["event"] == "transport_start")
                            _n_arrival = sum(1 for e in _events if e["event"] == "hospital_arrival")
                            _n_diversion = sum(1 for e in _events if e["event"] == "diversion")
                            _n_care = sum(1 for e in _events if e["event"] == "care_complete")

                            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
                            col_t1.metric("Rescues", _n_rescue)
                            col_t2.metric("Transports", _n_transport)
                            col_t3.metric("Arrivals", _n_arrival)
                            col_t4.metric("Diversions", _n_diversion)
                            col_t5.metric("Completed", _n_care)

            except Exception as e_trace:
                st.error(f"Failed to load trace: {e_trace}")


# ------------------------------
# Maps tab (multi-select + UAV dispatch/transport toggle + enhanced legend)
# ------------------------------
with tabs[0]:
    st.subheader("Map Visualization")
    bp   = st.session_state.base_path
    exp  = st.session_state.selected_exp
    coord= st.session_state.selected_coord

    _map_mode = st.radio("Mode", ["Static Map", "Animation"], horizontal=True, key="map_mode_radio")

    # -- Map drawn in 'top-level' container, options placed below --
    map_holder = st.container()
    st.divider()

    # Coordinates/defaults
    lat, lon = coord_center(coord)
    center = [lat, lon]
    site_addr = st.session_state.get("site_addr", "")

    # Load routes/data
    rdirs = get_routes_dirs(bp, exp, coord)
    patient_cnt = get_patient_count(bp, exp, coord)
    yaml_path = find_yaml_in_coord(bp, exp, coord)

    # JSON routes (full load; for map visualization - independent of simulation iterations)
    c2s_all = load_json_files(rdirs["center2site"])  # Center→Site
    h2s_all = load_json_files(rdirs["hos2site"])     # Site→Hospitals

    # Hospital/center metadata
    hosp_xl   = read_excel_hospital(bp)   # Excel (hospital name, grade code, x/y coords, phone/address, etc.)

    # road/euc CSV (for table/distance reference) -- cached (top-level function)
    hinfo_df, center_df, ambinfo_df, dist_road_df, hinfo_euc_df, dist_euc_df = _load_scenario_csvs(bp, exp, coord)
    dist_road_map = dict(zip(dist_road_df["Index"], dist_road_df["distance"])) if not dist_road_df.empty else {}
    dur_road_map = dict(zip(dist_road_df["Index"], dist_road_df["duration"])) if not dist_road_df.empty and "duration" in dist_road_df.columns else {}
    dist_euc_map  = dict(zip(dist_euc_df["Index"], dist_euc_df["distance"])) if not dist_euc_df.empty else {}

    # Excel name -> coordinate mapping
    xl_coord = {}
    if hosp_xl is not None and "institution_name" in hosp_xl.columns:
        for _, rr in hosp_xl.iterrows():
            nm = str(rr.get("institution_name","")).strip()
            y  = rr.get("y_coord", rr.get("y", None))
            x  = rr.get("x_coord", rr.get("x", None))
            if nm and pd.notna(y) and pd.notna(x):
                xl_coord[nm] = (float(y), float(x))

    # Grade code labeling (1=Tertiary Hospital, 11=General Hospital, else=Hospital)
    def code_to_grade(c):
        try:
            c = int(c)
        except Exception:
            return "Hospital"
        if c == 1:  return "Tertiary Hospital"
        if c == 11: return "General Hospital"
        return "Hospital"

    # -- Map theme (just above the map) --
    theme = st.radio("Map Theme", ["Light","Dark"], horizontal=True, key="theme_radio_maps_bottom")
    if theme == "Light":
        tile_name = st.selectbox("Light Tile", ["OpenStreetMap","CartoDB Positron"], index=0, key="light_tile_select")
    else:
        tile_name = "CartoDB Dark_Matter"

    if not (bp and exp and coord):
        st.info("Select base_path / Experiment / Coord from the sidebar.")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────
    # AMB Routes (C->S table / S->H table)
    # ─────────────────────────────────────────────────────────────────────
    _route_exp = st.expander("Route Details & Selection", expanded=False)
    with _route_exp:
        st.markdown("### AMB Routes")
        col_amb_c2s, col_amb_s2h = st.columns(2)

    # --- AMB: C->S (Dispatch) table (Show, Index, Fire Station, Distance) ---
    with col_amb_c2s:
        st.markdown("**Fire Station → Incident Site (Dispatch)**")

        # Build table based on amb_info_road.csv (index/name/distance=init_distance/time=duration)
        if not ambinfo_df.empty:
            rename_dict = {
                "Index":"Index",
                "Fire Station":"Fire Station",
                "init_distance":"Distance(km)",
            }
            cols = ["Index","Fire Station","Distance(km)"]
            if "Fleet Size" in ambinfo_df.columns:
                rename_dict["Fleet Size"] = "Fleet Size"
                cols.append("Fleet Size")
            if "duration" in ambinfo_df.columns:
                rename_dict["duration"] = "Duration(min)"
                cols.append("Duration(min)")

            c2s_df = ambinfo_df.rename(columns=rename_dict)[cols].copy()
            # Explicitly convert all numeric columns
            c2s_df["Index"] = pd.to_numeric(c2s_df["Index"], errors="coerce").fillna(0).astype(int)
            c2s_df["Distance(km)"] = pd.to_numeric(c2s_df["Distance(km)"], errors="coerce").fillna(0.0).round(2)
            if "Duration(min)" in c2s_df.columns:
                c2s_df["Duration(min)"] = pd.to_numeric(c2s_df["Duration(min)"], errors="coerce").fillna(0.0).round(1)
            if "Fleet Size" in c2s_df.columns:
                c2s_df["Fleet Size"] = pd.to_numeric(c2s_df["Fleet Size"], errors="coerce").fillna(1).astype(int)
            # Ensure fire station names are strings
            c2s_df["Fire Station"] = c2s_df["Fire Station"].astype(str)
        else:
            c2s_df = pd.DataFrame({
                "Index": pd.Series(dtype='int'),
                "Fire Station": pd.Series(dtype='str'),
                "Distance(km)": pd.Series(dtype='float'),
                "Duration(min)": pd.Series(dtype='float'),
            })

        # Default selection state
        if "amb_c2s_sel_idx" not in st.session_state:
            st.session_state.amb_c2s_sel_idx = set(c2s_df["Index"].tolist())

        b1, b2 = st.columns(2)
        if b1.button("Select All (C→S)"):
            st.session_state.amb_c2s_sel_idx = set(c2s_df["Index"].tolist())
        if b2.button("Deselect All (C→S)"):
            st.session_state.amb_c2s_sel_idx = set()

        c2s_df_show = c2s_df.copy()
        c2s_df_show["Show"] = c2s_df_show["Index"].apply(lambda i: i in st.session_state.amb_c2s_sel_idx)
        # Explicit boolean conversion to prevent React error #185
        c2s_df_show["Show"] = c2s_df_show["Show"].astype(bool)
        # Move 'Show' column to front
        show_cols = ["Show","Index","Fire Station","Distance(km)"]
        if "Duration(min)" in c2s_df_show.columns:
            show_cols.append("Duration(min)")
        if "Fleet Size" in c2s_df_show.columns:
            show_cols.insert(3, "Fleet Size")
        c2s_df_show = c2s_df_show[show_cols].reset_index(drop=True)

        col_cfg = {
            "Show": st.column_config.CheckboxColumn("Show"),
            "Index": st.column_config.NumberColumn("Index", disabled=True),
            "Fire Station": st.column_config.TextColumn("Fire Station", disabled=True),
            "Distance(km)": st.column_config.NumberColumn("Distance(km)", disabled=True, format="%.2f"),
        }
        if "Duration(min)" in c2s_df_show.columns:
            col_cfg["Duration(min)"] = st.column_config.NumberColumn("Duration(min)", disabled=True, format="%.1f")
        if "Fleet Size" in c2s_df_show.columns:
            col_cfg["Fleet Size"] = st.column_config.NumberColumn("Fleet Size", disabled=True)
        edited_c2s = st.data_editor(
            c2s_df_show,
            width='stretch',
            num_rows="fixed",
            hide_index=True,
            column_config=col_cfg,
            key="tbl_c2s"
        )
        st.session_state.amb_c2s_sel_idx = set(edited_c2s.loc[edited_c2s["Show"]==True, "Index"].tolist())

    # --- AMB: S->H (Transport) table (Show, Index, Hospital, Grade Code, Hospital Grade, Distance) ---
    with col_amb_s2h:
        st.markdown("**Incident Site → Hospital (Transport)**")

        # Based on hospital_info_road + distance_Hos2Site_road
        if not hinfo_df.empty:
            s2h_df = hinfo_df.rename(columns={
                "Hospital Name":"Hospital"
            })[["Index","Hospital","Grade Code"]].copy()
            s2h_df["Hospital Grade"] = s2h_df["Grade Code"].apply(code_to_grade)
            if dist_road_map:
                s2h_df["Distance(km)"] = s2h_df["Index"].map(dist_road_map).round(2)
            if dur_road_map:
                s2h_df["Duration(min)"] = s2h_df["Index"].map(dur_road_map).round(1)
        else:
            s2h_df = pd.DataFrame(columns=["Index","Hospital","Grade Code","Hospital Grade","Distance(km)","Duration(min)"])

        if "amb_s2h_sel_idx" not in st.session_state:
            st.session_state.amb_s2h_sel_idx = set(s2h_df["Index"].tolist())

        c, d = st.columns(2)
        if c.button("Select All (S→H)"):
            st.session_state.amb_s2h_sel_idx = set(s2h_df["Index"].tolist())
        if d.button("Deselect All (S→H)"):
            st.session_state.amb_s2h_sel_idx = set()

        s2h_df_show = s2h_df.copy()
        s2h_df_show["Show"] = s2h_df_show["Index"].apply(lambda i: i in st.session_state.amb_s2h_sel_idx)
        # Move 'Show' column to front
        s2h_show_cols = ["Show","Index","Hospital","Grade Code","Hospital Grade","Distance(km)"]
        if "Duration(min)" in s2h_df_show.columns:
            s2h_show_cols.append("Duration(min)")
        s2h_df_show = s2h_df_show[s2h_show_cols]

        s2h_col_cfg = {
            "Show":     st.column_config.CheckboxColumn("Show"),
            "Index":   st.column_config.NumberColumn("Index", disabled=True),
            "Hospital":     st.column_config.TextColumn("Hospital", disabled=True),
            "Grade Code": st.column_config.NumberColumn("Grade Code", disabled=True),
            "Hospital Grade": st.column_config.TextColumn("Hospital Grade", disabled=True),
            "Distance(km)": st.column_config.NumberColumn("Distance(km)", disabled=True, format="%.2f"),
        }
        if "Duration(min)" in s2h_df_show.columns:
            s2h_col_cfg["Duration(min)"] = st.column_config.NumberColumn("Duration(min)", disabled=True, format="%.1f")

        edited_s2h = st.data_editor(
            s2h_df_show,
            width='stretch',
            num_rows="fixed",
            hide_index=True,
            column_config=s2h_col_cfg,
            key="tbl_s2h"
        )
        st.session_state.amb_s2h_sel_idx = set(edited_s2h.loc[edited_s2h["Show"]==True, "Index"].tolist())

    # ─────────────────────────────────────────────────────────────────────
    # UAV Routes (dispatch/transport -- straight-line distance display)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### UAV Routes")
    col_uav_out, col_uav_back = st.columns(2)

    # Dispatch (Hospital -> Incident): Helipad hospitals (read from uav_info.csv)
    with col_uav_out:
        st.markdown("**Helipad Hospital → Incident Site (Dispatch)**")

        uav_dispatch_latlons = []

        # Read uav_info.csv
        uav_info_path = Path(bp) / "scenarios" / exp / coord / "uav_info.csv"
        if os.path.exists(uav_info_path):
            try:
                uav_df = pd.read_csv(uav_info_path, encoding="utf-8-sig")
                uav_df.rename(columns={
                    "type_code": "Grade Code", "institution_name": "Hospital Name",
                    "num_or": "ORs", "num_beds": "Beds",
                }, inplace=True)

                # Check new format (6 columns)
                if "Hospital Name" in uav_df.columns:
                    for _, row in uav_df.iterrows():
                        name = str(row["Hospital Name"]).strip()
                        code = row.get("Grade Code", 1)

                        # Look up coordinates from Excel
                        if name in xl_coord:
                            y, x = xl_coord[name]
                            uav_dispatch_latlons.append((y, x, name, code))
                        else:
                            print(f"Warning: UAV hospital '{name}' coordinates not found")
                else:
                    print("Warning: uav_info.csv old format (2 columns) - update needed")
                    # Fallback: use tier3
                    if not hinfo_df.empty and {"Grade Code","Hospital Name"}.issubset(hinfo_df.columns):
                        tmp = hinfo_df.copy()
                        tmp["Grade Code"] = pd.to_numeric(tmp["Grade Code"], errors="coerce")
                        for _, rr in tmp[tmp["Grade Code"]==1].iterrows():
                            name = str(rr.get("Hospital Name","Tier3")).strip()
                            if name in xl_coord:
                                y, x = xl_coord[name]
                                uav_dispatch_latlons.append((y, x, name, 1))
            except Exception as e:
                print(f"Warning: uav_info.csv load failed: {e}")
        else:
            print(f"Warning: uav_info.csv not found: {uav_info_path}")

        # Fallback to tier3 if no data
        if not uav_dispatch_latlons and not hinfo_df.empty and {"Grade Code","Hospital Name"}.issubset(hinfo_df.columns):
            tmp = hinfo_df.copy()
            tmp["Grade Code"] = pd.to_numeric(tmp["Grade Code"], errors="coerce")
            for _, rr in tmp[tmp["Grade Code"]==1].iterrows():
                name = str(rr.get("Hospital Name","Tier3")).strip()
                if name in xl_coord:
                    y, x = xl_coord[name]
                    uav_dispatch_latlons.append((y, x, name, 1))

        uav_out_rows = []
        # Read UAV speed from YAML (reflects speed changes from Rerun tab)
        _, uav_velocity_from_yaml = get_speed_from_yaml(yaml_path)
        uav_velocity = uav_velocity_from_yaml if uav_velocity_from_yaml else 80  # default 80 km/h

        for i, (y, x, nm, code) in enumerate(uav_dispatch_latlons):
            dkm = _haversine_km(y, x, lat, lon)  # straight-line distance
            duration_min = (dkm / uav_velocity) * 60  # time(min) = distance / speed * 60
            uav_out_rows.append({
                "Index": i,
                "Hospital": nm,
                "Grade Code": code,
                "Hospital Grade": code_to_grade(code),
                "Distance(km)": round(dkm, 2),
                "Duration(min)": round(duration_min, 1)
            })
        uav_out_df = pd.DataFrame(uav_out_rows)

        if "uav_c2s_sel_idx" not in st.session_state:
            st.session_state.uav_c2s_sel_idx = set(uav_out_df["Index"].tolist())

        f1, f2 = st.columns(2)
        if f1.button("UAV Dispatch Select All"):
            st.session_state.uav_c2s_sel_idx = set(uav_out_df["Index"].tolist())
        if f2.button("UAV Dispatch Deselect All"):
            st.session_state.uav_c2s_sel_idx = set()

        uav_out_df_show = uav_out_df.copy()
        uav_out_df_show["Show"] = uav_out_df_show["Index"].apply(lambda i: i in st.session_state.uav_c2s_sel_idx)
        # Move 'Show' to front, distance and duration to end
        uav_out_df_show = uav_out_df_show[["Show","Index","Hospital","Grade Code","Hospital Grade","Distance(km)","Duration(min)"]]

        edited_uav_out = st.data_editor(
            uav_out_df_show,
            width='stretch',
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Show":     st.column_config.CheckboxColumn("Show"),
                "Index":   st.column_config.NumberColumn("Index", disabled=True),
                "Hospital":     st.column_config.TextColumn("Hospital", disabled=True),
                "Grade Code": st.column_config.NumberColumn("Grade Code", disabled=True),
                "Hospital Grade": st.column_config.TextColumn("Hospital Grade", disabled=True),
                "Distance(km)": st.column_config.NumberColumn("Distance(km)", disabled=True, format="%.2f"),
                "Duration(min)": st.column_config.NumberColumn("Duration(min)", disabled=True, format="%.1f"),
            },
            key="tbl_uav_out"
        )
        # Synchronize dispatch and transport selection
        selected_indices = set(edited_uav_out.loc[edited_uav_out["Show"]==True, "Index"].tolist())
        st.session_state.uav_c2s_sel_idx = selected_indices
        st.session_state.uav_s2h_sel_idx = selected_indices

    # Transport (Incident -> Hospital): same helipad hospitals as dispatch (round-trip shuttle)
    with col_uav_back:
        st.markdown("**Incident Site → Helipad Hospital (Transport)**")

        # UAV operates as round-trip shuttle: transport to same helipad hospitals as dispatch
        # Use same data as uav_out_df
        uav_back_df = uav_out_df.copy()

        # Share session state (dispatch and transport have same selection state)
        if "uav_s2h_sel_idx" not in st.session_state:
            st.session_state.uav_s2h_sel_idx = st.session_state.uav_c2s_sel_idx.copy()

        g1, g2 = st.columns(2)
        if g1.button("UAV Transport Select All"):
            st.session_state.uav_s2h_sel_idx = set(uav_back_df["Index"].tolist())
            st.session_state.uav_c2s_sel_idx = set(uav_back_df["Index"].tolist())
        if g2.button("UAV Transport Deselect All"):
            st.session_state.uav_s2h_sel_idx = set()
            st.session_state.uav_c2s_sel_idx = set()

        uav_back_df_show = uav_back_df.copy()
        uav_back_df_show["Show"] = uav_back_df_show["Index"].apply(lambda i: i in st.session_state.uav_s2h_sel_idx)
        # Move 'Show' to front, distance and duration to end
        uav_back_df_show = uav_back_df_show[["Show","Index","Hospital","Grade Code","Hospital Grade","Distance(km)","Duration(min)"]]

        edited_uav_back = st.data_editor(
            uav_back_df_show,
            width='stretch',
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Show":     st.column_config.CheckboxColumn("Show"),
                "Index":   st.column_config.NumberColumn("Index", disabled=True),
                "Hospital":     st.column_config.TextColumn("Hospital", disabled=True),
                "Grade Code": st.column_config.NumberColumn("Grade Code", disabled=True),
                "Hospital Grade": st.column_config.TextColumn("Hospital Grade", disabled=True),
                "Distance(km)": st.column_config.NumberColumn("Distance(km)", disabled=True, format="%.2f"),
                "Duration(min)": st.column_config.NumberColumn("Duration(min)", disabled=True, format="%.1f"),
            },
            key="tbl_uav_back"
        )
        # Synchronize dispatch and transport selection
        selected_indices = set(edited_uav_back.loc[edited_uav_back["Show"]==True, "Index"].tolist())
        st.session_state.uav_s2h_sel_idx = selected_indices
        st.session_state.uav_c2s_sel_idx = selected_indices

    # ─────────────────────────────────────────────────────────────────────
    # Map creation and output (displayed in top-level map_holder)
    # ─────────────────────────────────────────────────────────────────────
    m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles=tile_name)

    # Incident site marker
    site_popup = f"Incident Site<br>lat,lon={lat:.6f},{lon:.6f}"
    if site_addr: site_popup += f"<br>Address: {site_addr}"
    folium.Marker(
        [lat,lon],
        icon=folium.Icon(color="purple", icon="map-pin", prefix="fa"),
        tooltip="Incident Site", popup=site_popup
    ).add_to(m)

    # -- AMB C->S lines/markers --
    # Name -> JSON route mapping (connected by center name)
    c2s_map = {}
    for obj in c2s_all:  # scan all
        meta = obj.get("meta", {})
        nm = str(meta.get("name","")).strip()
        if nm:
            c2s_map[nm] = obj

    # Draw based on amb_info_road order/selection
    for _, row in c2s_df.iterrows():
        i = int(row["Index"])
        if i not in st.session_state.amb_c2s_sel_idx:
            continue
        cname = str(row["Fire Station"]).strip()
        obj   = c2s_map.get(cname)
        if obj is None:
            continue

        meta = obj.get("meta", {})
        c    = meta.get("center") or meta.get("start")  # [lon, lat]
        clatlon = (c[1], c[0]) if (isinstance(c, list) and len(c)==2) else None

        addr = tel = ""
        if not center_df.empty and "station_name" in center_df.columns and clatlon:  # station_name = institution name column
            # Coordinate-based matching: select nearest center when multiple share the same name
            msk = (center_df["station_name"].astype(str) == cname)
            if msk.any():
                candidates = center_df[msk].copy()
                # Distance calculation (simple Euclidean, not Haversine)
                if "y_coord" in candidates.columns and "x_coord" in candidates.columns:
                    candidates["_dist"] = ((candidates["y_coord"] - clatlon[0])**2 +
                                          (candidates["x_coord"] - clatlon[1])**2)**0.5
                    rowc = candidates.sort_values("_dist").iloc[0]
                else:
                    rowc = candidates.iloc[0]
                addr = str(rowc.get("address","")); tel = str(rowc.get("phone",""))

        # Distance uses amb_info_road's init_distance
        # (existing) extra construction replaced as follows
        dist_csv = row["Distance(km)"] if "Distance(km)" in row and pd.notna(row["Distance(km)"]) else None
        dist_json, dur_min, _ = _extract_summary_meta(obj)  # JSON route summary

        extra = []
        if addr: extra.append(f"Address: {addr}")       # address
        if tel:  extra.append(f"Phone: {tel}")        # phone
        qty = row.get("Fleet Size", None)
        if qty is not None and pd.notna(qty):
            extra.append(f"Fleet: {int(qty)}")
        # Distance (prefer CSV, fallback to JSON)
        dk = float(dist_csv) if dist_csv is not None else (float(dist_json) if dist_json is not None else None)
        if dk is not None:
            extra.append(f"🚑 Center→Site: {dk:.2f} km")
        # Duration
        if dur_min is not None and dur_min > 0:
            extra.append(f"Duration: {dur_min:.1f} min")

        if clatlon:
            add_center_marker(m, cname, clatlon, extra)


        # Actual line is from JSON route
        draw_route_from_json(m, obj, highlight=False)

    # -- AMB S->H lines/markers --
    # Name/Index -> JSON route mapping (connected by hospital name)
    h2s_map = {}
    h2s_idx_map = {}
    for obj in h2s_all:  # scan all
        meta = obj.get("meta", {})
        nm = str(meta.get("name","")).strip()
        if nm:
            h2s_map[nm] = obj
        try:
            idx = int(meta.get("source_index"))
            h2s_idx_map[idx] = obj
        except Exception:
            pass

    # hospital_info_road order/selection + distance_Hos2Site_road distance display
    for _, row in s2h_df.iterrows():
        i = int(row["Index"])
        if i not in st.session_state.amb_s2h_sel_idx:
            continue
        name = str(row["Hospital"]).strip()
        euc_idx = int(row.get("euc_idx", i))
        obj = h2s_idx_map.get(euc_idx) or h2s_map.get(name)
        if obj is None:
            continue

        # Coordinates/metadata supplemented from Excel
        latlon = None; phone = addr = None
        code   = row.get("Grade Code", None)
        # 1) Route meta coordinates (Index preferred when names duplicate)
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        hosp_meta = meta.get("hospital")
        if isinstance(hosp_meta, (list, tuple)) and len(hosp_meta) == 2:
            latlon = (float(hosp_meta[1]), float(hosp_meta[0]))

        # 2) Excel: supplement address/phone, coordinates only when missing
        if hosp_xl is not None and "institution_name" in hosp_xl.columns:
            rx = hosp_xl[hosp_xl["institution_name"] == name]
            if not rx.empty:
                phone = rx.iloc[0].get("phone", None)
                addr  = rx.iloc[0].get("address", None)
                if latlon is None:
                    y = rx.iloc[0].get("y_coord", rx.iloc[0].get("y", None))
                    x = rx.iloc[0].get("x_coord", rx.iloc[0].get("x", None))
                    if pd.notna(y) and pd.notna(x):
                        latlon = (float(y), float(x))

            # (existing) Keep coordinate/meta retrieval as is,
            #       additionally read beds/ORs from hinfo_df, duration from JSON

            # Distance: prefer table distance(km), fallback to JSON summary distance
            dkm_csv = row.get("Distance(km)", None)
            dkm_json, dur_min, _ = _extract_summary_meta(obj)

            dk = float(dkm_csv) if pd.notna(dkm_csv) else (float(dkm_json) if dkm_json is not None else None)

            # Beds/ORs: look up from original hospital_info_road.csv by index i
            beds_val = None
            ops_val  = None
            orig = hinfo_df[hinfo_df["Index"] == i]
            if not orig.empty:
                beds_val = orig.iloc[0].get("Beds", None)
                ops_val  = orig.iloc[0].get("ORs", None)

            grade_label = code_to_grade(code)
            extras = [f"Grade: {grade_label}"]                 # hospital grade
            if dk is not None:
                extras.append(f"🏥 Site→Hospital: {dk:.2f} km")   # distance
            if dur_min is not None and dur_min > 0:
                extras.append(f"Duration: {dur_min:.1f} min")      # duration

            if latlon:
                add_hospital_marker(
                    m, name, code, latlon, ops_val, beds_val, extras
                )

        # Line is from JSON route
        draw_route_from_json(m, obj, highlight=False)

    # ─ UAV Dispatch(Hosp→Site) ─
    for i, (y, x, name, code) in enumerate(uav_dispatch_latlons):
        if i not in st.session_state.uav_c2s_sel_idx:
            continue
        dkm = _haversine_km(y, x, lat, lon)

        # Add helipad hospital marker
        # Get beds/ORs from uav_out_df
        beds_val = None
        ops_val = None
        # Reuse already-loaded uav_df instead of re-reading CSV
        if 'uav_df' in dir() and uav_df is not None and i < len(uav_df):
            if "Beds" in uav_df.columns:
                beds_val = uav_df.iloc[i].get("Beds", None)
            if "ORs" in uav_df.columns:
                ops_val = uav_df.iloc[i].get("ORs", None)

        grade_label = code_to_grade(code)
        extras = [
            f"Grade: {grade_label}",
            f"🛩️ Helipad→Site: {dkm:.2f} km"
        ]

        add_hospital_marker(m, name, code, (y, x), ops_val, beds_val, extras)

        # Draw dispatch route
        draw_uav_dash(
            m, (y,x), (lat,lon),
            UAV_OUT_COLOR,
            f"🛩️ Dispatch {name}→Site · {dkm:.2f} km"
        )

    # -- UAV Transport (Incident -> Helipad Hospital) --
    # UAV operates as round-trip shuttle: departs from helipad hospital and returns to same hospital
    # Uses same hospital list as uav_dispatch_latlons
    for i, (y, x, name, code) in enumerate(uav_dispatch_latlons):
        if i not in st.session_state.uav_s2h_sel_idx:
            continue
        dkm = _haversine_km(lat, lon, y, x)
        draw_uav_dash(
            m, (lat,lon), (y,x),
            UAV_BACK_COLOR,
            f"🛩️ Transport Site→{name} · {dkm:.2f} km"
        )

    # -- Legend/speed --
    amb_speed, uav_speed = get_speed_from_yaml(find_yaml_in_coord(bp, exp, coord))
    legend_html = [
        '<div style="position: fixed; bottom: 18px; left: 12px; z-index: 9999; background: rgba(255,255,255,0.96); color: #222; padding: 10px 12px; border-radius: 10px; font-size: 12px; line-height: 1.35; box-shadow: 0 2px 8px rgba(0,0,0,.25); border: 1px solid #ccc;">',
        '<b>Legend (Kakao Traffic)</b><br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[0]};margin:2px 6px 2px 0;"></span>Unknown(0)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[1]};margin:2px 6px 2px 0;"></span>Congested(1)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[2]};margin:2px 6px 2px 0;"></span>Slow(2)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[3]};margin:2px 6px 2px 0;"></span>Moderate(3)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[4]};margin:2px 6px 2px 0;"></span>Clear(4)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:5px solid {CONG_COLORS[6]};margin:2px 6px 2px 0;"></span>Accident(6)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:3px dashed {UAV_OUT_COLOR};margin:6px 6px 2px 0;"></span>UAV Dispatch(Hosp→Site)<br>',
        f'<span style="display:inline-block;width:26px;height:0;border-top:3px dashed {UAV_BACK_COLOR};margin:2px 6px 0 0;"></span>UAV Transport(Site→Hosp)<br>',
    ]
    if amb_speed or uav_speed:
        sp = []
        if amb_speed: sp.append(f"🚑 AMB≈{amb_speed} km/h")
        if uav_speed: sp.append(f"🛩️ UAV≈{uav_speed} km/h")
        legend_html.append(" · ".join(sp) + '<br>')
    legend_html.append('<span style="opacity:.7; color:#555;">* Click marker popup for route details</span>')
    legend_html.append('</div>')
    m.get_root().html.add_child(folium.Element("".join(legend_html)))

    # Minimize attribution display (optional)
    from folium import Element
    m.get_root().html.add_child(Element("""
    <style>
    .leaflet-control-attribution {
      font-size: 1px !important;
      opacity: .55 !important;
      background: rgba(255,255,255,.6) !important;
      padding: 2px 6px !important;
      border-radius: 1px !important;
    }
    .leaflet-control-attribution a { color: inherit !important; text-decoration: none !important; }
    .leaflet-bottom.leaflet-right { bottom: auto !important; top: 6px !important; right: 8px !important; }
    </style>
    """))

    # Output map to 'top-level' container
    with map_holder:
        if _map_mode == "Static Map":
            st_folium(m, width=None, height=690, key="main_map", returned_objects=[])
        else:
            # ─────────────────────────────────────────────────────────────
            # Animation Mode — Folium map + Leaflet JS animation overlay
            # ─────────────────────────────────────────────────────────────
            _anim_log_cands_raw = experiment_log_candidates(bp, exp, coord)
            # Filter: only simulation logs (first line starts with "=== SIM_START")
            _anim_log_cands = []
            for _lp in _anim_log_cands_raw:
                try:
                    with open(_lp, "r", encoding="utf-8-sig") as _lf:
                        _first = _lf.readline().strip()
                    if _first.startswith("=== SIM_START"):
                        _anim_log_cands.append(_lp)
                except Exception:
                    pass
            if not _anim_log_cands:
                st.info("No simulation logs found for this coordinate.")
            else:
                _anim_log_sel = st.selectbox("Log file", [Path(p).name for p in _anim_log_cands], key="anim_log_sel")
                _anim_log_path = _anim_log_cands[[Path(p).name for p in _anim_log_cands].index(_anim_log_sel)]

                try:
                    _anim_blocks = parse_log_blocks(open(_anim_log_path, "r", encoding="utf-8-sig").read())
                    # Preserve log order for rules; exclude (Unlabeled) -- env init artifact
                    _anim_rules = list(dict.fromkeys(
                        b["rule"] for b in _anim_blocks
                        if b.get("rule") and b["rule"] != "(Unlabeled)"
                    ))
                    if not _anim_rules:
                        st.warning("No rules found in log.")
                    else:
                        _anim_rule_sel = st.selectbox("Rule", _anim_rules, key="anim_rule_sel2")
                        _anim_rule_blks = [b for b in _anim_blocks if b.get("rule") == _anim_rule_sel]
                        _anim_iters = sorted({b.get("iter") for b in _anim_rule_blks if b.get("iter") is not None})
                        _anim_iter_sel = None
                        if _anim_iters:
                            _anim_iter_sel = st.selectbox("Iteration", _anim_iters, key="anim_iter_sel2")
                            _anim_cand = [b for b in _anim_rule_blks if b.get("iter") == _anim_iter_sel]
                        else:
                            _anim_cand = _anim_rule_blks
                        _anim_blk = _anim_cand[0] if _anim_cand else None
                        if not _anim_blk or not _anim_blk.get("events"):
                            st.info("No events in selected block.")
                        else:
                            _log_ev = sorted(_anim_blk["events"], key=lambda r: r["t"])
                            _incident_ll = (lat, lon)

                            # ── Coordinate lookups ──
                            _hosp_coords = {}
                            if not hinfo_df.empty:
                                for _, _hr in hinfo_df.iterrows():
                                    _hi = int(_hr["Index"])
                                    _hn = str(_hr.get("Hospital Name", "")).strip()
                                    if _hn in xl_coord:
                                        _hosp_coords[_hi] = (*xl_coord[_hn], _hn)

                            _amb2station = {}
                            _station_info = {}
                            _amb_cursor = 0
                            if not ambinfo_df.empty:
                                for _, _sr in ambinfo_df.iterrows():
                                    _si = int(_sr["Index"])
                                    _sn = str(_sr.get("Fire Station", "")).strip()
                                    _fleet = int(_sr.get("Fleet Size", 1)) if "Fleet Size" in _sr.index and pd.notna(_sr.get("Fleet Size")) else 1
                                    for _robj in c2s_all:
                                        _rm = _robj.get("meta", {})
                                        if str(_rm.get("name", "")).strip() == _sn:
                                            _sc = _rm.get("center") or _rm.get("start")
                                            if isinstance(_sc, (list, tuple)) and len(_sc) == 2:
                                                _station_info[_si] = (float(_sc[1]), float(_sc[0]), _sn)
                                            break
                                    for _fi in range(_fleet):
                                        _amb2station[_amb_cursor] = _si
                                        _amb_cursor += 1

                            _uav_base = {}
                            for _ui, (_uy, _ux, _un, _uc) in enumerate(uav_dispatch_latlons):
                                _uav_base[_ui] = (_uy, _ux, _un)

                            _c2s_poly = {}
                            for _robj in c2s_all:
                                _rm = _robj.get("meta", {})
                                _rn = str(_rm.get("name", "")).strip()
                                if _rn:
                                    _pl = extract_polyline(_robj)
                                    if _pl:
                                        _c2s_poly[_rn] = _pl

                            _s2h_poly = {}
                            _h2s_poly = {}
                            for _robj in h2s_all:
                                _rm = _robj.get("meta", {})
                                try:
                                    _ri = int(_rm.get("source_index"))
                                    _pl = extract_polyline(_robj)
                                    if _pl:
                                        _s2h_poly[_ri] = _pl
                                        _h2s_poly[_ri] = list(reversed(_pl))
                                except (TypeError, ValueError):
                                    pass

                            # ── Parse log events ──
                            _amb_site_t = {}
                            _uav_site_t = {}
                            _p_rescue_t = {}
                            _p_care_ready_t = {}
                            _p_care_done_t = {}
                            _transport_arrivals = []

                            for _ev in _log_ev:
                                _evn = _ev.get("ev", "")
                                _t = _ev["t"]
                                if _evn == "amb_arrival_site":
                                    _a = _ev.get("a")
                                    if _a is not None:
                                        _amb_site_t.setdefault(_a, []).append(_t)
                                elif _evn == "uav_arrival_site":
                                    _u = _ev.get("u")
                                    if _u is not None:
                                        _uav_site_t.setdefault(_u, []).append(_t)
                                elif _evn == "p_rescue":
                                    _p = _ev.get("p")
                                    if _p is not None:
                                        _p_rescue_t.setdefault(_p, _t)
                                elif _evn == "amb_arrival_hospital":
                                    _p, _a, _h = _ev.get("p"), _ev.get("a"), _ev.get("h")
                                    if _p is not None:
                                        _transport_arrivals.append({"p": _p, "veh": "AMB", "vid": _a, "h": _h, "t": _t})
                                elif _evn == "uav_arrival_hospital":
                                    _p, _u, _h = _ev.get("p"), _ev.get("u"), _ev.get("h")
                                    if _p is not None:
                                        _transport_arrivals.append({"p": _p, "veh": "UAV", "vid": _u, "h": _h, "t": _t})
                                elif _evn == "p_care_ready":
                                    _p = _ev.get("p")
                                    if _p is not None:
                                        _p_care_ready_t.setdefault(_p, _t)
                                elif _evn == "p_def_care":
                                    _p = _ev.get("p")
                                    if _p is not None:
                                        _p_care_done_t.setdefault(_p, _t)

                            # ── Build movement segments ──
                            _segments = []
                            for _a_idx, _t_list in _amb_site_t.items():
                                _t_arr = _t_list[0]
                                _st_idx = _amb2station.get(_a_idx)
                                if _st_idx is not None and _st_idx in _station_info:
                                    _sname = _station_info[_st_idx][2]
                                    _poly = _c2s_poly.get(_sname, [(_station_info[_st_idx][0], _station_info[_st_idx][1]), _incident_ll])
                                    _segments.append({"veh": "AMB", "vid": _a_idx, "phase": "dispatch",
                                                      "poly": _poly, "t0": 0.0, "t1": _t_arr, "label": f"AMB-{_a_idx}"})
                            for _u_idx, _t_list in _uav_site_t.items():
                                if _u_idx in _uav_base:
                                    _ub = _uav_base[_u_idx]
                                    _segments.append({"veh": "UAV", "vid": _u_idx, "phase": "dispatch",
                                                      "poly": [(_ub[0], _ub[1]), _incident_ll], "t0": 0.0, "t1": _t_list[0],
                                                      "label": f"UAV-{_u_idx}"})

                            _amb_avail = {a: list(tl) for a, tl in _amb_site_t.items()}
                            _uav_avail = {u: list(tl) for u, tl in _uav_site_t.items()}
                            for _ta in sorted(_transport_arrivals, key=lambda x: x["t"]):
                                _p, _veh, _vid, _hid, _t_hosp = _ta["p"], _ta["veh"], _ta["vid"], _ta["h"], _ta["t"]
                                _h_name = _hosp_coords[_hid][2] if _hid in _hosp_coords else f"H{_hid}"
                                _t_rescue = _p_rescue_t.get(_p, 0)
                                if _veh == "AMB":
                                    _st = _amb_avail.get(_vid, [0])
                                    _t_dep = max((s for s in _st if s <= _t_hosp), default=0)
                                    _t_dep = max(_t_dep, _t_rescue)
                                    _poly = _s2h_poly.get(_hid, [_incident_ll, (_hosp_coords[_hid][0], _hosp_coords[_hid][1])] if _hid in _hosp_coords else [])
                                    if _poly:
                                        _segments.append({"veh": "AMB", "vid": _vid, "phase": "transport",
                                                          "poly": _poly, "t0": _t_dep, "t1": _t_hosp,
                                                          "label": f"AMB-{_vid} P{_p}", "pid": _p})
                                    _nxt = [s for s in _st if s > _t_hosp]
                                    if _nxt:
                                        _rpoly = _h2s_poly.get(_hid, [(_hosp_coords[_hid][0], _hosp_coords[_hid][1]), _incident_ll] if _hid in _hosp_coords else [])
                                        if _rpoly:
                                            _segments.append({"veh": "AMB", "vid": _vid, "phase": "return",
                                                              "poly": _rpoly, "t0": _t_hosp, "t1": _nxt[0], "label": f"AMB-{_vid}"})
                                else:
                                    _st = _uav_avail.get(_vid, [0])
                                    _t_dep = max((s for s in _st if s <= _t_hosp), default=0)
                                    _t_dep = max(_t_dep, _t_rescue)
                                    _dest = (_hosp_coords[_hid][0], _hosp_coords[_hid][1]) if _hid in _hosp_coords else (_uav_base[_vid][0], _uav_base[_vid][1]) if _vid in _uav_base else None
                                    if _dest:
                                        _segments.append({"veh": "UAV", "vid": _vid, "phase": "transport",
                                                          "poly": [_incident_ll, _dest], "t0": _t_dep, "t1": _t_hosp,
                                                          "label": f"UAV-{_vid} P{_p}", "pid": _p})
                                        _nxt = [s for s in _st if s > _t_hosp]
                                        if _nxt:
                                            _segments.append({"veh": "UAV", "vid": _vid, "phase": "return",
                                                              "poly": [_dest, _incident_ll], "t0": _t_hosp, "t1": _nxt[0],
                                                              "label": f"UAV-{_vid}"})

                            if not _segments:
                                st.warning("No vehicle movements found.")
                            else:
                                # ── Pre-compute frame-by-frame positions ──
                                _all_patients = sorted(_p_rescue_t.keys())
                                _patient_hospital = {}
                                for _ta in _transport_arrivals:
                                    _patient_hospital.setdefault(_ta["p"], _ta["h"])

                                _t_max_seg = max(s["t1"] for s in _segments)
                                _t_max_care = max(_p_care_done_t.values()) if _p_care_done_t else _t_max_seg
                                _t_max = max(_t_max_seg, _t_max_care)
                                _n_frames = st.slider("Animation frames", 30, 300, 100, key="anim_frames_sl")
                                _time_steps = np.linspace(0, _t_max, _n_frames).tolist()

                                # Collect unique vehicles
                                _veh_ids = list(dict.fromkeys((s["veh"], s["vid"]) for s in _segments))
                                _veh_data = []
                                for (_vtype, _vid) in _veh_ids:
                                    _emoji = "\U0001F691" if _vtype == "AMB" else "\U0001F681"
                                    _positions = []
                                    for _ti, _t in enumerate(_time_steps):
                                        _pos = None
                                        _carrying = False
                                        _phase = ""
                                        for _seg in _segments:
                                            if _seg["veh"] == _vtype and _seg["vid"] == _vid and _seg["t0"] <= _t <= _seg["t1"]:
                                                _frac = (_t - _seg["t0"]) / max(0.001, _seg["t1"] - _seg["t0"])
                                                _pp = _interpolate_along_polyline(_seg["poly"], _frac)
                                                _pos = [_pp[0], _pp[1]]
                                                _carrying = _seg.get("phase") == "transport"
                                                _phase = _seg.get("phase", "")
                                                break
                                        # f: flip flag -- compare with previous position to determine heading
                                        _flip = False
                                        if _pos and _ti > 0 and _positions[_ti - 1] is not None:
                                            _prev_lon = _positions[_ti - 1]["c"][1]
                                            _flip = _pos[1] < _prev_lon  # moving west -> flip
                                        _positions.append({"c": _pos, "k": _carrying, "f": _flip} if _pos else None)
                                    _veh_data.append({"id": f"{_vtype}-{_vid}", "emoji": _emoji, "pos": _positions})

                                # Patient positions per frame
                                _pat_data = []
                                for _pid in _all_patients:
                                    _positions = []
                                    _rt = _p_rescue_t.get(_pid)
                                    _p_h = _patient_hospital.get(_pid)
                                    _t_hosp_arr = None
                                    for _ta in _transport_arrivals:
                                        if _ta["p"] == _pid:
                                            _t_hosp_arr = _ta["t"]
                                            break
                                    _t_done = _p_care_done_t.get(_pid)

                                    for _t in _time_steps:
                                        if _rt is None or _t < _rt:
                                            _positions.append(None)
                                        elif _t_done is not None and _t >= _t_done and _p_h in _hosp_coords:
                                            _hc = _hosp_coords[_p_h]
                                            _positions.append({"la": _hc[0]+0.001*((_pid%5)-2), "lo": _hc[1]+0.001*((_pid%3)-1), "e": "\u2705"})
                                        elif _t_hosp_arr is not None and _t >= _t_hosp_arr and _p_h in _hosp_coords:
                                            _hc = _hosp_coords[_p_h]
                                            _positions.append({"la": _hc[0]+0.0008*((_pid%5)-2), "lo": _hc[1]+0.0008*((_pid%3)-1), "e": "\U0001F3E5"})
                                        else:
                                            _positions.append({"la": lat+0.0005*((_pid%7)-3), "lo": lon+0.0005*((_pid%5)-2), "e": "\U0001F6D1"})
                                    _pat_data.append({"id": f"P{_pid}", "pos": _positions})

                                # Build per-patient detail info for popup
                                _pat_info = {}
                                for _pid in _all_patients:
                                    _info = {"id": _pid}
                                    # Transport vehicle
                                    for _ta in _transport_arrivals:
                                        if _ta["p"] == _pid:
                                            _info["veh"] = f"{_ta['veh']}-{_ta['vid']}"
                                            _info["t_hosp"] = round(_ta["t"], 2)
                                            _h_id = _ta["h"]
                                            _info["hosp"] = _hosp_coords[_h_id][2] if _h_id in _hosp_coords else f"H{_h_id}"
                                            break
                                    _info["t_rescue"] = round(_p_rescue_t.get(_pid, 0), 2)
                                    _t_care_ready = _p_care_ready_t.get(_pid)
                                    _t_done = _p_care_done_t.get(_pid)
                                    _t_hosp_a = _info.get("t_hosp")
                                    # ER wait = care_ready - hospital_arrival (handover period)
                                    # Treatment = def_care - care_ready
                                    if _t_care_ready is not None and _t_hosp_a is not None:
                                        _info["t_wait"] = round(_t_care_ready - _t_hosp_a, 2)
                                    if _t_done is not None and _t_care_ready is not None:
                                        _info["t_treat"] = round(_t_done - _t_care_ready, 2)
                                    if _t_done is not None and _t_hosp_a is not None:
                                        _info["t_stay"] = round(_t_done - _t_hosp_a, 2)
                                    _info["t_done"] = round(_t_done, 2) if _t_done else None
                                    _pat_info[f"P{_pid}"] = _info

                                # Serialize animation data
                                _anim_payload = json.dumps({"ts": _time_steps, "v": _veh_data, "p": _pat_data, "pi": _pat_info}, ensure_ascii=False)
                                _map_var = m.get_name()

                                # Inject Leaflet JS animation into the Folium map HTML
                                # Control bar is placed BELOW the map (outside) to avoid legend overlap
                                _anim_js = f"""
<style>
.anim-icon{{background:none!important;border:none!important;}}
#anim-ctrl-bar{{
  background:rgba(255,255,255,0.97);padding:10px 20px;border-radius:8px;
  box-shadow:0 2px 8px rgba(0,0,0,.2);display:flex;align-items:center;
  gap:12px;font-family:sans-serif;margin:8px auto 0;width:fit-content;
}}
#anim-ctrl-bar button{{
  padding:5px 14px;cursor:pointer;font-size:14px;border:1px solid #bbb;
  border-radius:5px;background:#fff;transition:background .15s;
}}
#anim-ctrl-bar button:hover{{background:#f0f0f0;}}
</style>
<div id="anim-ctrl-bar">
  <button id="acb-play">&#9654; Play</button>
  <button id="acb-replay" style="display:none;">&#x21BA; Replay</button>
  <input id="acb-slider" type="range" min="0" max="1" value="0" style="width:300px;">
  <span id="acb-time" style="font-size:13px;min-width:110px;">t = 0.0 min</span>
</div>
<script>
(function(){{
var POLL=setInterval(function(){{
  if(typeof {_map_var}==='undefined') return;
  clearInterval(POLL);
  var map={_map_var};
  var D={_anim_payload};
  var ts=D.ts, nf=ts.length, cur=0, playing=false, tid=null;
  var pbtn=document.getElementById('acb-play');
  var rbtn=document.getElementById('acb-replay');
  var sl=document.getElementById('acb-slider');
  var tdisp=document.getElementById('acb-time');
  sl.max=nf-1;
  // Vehicle markers
  var vm={{}};
  D.v.forEach(function(v){{
    var ic=L.divIcon({{html:'<span style="font-size:26px;filter:drop-shadow(0 0 2px white);">'+v.emoji+'</span>',className:'anim-icon',iconSize:[32,32],iconAnchor:[16,16]}});
    var mk=L.marker([0,0],{{icon:ic,opacity:0,zIndexOffset:1000}});
    mk.bindTooltip(v.id,{{permanent:false,direction:'top'}});
    mk.addTo(map);
    vm[v.id]={{mk:mk,base:v.emoji}};
  }});
  // Patient markers
  var pm={{}};
  var PI=D.pi||{{}};
  function patPopup(pid){{
    var d=PI[pid];if(!d)return '<b>'+pid+'</b><br>No detail available';
    var h='<div style="font-size:13px;line-height:1.6;min-width:180px;">';
    h+='<b style="font-size:14px;">'+pid+'</b><br>';
    if(d.veh)h+='\U0001F698 Transport: <b>'+d.veh+'</b><br>';
    if(d.hosp)h+='\U0001F3E5 Hospital: <b>'+d.hosp+'</b><br>';
    h+='<hr style="margin:4px 0;border-color:#ddd;">';
    if(d.t_rescue!=null)h+='\u23F1 Rescue: <b>'+d.t_rescue.toFixed(1)+'</b> min<br>';
    if(d.t_hosp!=null)h+='\U0001F3E5 Arrival: <b>'+d.t_hosp.toFixed(1)+'</b> min<br>';
    if(d.t_wait!=null)h+='\u23F3 ER Wait (Handover): <b>'+d.t_wait.toFixed(1)+'</b> min<br>';
    if(d.t_treat!=null)h+='\U0001FA7A Treatment: <b>'+d.t_treat.toFixed(1)+'</b> min<br>';
    if(d.t_stay!=null){{h+='<hr style="margin:4px 0;border-color:#ddd;">';h+='\U0001F4CB Total Stay: <b>'+d.t_stay.toFixed(1)+'</b> min<br>';}}
    if(d.t_done!=null)h+='\u2705 Completed at: <b>'+d.t_done.toFixed(1)+'</b> min';
    h+='</div>';return h;
  }}
  D.p.forEach(function(p){{
    var ic=L.divIcon({{html:'<span style="font-size:14px;">\\u26AA</span>',className:'anim-icon',iconSize:[18,18],iconAnchor:[9,9]}});
    var mk=L.marker([0,0],{{icon:ic,opacity:0,zIndexOffset:500}});
    mk.bindTooltip(p.id,{{permanent:false,direction:'top'}});
    mk.bindPopup(function(){{return patPopup(p.id);}},{{maxWidth:250}});
    mk.addTo(map);
    pm[p.id]=mk;
  }});
  function vehHtml(base,carrying,flip){{
    var sx=flip?'transform:scaleX(-1);':'';
    if(carrying){{
      return '<span style="font-size:26px;display:inline-block;'+sx+'filter:drop-shadow(0 0 3px #ff4444);">'+base+'</span>'
           + '<span style="font-size:13px;position:absolute;top:-6px;'+(flip?'left:-8px;':'right:-8px;')+'">\U0001F9D1\u200D\u2695\uFE0F</span>';
    }}
    return '<span style="font-size:26px;display:inline-block;'+sx+'filter:drop-shadow(0 0 2px white);">'+base+'</span>';
  }}
  function upd(idx){{
    cur=idx;
    D.v.forEach(function(v){{
      var fr=v.pos[idx];
      if(fr){{
        var m2=vm[v.id].mk;
        m2.setLatLng([fr.c[0],fr.c[1]]);m2.setOpacity(1);
        var ni=L.divIcon({{html:vehHtml(vm[v.id].base,fr.k,fr.f),className:'anim-icon',iconSize:[32,32],iconAnchor:[16,16]}});
        m2.setIcon(ni);
      }} else{{vm[v.id].mk.setOpacity(0);}}
    }});
    D.p.forEach(function(p){{
      var pos=p.pos[idx];
      if(pos){{
        pm[p.id].setLatLng([pos.la,pos.lo]);pm[p.id].setOpacity(1);
        var sp=pm[p.id]._icon?pm[p.id]._icon.querySelector('span'):null;
        if(sp)sp.textContent=pos.e;
      }} else{{pm[p.id].setOpacity(0);}}
    }});
    sl.value=idx;
    tdisp.textContent='t = '+ts[idx].toFixed(1)+' min';
  }}
  function play(){{
    if(playing)return;playing=true;
    rbtn.style.display='none';
    pbtn.textContent='\\u23F8 Pause';
    tid=setInterval(function(){{
      if(cur>=nf-1){{pause();rbtn.style.display='inline-block';return;}}
      upd(cur+1);
    }},120);
  }}
  function pause(){{playing=false;pbtn.textContent='\\u25B6 Play';if(tid)clearInterval(tid);}}
  function replay(){{pause();upd(0);rbtn.style.display='none';play();}}
  pbtn.onclick=function(){{playing?pause():play();}};
  rbtn.onclick=replay;
  sl.oninput=function(){{upd(parseInt(this.value));}};
  upd(0);
}},200);
}})();
</script>
"""
                                m.get_root().html.add_child(folium.Element(_anim_js))
                                import streamlit.components.v1 as stc
                                _map_html = m.get_root().render()
                                stc.html(_map_html, height=780, scrolling=False)

                                _n_amb = sum(1 for s in _segments if s["veh"]=="AMB" and s["phase"]=="transport")
                                _n_uav = sum(1 for s in _segments if s["veh"]=="UAV" and s["phase"]=="transport")
                                _n_done = len(_p_care_done_t)
                                st.caption(f"{_n_amb} AMB transports · {_n_uav} UAV transports · "
                                           f"{_n_done}/{len(_all_patients)} patients completed")
                                st.markdown(
                                    "\U0001F691 AMB (empty) · \U0001F691+\U0001F9D1\u200D\u2695\uFE0F AMB (carrying patient) · "
                                    "\U0001F681 UAV · "
                                    "\U0001F6D1 Patient waiting · \U0001F3E5 Treating · \u2705 Done")

                except Exception as _anim_err:
                    st.error(f"Animation error: {_anim_err}")
                    import traceback
                    st.code(traceback.format_exc())


# ------------------------------
# Analytics tab (sortable table + ANOVA suite)
# ------------------------------

@st.cache_data
def gen_scenario_keys() -> pd.DataFrame:
    rows = []
    for ph in PHASES:
        for rp in RED_POLICY:
            for ra in ACTIONS:
                for ya in ACTIONS:
                    rows.append((ph, rp, ra, ya))
    df = pd.DataFrame(rows, columns=["Phase","RedPolicy","RedAction","YellowAction"])
    df["ScenarioIdx"] = np.arange(len(df))
    return df


def parse_stat_file(stat_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reorder STAT file based on RAW order.
    Also handles missing scenarios (UAV=0 case).
    """
    
    # 1. Read STAT file - use rule name as key
    stat_dict = {}  # {rule_name: [(mean, std, ci) for each metric]}
    
    with open(stat_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    st.caption(f"📊 STAT file: {len(lines)} lines total")
    
    # Check expected line count
    expected = 320  # 64 × 5
    if len(lines) != expected:
        st.warning(f"⚠️ STAT line count mismatch: {len(lines)} (expected: {expected})")
        st.caption(f"→ {expected - len(lines)} scenarios may be missing")
    
    # 2. Parse each line and store in dictionary
    for line in lines:
        parts = line.split()
        
        if len(parts) < 3:
            continue
        
        try:
            mean = float(parts[-3])
            std = float(parts[-2])
            ci = float(parts[-1])
            
            # Extract rule name
            rule_name = " ".join(parts[:-3]).strip()
            
            # Store metrics per rule (5 in order: Reward, Time, PDR, Reward_woG, PDR_woG)
            if rule_name not in stat_dict:
                stat_dict[rule_name] = []
            
            stat_dict[rule_name].append((mean, std, ci))
            
        except ValueError:
            continue
    
    # 3. Get exact rule order from RAW file
    raw_path = stat_path.replace("_stat.txt", ".txt")
    
    if not os.path.exists(raw_path):
        st.error("❌ RAW file not found!")
        return pd.DataFrame(), pd.DataFrame()
    
    dfraw = parse_raw_results(raw_path)
    reward_data = dfraw[dfraw["metric"] == "Reward"]
    
    # Rule order from RAW (actual execution order)
    rule_order = reward_data[reward_data["run"] == 1]["rule"].tolist()
    
    st.info(f"✅ RAW file: {len(rule_order)} scenarios confirmed")
    
    # 4. Match STAT data in RAW order
    result_rows = []
    missing_count = 0
    
    for idx, rule in enumerate(rule_order):
        row = {
            "ScenarioIdx": idx,
        }
        
        # Parse Phase, RedPolicy, etc.
        match = re.match(
            r'(START|ReSTART),\s*(RedOnly|YellowNearest),\s*Red\s+(\w+),\s*Yellow\s+(\w+)', 
            rule
        )
        if match:
            row["Phase"] = match.group(1)
            row["RedPolicy"] = match.group(2)
            row["RedAction"] = match.group(3)
            row["YellowAction"] = match.group(4)
        
        # Find the rule in STAT
        if rule in stat_dict:
            stats = stat_dict[rule]
            
            # Keep existing column name format: M1_mean, M2_mean, ...
            for m_idx in range(5):
                if m_idx < len(stats):
                    mean, std, ci = stats[m_idx]
                    row[f"M{m_idx+1}_mean"] = mean
                    row[f"M{m_idx+1}_std"] = std
                    row[f"M{m_idx+1}_ci"] = ci
                else:
                    row[f"M{m_idx+1}_mean"] = np.nan
                    row[f"M{m_idx+1}_std"] = np.nan
                    row[f"M{m_idx+1}_ci"] = np.nan
        else:
            # Rule not in STAT -> compute directly from RAW
            st.warning(f"⚠️ Not in STAT: {rule}")
            missing_count += 1
            
            # Compute statistics directly from RAW data
            metric_names = ["Reward", "Time", "PDR", "Reward_woG", "PDR_woG"]
            
            for m_idx, metric in enumerate(metric_names):
                metric_data = dfraw[(dfraw["metric"] == metric) & (dfraw["rule"] == rule)]
                
                if not metric_data.empty:
                    values = metric_data["value"].values
                    mean = np.mean(values)
                    std = np.std(values, ddof=1) if len(values) > 1 else 0
                    
                    # Calculate 95% CI
                    if len(values) > 1:
                        from scipy.stats import t
                        se = std / np.sqrt(len(values))
                        ci = t.interval(0.95, len(values)-1, loc=mean, scale=se)
                        ci_half = (ci[1] - ci[0]) / 2
                    else:
                        ci_half = 0
                    
                    row[f"M{m_idx+1}_mean"] = mean
                    row[f"M{m_idx+1}_std"] = std
                    row[f"M{m_idx+1}_ci"] = ci_half
                else:
                    row[f"M{m_idx+1}_mean"] = np.nan
                    row[f"M{m_idx+1}_std"] = np.nan
                    row[f"M{m_idx+1}_ci"] = np.nan
        
        result_rows.append(row)
    
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count} scenarios computed directly from RAW")
    
    wide = pd.DataFrame(result_rows)
    

    
    # Also generate long format (for backward compatibility)
    long_rows = []
    metric_names = ["Reward", "Time", "PDR", "Reward_woG", "PDR_woG"]
    
    for _, row in wide.iterrows():
        for m_idx, metric in enumerate(metric_names):
            long_rows.append({
                "ScenarioIdx": row["ScenarioIdx"],
                "Metric": f"M{m_idx+1}",
                "mean": row.get(f"M{m_idx+1}_mean", np.nan),
                "std": row.get(f"M{m_idx+1}_std", np.nan),
                "ci": row.get(f"M{m_idx+1}_ci", np.nan),
                "Phase": row.get("Phase", ""),
                "RedPolicy": row.get("RedPolicy", ""),
                "RedAction": row.get("RedAction", ""),
                "YellowAction": row.get("YellowAction", ""),
            })
    
    long_df = pd.DataFrame(long_rows)
    
    return wide, long_df

# Raw result parsing (Rule x Sample stack)
RAW_RE = re.compile(r'^(START|ReSTART),\s*(RedOnly|YellowNearest),\s*Red\s+([A-Za-z_]+),\s*Yellow\s+([A-Za-z_]+)')



# ------------------------------
# Analytics tab (ANOVA analysis)
# ------------------------------

# ===== Analysis tab =====
with tabs[2]:
    st.subheader("RAW Result Analysis (results_{coord}.txt)")

    bp   = st.session_state.base_path
    exp  = st.session_state.selected_exp
    coord= st.session_state.selected_coord

    # Reset Analytics load state when scenario changes
    current_analytics_key = f"{exp}_{coord}"
    if st.session_state.get("last_analytics_key") != current_analytics_key:
        st.session_state.analytics_loaded = False
        st.session_state.last_analytics_key = current_analytics_key

    if not (bp and exp and coord):
        st.info("Select a scenario from the sidebar first.")
    else:
        spath = results_stat_path(bp, exp, coord)   # existing function
        rpath = results_raw_path(bp, exp, coord)    # existing function

        # Performance optimization: load Analytics data only on button click
        st.info("💡 Large simulation results may take time to load. Click the button below to start analysis.")

        if st.button("Load Analysis Data", key="load_analytics_btn", help="Parse and analyze RAW results"):
            st.session_state.analytics_loaded = True

        if not st.session_state.get("analytics_loaded", False):
            st.caption("💡 Click 'Load Analysis Data' above to view analysis. (Disabled by default for faster tab loading)")
        else:
            analytics_tabs = st.tabs(["RAW Data", "STAT Summary", "ANOVA Suite",
                                       "Pareto Dominance", "Bootstrap / Non-Parametric",
                                       "Power Analysis", "Export"])

            # -- RAW Data sub-tab --
            with analytics_tabs[0]:
                raw_tables = {}
                if rpath and os.path.exists(rpath):
                    with st.spinner("Parsing RAW results... (large files may take a moment)"):
                        raw_tables = parse_raw_all_metrics(rpath)  # {metric: df}

                    if raw_tables:
                        # Only expose metrics actually present in the file
                        metric_options = [m for m in RAW_METRIC_NAMES if m in raw_tables.keys()]
                        picked = st.multiselect(
                            "Select metrics to display",
                            options=metric_options,
                            default=[metric_options[0]] if metric_options else [],
                            help="Only selected metrics will be shown below."
                        )
                        for m in picked:
                            st.markdown(f"#### RAW Table -- **{m}** (per run)")
                            st.dataframe(raw_tables[m], width='stretch')
                    else:
                        st.warning("No readable blocks found in RAW (results_*.txt).")
                else:
                    st.warning("RAW (results_*.txt) file not found.")

            # -- STAT Summary sub-tab --
            with analytics_tabs[1]:
                st.subheader("STAT Summary Analysis (_stat.txt)")
                st.info(
                    "results/exp_YYYYMMDD_HHMMSS/(lat,lon)/results_{coord}.txt (Raw), results_{coord}_stat.txt (Stat)\n\n"
                    "- **Reward**: Survival probability sum\n- **Time**: Elapsed time\n- **PDR**\n- **w.o.G**: Excluding Green"
                )

            wide, long_df = (pd.DataFrame(), pd.DataFrame())
            if spath and os.path.exists(spath):
                wide, long_df = parse_stat_file(spath)

            if not wide.empty:
                display = wide.rename(columns={
                    "M1_mean":"Reward Mean","M1_std":"Reward Std","M1_ci":"Reward 95%CI",
                    "M2_mean":"Time Mean","M2_std":"Time Std","M2_ci":"Time 95%CI",
                    "M3_mean":"PDR Mean","M3_std":"PDR Std","M3_ci":"PDR 95%CI",
                    "M4_mean":"Reward w.o.G Mean","M4_std":"Reward w.o.G Std","M4_ci":"Reward w.o.G 95%CI",
                    "M5_mean":"PDR w.o.G Mean","M5_std":"PDR w.o.G Std","M5_ci":"PDR w.o.G 95%CI",
                })
                st.dataframe(display, width='stretch')

                st.markdown("#### Scenario Ranking (Sort by)")
                crit = st.selectbox("Sort by", ["Reward (desc)","PDR (asc)","Time (asc)"], index=0)
                if crit == "Reward (desc)":
                    df_sorted = wide.sort_values("M1_mean", ascending=False)
                    cols = ["ScenarioIdx","Phase","RedPolicy","RedAction","YellowAction","M1_mean","M1_ci"]
                elif crit == "PDR (asc)":
                    df_sorted = wide.sort_values("M3_mean", ascending=True)
                    cols = ["ScenarioIdx","Phase","RedPolicy","RedAction","YellowAction","M3_mean","M3_ci"]
                else:
                    df_sorted = wide.sort_values("M2_mean", ascending=True)
                    cols = ["ScenarioIdx","Phase","RedPolicy","RedAction","YellowAction","M2_mean","M2_ci"]
                st.dataframe(df_sorted[cols], width='stretch')
            else:
                st.info("STAT summary file not found or empty.")


            # -- ANOVA Suite sub-tab --
            with analytics_tabs[2]:
              st.markdown("#### ANOVA (One-way / RCBD / Reduced Factorial)")

            import itertools

            def make_total_row(anova_tbl: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
                ss_total = float(((y - y.mean())**2).sum())
                N = len(y)
                total = pd.DataFrame(
                    {"sum_sq":[ss_total], "df":[N-1], "mean_sq":[np.nan], "F":[np.nan], "PR(>F)":[np.nan], "eta_sq":[np.nan], "omega_sq":[np.nan]},
                    index=["Total"]
                )
                out = anova_tbl.copy()
                # eta-squared
                out["eta_sq"] = out["sum_sq"] / ss_total
                # omega-squared (bias-corrected)
                if "Residual" in out.index:
                    ms_res = float(out.loc["Residual", "sum_sq"] / out.loc["Residual", "df"]) if out.loc["Residual", "df"] > 0 else 0.0
                    out["omega_sq"] = (out["sum_sq"] - out["df"] * ms_res) / (ss_total + ms_res)
                    out.loc[out["omega_sq"] < 0, "omega_sq"] = 0.0
                    out.loc["Residual", "omega_sq"] = np.nan
                else:
                    out["omega_sq"] = np.nan
                if "df" in out.columns and "sum_sq" in out.columns:
                    out["mean_sq"] = out["sum_sq"] / out["df"]
                cols = ["sum_sq","df","mean_sq","F","PR(>F)","eta_sq","omega_sq"]
                out = out.reindex(columns=cols)
                return pd.concat([out, total], axis=0)


            def block_adjust(df, yvar, block_col):
                """Block-residualize: y* = y - block mean."""
                df = df.copy()
                df["y_adj"] = df[yvar] - df.groupby(block_col)[yvar].transform("mean")
                return df
            def _means_series(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
                """
                Always return group means as Series (defensive against duplicate column names/version issues).
                - If df[value_col] returns a DataFrame, use only the first column
                - Values are coerced to numeric
                """
                col = df.loc[:, value_col]  # DataFrame if duplicate names
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]     # use first column only
                col = pd.to_numeric(col.squeeze(), errors="coerce")
                g = pd.DataFrame({group_col: df[group_col].values, "__y__": col.values})
                tmp = g.groupby(group_col, as_index=False)["__y__"].mean()
                s = tmp.set_index(group_col)["__y__"]
                return s
        
            def _scalar(x, default=np.nan) -> float:
                """Convert any type to a guaranteed float scalar."""
                try:
                    a = np.asarray(x)
                    return float(a.ravel()[0])
                except Exception:
                    return float(default)


            def welch_holm_posthoc(df, grp, yvar, alpha=0.05):
                """Pairwise Welch t-tests with Holm correction (used when pingouin is unavailable)."""
                from scipy import stats as sps
                pairs, pvals = [], []
                for g1, g2 in itertools.combinations(sorted(df[grp].unique()), 2):
                    x = df.loc[df[grp]==g1, yvar].values
                    y = df.loc[df[grp]==g2, yvar].values
                    _, p = sps.ttest_ind(x, y, equal_var=False)
                    pairs.append((g1,g2)); pvals.append(p)
                ph = pd.DataFrame(pairs, columns=["group1","group2"])
                from statsmodels.stats.multitest import multipletests
                ph["p-adj"] = multipletests(pvals, method="holm")[1]
                ph["reject"] = ph["p-adj"] < alpha
                return ph

            def tukey_table(endog, groups, alpha=0.05):
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                th = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)
                tb = pd.DataFrame(th.summary().data[1:], columns=th.summary().data[0])
                # Standardize column names
                tb = tb.rename(columns={"group1":"group1","group2":"group2","reject":"reject","p-adj":"p-adj"})
                if "p-adj" not in tb.columns:
                    # p-adj may be absent depending on statsmodels version -> NaN if unavailable
                    tb["p-adj"] = np.nan
                return tb

            def conover_friedman(df_block_rule: pd.DataFrame, alpha: float = 0.05):
                """
                Post-hoc test after significant Friedman:
                Primary: Conover + Holm (scikit-posthocs)
                Fallback: Nemenyi
                Always returns (posthoc_df, err_msg) tuple.
                """
                import numpy as np
                import pandas as pd
                try:
                    import scikit_posthocs as sp
                except Exception as e:
                    return pd.DataFrame(), f"scikit-posthocs import failed: {e}"

                # Common: wide -> long conversion function
                def _wide_to_long(ph_wide: pd.DataFrame) -> pd.DataFrame:
                    tmp = ph_wide.copy()
                    # Fix index name
                    if tmp.index.name is None:
                        tmp.index.name = "group1"
                    long = tmp.reset_index().melt(id_vars=tmp.index.name, var_name="group2", value_name="p-adj")
                    long = long[long["group1"] < long["group2"]].reset_index(drop=True)
                    long["reject"] = long["p-adj"] < alpha
                    return long

                # 1) Conover + Holm
                try:
                    ph = sp.posthoc_conover_friedman(df_block_rule, p_adjust='holm')
                    return _wide_to_long(ph), None
                except Exception as e1:
                    # 2) Nemenyi fallback
                    try:
                        ph2 = sp.posthoc_nemenyi_friedman(df_block_rule)
                        msg = f"Conover failed, using Nemenyi instead: {e1}"
                        return _wide_to_long(ph2), msg
                    except Exception as e2:
                        return pd.DataFrame(), f"Both Conover/Nemenyi failed: {e1} / {e2}"

            def cld_from_pairs(means: pd.Series, pair_tbl: pd.DataFrame, alpha=0.05):
                """
                Compact Letter Display based on the absorption algorithm
                (Piepho 2004, "An algorithm for a letter-based representation
                of all pairwise comparisons").

                Each group may receive multiple letters (e.g. "ab"). Two groups
                sharing at least one letter are NOT significantly different.
                Groups sharing NO letters ARE significantly different.

                - means: index=group name, values=mean (pre-sorted by caller)
                - pair_tbl: must contain ['group1','group2'] and ['reject'] or ['p-adj']
                Returns DataFrame with columns [rule, mean, CLD].
                """
                # If empty table, assign all 'a'
                if pair_tbl is None or pair_tbl.empty or len(means) <= 1:
                    return pd.DataFrame({"rule": means.index, "mean": means.values, "CLD": ["a"]*len(means)})

                ph = pair_tbl.copy()
                if "reject" in ph.columns:
                    ph["sig"] = ph["reject"].astype(bool)
                else:
                    ph["sig"] = ph["p-adj"] < alpha

                # Significant pairs set (sorted tuples)
                _key = lambda a, b: tuple(sorted((a, b)))
                sig_pairs = set()
                for _, r in ph.iterrows():
                    if bool(r["sig"]):
                        sig_pairs.add(_key(r["group1"], r["group2"]))

                ordered = list(means.index)
                n = len(ordered)

                # --- Absorption algorithm ---
                # Start: single letter 'a' assigned to all groups
                # Each letter defines a "family" of groups that are mutually non-significant.
                # If a family contains a significant pair, split it by removing one member
                # and assigning a new letter.

                # Initial: one family containing all groups
                families = [set(ordered)]  # list of sets

                changed = True
                max_iter = n * 26  # safety limit
                iteration = 0
                while changed and iteration < max_iter:
                    changed = False
                    iteration += 1
                    new_families = []
                    for fam in families:
                        # Check if this family contains any significant pair
                        has_sig = False
                        for a_grp in fam:
                            for b_grp in fam:
                                if a_grp < b_grp and _key(a_grp, b_grp) in sig_pairs:
                                    has_sig = True
                                    break
                            if has_sig:
                                break

                        if not has_sig:
                            new_families.append(fam)
                        else:
                            # Split: find the member whose removal resolves the most conflicts
                            best_remove = None
                            best_conflicts = -1
                            for candidate in fam:
                                conflicts = sum(
                                    1 for other in fam
                                    if other != candidate and _key(candidate, other) in sig_pairs
                                )
                                if conflicts > best_conflicts:
                                    best_conflicts = conflicts
                                    best_remove = candidate

                            # Keep family without the removed member
                            remaining = fam - {best_remove}
                            if remaining:
                                new_families.append(remaining)
                            # New family: the removed member + all non-significant partners from original
                            new_fam = {best_remove}
                            for other in fam:
                                if other != best_remove and _key(best_remove, other) not in sig_pairs:
                                    new_fam.add(other)
                            new_families.append(new_fam)
                            changed = True

                    # Deduplicate families (same set of members)
                    unique = []
                    seen = set()
                    for fam in new_families:
                        key_frozen = frozenset(fam)
                        if key_frozen not in seen:
                            seen.add(key_frozen)
                            unique.append(fam)
                    families = unique

                # Absorb: remove families that are subsets of other families
                families.sort(key=len, reverse=True)
                absorbed = []
                for i, fam in enumerate(families):
                    is_subset = False
                    for j, other in enumerate(absorbed):
                        if fam.issubset(other):
                            is_subset = True
                            break
                    if not is_subset:
                        absorbed.append(fam)
                families = absorbed

                # Assign letters (a, b, c, ...) to families, sorted by best mean
                def family_rank(fam):
                    # rank by the best (first in ordered list) member
                    return min(ordered.index(g) for g in fam)
                families.sort(key=family_rank)

                letters_list = [chr(ord('a') + i) if i < 26 else chr(ord('a') + i - 26).upper()
                                for i in range(len(families))]

                group_letters = {g: [] for g in ordered}
                for letter, fam in zip(letters_list, families):
                    for g in fam:
                        group_letters[g].append(letter)

                return pd.DataFrame({
                    "rule": ordered,
                    "mean": [means[g] for g in ordered],
                    "CLD":  ["".join(group_letters[g]) for g in ordered],
                })
            def _series_1d(obj) -> pd.Series:
                """Force any input (DataFrame/Series/ndarray/list) to 1D Series(float)."""
                if isinstance(obj, pd.DataFrame):
                    s = obj.iloc[:, 0]
                elif isinstance(obj, pd.Series):
                    s = obj
                else:
                    s = pd.Series(obj)
                return pd.to_numeric(s.astype(float), errors="coerce")

            def _make_dd_work(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
                """
                Create post-hoc work table always with 'rule' + '__y__' 2 columns.
                - Even if df[yvar] returns a DataFrame, only first column is used (forced 1D)
                - '__y__' is used as dv/dependent variable name in pingouin/Tukey etc.
                """
                y_s = _series_1d(df.loc[:, yvar])
                return pd.DataFrame({"rule": df["rule"].values, "__y__": y_s.values})


            if not rpath:
                st.caption(f"Raw file (results_{coord}.txt) not found; skipping ANOVA.")
            else:
                dfraw = parse_raw_results(rpath)  # must be long format
                if dfraw.empty:
                    st.caption("RAW parsing result is empty. Check file format.")
                else:
                    metric = st.selectbox("Metric", ["Reward","Time","PDR","Reward_woG","PDR_woG"], index=0)
                    d = dfraw[dfraw["metric"] == metric].copy()
                    if d.empty:
                        st.warning("No data found for the selected metric.")
                    else:
                        # Transform
                        # Time uses original scale, PDR allows logit selection, Reward uses original scale
                        if metric == "Time":
                            trans_opts, trans_idx = ["None"], 0
                        elif metric.startswith("PDR"):
                            trans_opts, trans_idx = ["None","logit(PDR)"], 1   # default logit
                        else:  # Reward, Reward_woG
                            trans_opts, trans_idx = ["None"], 0

                        trans = st.selectbox("Transform", trans_opts, index=trans_idx)
                        yvar = "value"; eps = 1e-6

                        if trans == "logit(PDR)":
                            d[yvar] = np.log((d[yvar]+eps)/(1-d[yvar]+eps)); st.caption("Logit transform applied to PDR.")
                        # (No transform for Time/Reward)

                        if "rule" not in d.columns:
                            d["rule"] = d[["Phase","RedPolicy","RedAction","YellowAction"]].agg(", ".join, axis=1)

                        mode = st.radio("Analysis Type", ["One-way (rule only)","One-way + Block(run) (RCBD recommended)","Reduced Factorial (main + 2-way)"],
                                        index=1, horizontal=True)
                        if "Block" in mode or "Factorial" in mode:
                            st.caption("RCBD assumes Common Random Numbers (CRN): all 64 rules within each run share the same random seed, so `run` is a valid block variable.")

                        if not HAS_SM:
                            st.warning("statsmodels not installed. Cannot run ANOVA. `pip install statsmodels` and retry.")
                        else:
                            import statsmodels.api as sm
                            import statsmodels.formula.api as smf
                            from scipy import stats as sps

                            # Fit model
                            if mode == "One-way (rule only)":
                                formula = f"{yvar} ~ C(rule)"
                                st.caption("Model: value ~ C(rule)")
                            elif mode == "One-way + Block(run) (RCBD recommended)":
                                formula = f"{yvar} ~ C(rule) + C(run)"
                                st.caption("Model: value ~ C(rule) + C(run)  (run=block)")
                            else:
                                # Reduced factorial: main effects + 2-way interactions + block(run)
                                formula = (f"{yvar} ~ C(run) + C(Phase) + C(RedPolicy) + C(RedAction) + C(YellowAction)"
                                           " + C(Phase):C(RedPolicy) + C(Phase):C(RedAction) + C(Phase):C(YellowAction)"
                                           " + C(RedPolicy):C(RedAction) + C(RedPolicy):C(YellowAction)"
                                           " + C(RedAction):C(YellowAction)")
                                st.caption("Model: C(run) + main effects + all 2-way interactions (3/4-way excluded for power)")

                            model = smf.ols(formula, data=d).fit()
                            anova_tbl = sm.stats.anova_lm(model, typ=2)

                            # Include Total + eta-squared
                            out = make_total_row(anova_tbl, d[yvar])
                            st.dataframe(out, width='stretch')

                            # Significant effects summary
                            alpha = st.slider("Significance Level (alpha)", 0.001, 0.1, 0.05, 0.001)
                            sig = out[(out.index!="Total") & (out["PR(>F)"] < alpha)].sort_values("PR(>F)")
                            if not sig.empty:
                                st.markdown("##### Interpretation Summary")
                                lines = []
                                for idx, r in sig.iterrows():
                                    omega = f", ω²={r['omega_sq']:.3f}" if pd.notna(r.get('omega_sq')) else ""
                                    lines.append(f"- **{idx}**: p={r['PR(>F)']:.3g}, η²={r['eta_sq']:.3f}{omega}")
                                st.markdown("\n".join(lines))
                            else:
                                st.caption("No significant effects found.")
                            st.caption(f"Model fit: R²={model.rsquared:.3f}, Adj.R²={model.rsquared_adj:.3f}")

                            # RCBD assumption check: Tukey non-additivity test
                            if mode == "One-way + Block(run) (RCBD recommended)":
                                try:
                                    # Tukey 1-df test for non-additivity
                                    fitted = model.fittedvalues
                                    resid_vals = model.resid
                                    d_tukey = d.copy()
                                    d_tukey["_fitted_sq"] = fitted ** 2
                                    model_aug = smf.ols(f"{yvar} ~ C(rule) + C(run) + _fitted_sq", data=d_tukey).fit()
                                    anova_aug = sm.stats.anova_lm(model_aug, typ=2)
                                    if "_fitted_sq" in anova_aug.index:
                                        p_nonadd = float(anova_aug.loc["_fitted_sq", "PR(>F)"])
                                        st.write(f"Tukey Non-additivity: p={p_nonadd:.3g}")
                                        if p_nonadd < alpha:
                                            st.warning("⚠️ Significant block×treatment interaction detected (Tukey non-additivity p < alpha). RCBD additivity assumption may be violated.")
                                except Exception as e_tukey:
                                    st.caption(f"Tukey non-additivity test skipped: {e_tukey}")

                            # Residual Diagnostics
                            st.markdown("##### Residual Diagnostics")
                            resid = model.resid
                            fitted_vals = model.fittedvalues

                            # Shapiro-Wilk
                            if len(resid) >= 3:
                                try:
                                    W, p_shap = (sps.shapiro(resid.sample(min(len(resid), 500), random_state=0))
                                                if len(resid) > 500 else sps.shapiro(resid))
                                    st.write(f"Shapiro-Wilk: W={W:.4f}, p={p_shap:.3g}")
                                except Exception as e:
                                    p_shap = 1.0; st.caption(f"Shapiro-Wilk computation failed: {e}")
                            else:
                                p_shap = 1.0

                            # Anderson-Darling
                            try:
                                ad_result = sps.anderson(resid, dist="norm", method="interpolate")
                                st.write(f"Anderson-Darling: A²={ad_result.statistic:.4f}, "
                                         f"p={ad_result.pvalue:.4f}")
                                if ad_result.pvalue < 0.05:
                                    st.caption("Anderson-Darling rejects normality at 5% level.")
                            except Exception:
                                pass

                            # QQ plot
                            qq = sps.probplot(resid, dist="norm")
                            qq_df = pd.DataFrame({"Theoretical": qq[0][0], "Residual": np.sort(resid)})
                            st.altair_chart(alt.Chart(qq_df).mark_point().encode(x="Theoretical:Q", y="Residual:Q").properties(title="QQ Plot", height=280), width='stretch')

                            # Residual histogram
                            st.altair_chart(alt.Chart(pd.DataFrame({"resid": resid})).mark_bar().encode(x=alt.X("resid:Q", bin=alt.Bin(maxbins=40)), y="count()").properties(title="Residual Histogram", height=200), width='stretch')

                            # Residuals vs Fitted scatter
                            rvf_df = pd.DataFrame({"Fitted": fitted_vals, "Residual": resid})
                            rvf_chart = alt.Chart(rvf_df).mark_point(opacity=0.5).encode(
                                x=alt.X("Fitted:Q"), y=alt.Y("Residual:Q")
                            ).properties(title="Residuals vs Fitted", height=280)
                            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4,4]).encode(y="y:Q")
                            st.altair_chart(rvf_chart + zero_line, width='stretch')
                        

                            # p_shap scalar fix (Levene computed once in post-hoc section below)
                            p_shap = _scalar(p_shap, default=1.0)
                            p_lev  = np.nan  # computed in post-hoc section below
                            alpha  = _scalar(alpha,  default=0.05)

                            # ===== Post-hoc tests & CLD =====
                            st.markdown("##### Post-hoc Tests")
                            posthoc = pd.DataFrame(); explain = ""

                            def _small_is_better(m: str) -> bool:
                                # Time, PDR(incl. woG)=smaller is better / Reward types=larger is better
                                return (m == "Time") or m.startswith("PDR")

                            # --- EMM-based post-hoc for RCBD; Games-Howell for One-way ---
                            def _emm_pairwise(model_obj, data, rule_col, block_col, yvar, alpha_val):
                                """
                                Estimated Marginal Means (EMM) pairwise comparison.
                                Uses RCBD model's MS_residual as the pooled error term.
                                Pairwise differences tested with t-distribution, Holm-corrected.
                                """
                                ms_res = model_obj.mse_resid
                                df_res = model_obj.df_resid
                                rules = sorted(data[rule_col].unique())
                                n_per_cell = data.groupby(rule_col).size()

                                # EMM = marginal mean of each rule (averaged over blocks)
                                emm = data.groupby(rule_col)[yvar].mean()

                                pairs, pvals, diffs = [], [], []
                                for g1, g2 in itertools.combinations(rules, 2):
                                    diff = emm[g1] - emm[g2]
                                    n1, n2 = n_per_cell[g1], n_per_cell[g2]
                                    se = np.sqrt(ms_res * (1.0/n1 + 1.0/n2))
                                    t_stat = diff / se if se > 0 else 0
                                    p_val = 2.0 * (1.0 - sps.t.cdf(abs(t_stat), df_res))
                                    pairs.append((g1, g2))
                                    pvals.append(p_val)
                                    diffs.append(diff)

                                from statsmodels.stats.multitest import multipletests
                                reject, p_adj, _, _ = multipletests(pvals, method="holm", alpha=alpha_val)
                                ph = pd.DataFrame({
                                    "group1": [p[0] for p in pairs],
                                    "group2": [p[1] for p in pairs],
                                    "diff": diffs,
                                    "p-adj": p_adj,
                                    "reject": reject,
                                })
                                return ph, emm

                            if mode == "One-way + Block(run) (RCBD recommended)":
                                # EMM-based pairwise comparison using RCBD model error
                                try:
                                    posthoc, emm_means = _emm_pairwise(model, d, "rule", "run", yvar, alpha)
                                    means_for_cld = emm_means.sort_values(ascending=_small_is_better(metric))
                                    explain = "EMM pairwise t-tests (RCBD MS_residual) + Holm correction"
                                except Exception as e_emm:
                                    # Fallback to block-adjusted Games-Howell
                                    dd = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                    dd_work = _make_dd_work(dd, yvar)
                                    means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                        ascending=_small_is_better(metric))
                                    try:
                                        import pingouin as pg
                                        gh = pg.pairwise_gameshowell(dv="__y__", between="rule", data=dd_work)
                                        posthoc = gh.rename(columns={"A":"group1","B":"group2","pval":"p-adj"})
                                        posthoc["reject"] = posthoc["p-adj"] < alpha
                                        explain = f"Games-Howell on block-adjusted y* (EMM failed: {e_emm})"
                                    except Exception:
                                        y_post = dd_work["__y__"]; grp_post = dd_work["rule"]
                                        posthoc = welch_holm_posthoc(
                                            pd.DataFrame({"rule": grp_post.values, "y": y_post.values}),
                                            "rule", "y", alpha=alpha)
                                        explain = f"Welch t + Holm on block-adjusted y* (EMM failed: {e_emm})"
                                # Levene on block-adjusted residuals
                                dd_lev = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                dd_lev_work = _make_dd_work(dd_lev, yvar)
                                lev_groups = [g["__y__"].values for _, g in dd_lev_work.groupby("rule")]
                            else:
                                # One-way: Games-Howell (robust to unequal variance)
                                dd_work = _make_dd_work(d, yvar)
                                y_post  = dd_work["__y__"]
                                grp_post= dd_work["rule"]
                                means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                    ascending=_small_is_better(metric))
                                lev_groups = [g["__y__"].values for _, g in dd_work.groupby("rule")]
                                try:
                                    import pingouin as pg
                                    gh = pg.pairwise_gameshowell(dv="__y__", between="rule", data=dd_work)
                                    posthoc = gh.rename(columns={"A":"group1","B":"group2","pval":"p-adj"})
                                    posthoc["reject"] = posthoc["p-adj"] < alpha
                                    explain = "Games-Howell"
                                except Exception:
                                    posthoc = welch_holm_posthoc(
                                        pd.DataFrame({"rule": grp_post.values, "y": y_post.values}),
                                        "rule", "y", alpha=alpha)
                                    explain = "Pairwise Welch t-tests + Holm correction (pingouin unavailable)"

                            try:
                                if len(lev_groups) >= 2 and all(len(x) > 1 for x in lev_groups):
                                    p_lev = sps.levene(*lev_groups, center="median").pvalue
                                else:
                                    p_lev = np.nan
                                st.write(f"Levene(Brown-Forsythe): p={_scalar(p_lev):.3g}")
                            except Exception:
                                p_lev = np.nan

                            # --- Output results & CLD ---
                            if not posthoc.empty:
                                st.caption(explain)
                                st.dataframe(posthoc, width='stretch')

                                ph = posthoc.copy()
                                if ("p-adj" not in ph.columns) and ("reject" not in ph.columns):
                                    st.info("No p-value information available for CLD.")
                                else:
                                    cld = cld_from_pairs(means_for_cld, ph, alpha=alpha)
                                    st.markdown("##### CLD (shared letter = no significant difference, 'a' = best)")
                                    st.caption("CLD uses the absorption algorithm (Piepho 2004). Groups may have multiple letters (e.g. 'ab'). "
                                               "Two groups sharing at least one letter are not significantly different.")
                                    st.dataframe(cld, width='stretch')

                                    # Top candidates (‘A’ group) -- sorted by metric direction
                                    st.markdown(f"#### Top Candidates (**{metric}**, groups containing letter 'a')")
                                    top = cld[cld["CLD"].str.contains("a", na=False)].sort_values("mean", ascending=_small_is_better(metric))
                                    st.dataframe(top, width='stretch')

                                    # ================== A-Group Intersection (Reward, Time, PDR, RCBD-based) ==================
                                    st.markdown("### A-Group Intersection (Reward ∩ Time(asc) ∩ PDR(asc), RCBD)")

                                    def _prep_metric(dfraw_all: pd.DataFrame, metric_name: str):
                                        """Extract metric rows from raw(long) + fix rule column."""
                                        dsub = dfraw_all[dfraw_all["metric"] == metric_name].copy()
                                        if dsub.empty:
                                            return pd.DataFrame()
                                        if "rule" not in dsub.columns:
                                            dsub["rule"] = dsub[["Phase","RedPolicy","RedAction","YellowAction"]].agg(", ".join, axis=1)
                                        return dsub

                                    def _transform_for_metric(d: pd.DataFrame, metric_name: str, eps: float = 1e-6):
                                        """
                                        Transform scale:
                                        - PDR (incl. woG): logit
                                        - Time, Reward (incl. woG): original scale
                                        """
                                        d = d.copy(); yvar = "value"
                                        if metric_name.startswith("PDR"):
                                            d[yvar] = np.log((d[yvar] + eps)/(1 - d[yvar] + eps))  # logit
                                            scale = "logit"
                                        else:
                                            scale = "original"
                                        return d, yvar, scale

                                    def _rcbd_posthoc_cld(d: pd.DataFrame, yvar: str, alpha: float = 0.05, prefer_small_is_A: bool = False):
                                        """
                                        RCBD: y ~ C(rule) + C(run), EMM-based pairwise t-tests using
                                        the model's MS_residual as pooled error, Holm-corrected.
                                        CLD via absorption algorithm (Piepho 2004).
                                        prefer_small_is_A=True  -> ascending sort (a = smallest = Best)
                                        prefer_small_is_A=False -> descending sort (a = largest = Best)
                                        """
                                        from scipy import stats as sps

                                        posthoc = pd.DataFrame()
                                        explain = ""

                                        # Fit RCBD model for this metric
                                        try:
                                            rcbd_model = smf.ols(f"{yvar} ~ C(rule) + C(run)", data=d).fit()
                                            posthoc_emm, emm_means = _emm_pairwise(rcbd_model, d, "rule", "run", yvar, alpha)
                                            posthoc = posthoc_emm
                                            means_for_cld = emm_means.sort_values(ascending=prefer_small_is_A)
                                            explain = "EMM pairwise t-tests (RCBD MS_residual) + Holm"
                                        except Exception as e_emm:
                                            # Fallback: block-adjusted Games-Howell
                                            dd = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                            dd_work = _make_dd_work(dd, yvar)
                                            means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                                ascending=prefer_small_is_A)
                                            try:
                                                import pingouin as pg
                                                gh = pg.pairwise_gameshowell(dv="__y__", between="rule", data=dd_work)
                                                posthoc = gh.rename(columns={"A":"group1","B":"group2","pval":"p-adj"})
                                                posthoc["reject"] = posthoc["p-adj"] < alpha
                                                explain = f"Games-Howell block-adjusted (EMM failed: {e_emm})"
                                            except Exception:
                                                y_post = dd_work["__y__"]; grp_post = dd_work["rule"]
                                                posthoc = welch_holm_posthoc(
                                                    pd.DataFrame({"rule": grp_post.values, "y": y_post.values}),
                                                    "rule", "y", alpha=alpha)
                                                explain = f"Welch t + Holm block-adjusted (EMM failed: {e_emm})"

                                        # --- CLD computation
                                        if posthoc.empty or (("p-adj" not in posthoc.columns) and ("reject" not in posthoc.columns)):
                                            return means_for_cld, pd.DataFrame(), explain

                                        cld = cld_from_pairs(means_for_cld, posthoc, alpha=alpha)
                                        return means_for_cld, cld, explain

                                    with st.expander("A-Group Intersection (Reward up, Time down, PDR down)", expanded=True):
                                        alpha_int = st.slider("Alpha for intersection", 0.001, 0.1, 0.05, 0.001, key="alpha_intersect_all")
                                        alpha_int = float(alpha_int)   # scalar from slider value
                                        # --- Reward (larger = A) ---
                                        d_rew = _prep_metric(dfraw, "Reward")
                                        if d_rew.empty:
                                            st.info("No Reward data.")
                                            A_rew, disp_rew = set(), pd.Series(dtype=float)
                                        else:
                                            d_rew_tr, y_rew, _ = _transform_for_metric(d_rew, "Reward")
                                            means_rew, cld_rew, _ = _rcbd_posthoc_cld(d_rew_tr, y_rew, alpha=alpha_int, prefer_small_is_A=False)
                                            A_rew = set(cld_rew.loc[cld_rew["CLD"].str.contains("a", na=False),"rule"]) if not cld_rew.empty else set()
                                            disp_rew = d_rew.groupby("rule")["value"].mean().rename("Reward_mean(orig)")

                                        # --- Time (smaller = A) ---
                                        d_time = _prep_metric(dfraw, "Time")
                                        if d_time.empty:
                                            st.info("No Time data.")
                                            A_time, disp_time = set(), pd.Series(dtype=float)
                                        else:
                                            d_time_tr, y_time, _ = _transform_for_metric(d_time, "Time")
                                            means_time, cld_time, _ = _rcbd_posthoc_cld(d_time_tr, y_time, alpha=alpha_int, prefer_small_is_A=True)
                                            A_time = set(cld_time.loc[cld_time["CLD"].str.contains("a", na=False),"rule"]) if not cld_time.empty else set()
                                            disp_time = d_time.groupby("rule")["value"].mean().rename("Time_mean(orig)")

                                        # --- PDR (smaller = A; logit analysis, display uses original scale mean) ---
                                        d_pdr = _prep_metric(dfraw, "PDR")
                                        if d_pdr.empty:
                                            st.info("No PDR data.")
                                            A_pdr, disp_pdr = set(), pd.Series(dtype=float)
                                        else:
                                            d_pdr_tr, y_pdr, _ = _transform_for_metric(d_pdr, "PDR")
                                            means_pdr, cld_pdr, _ = _rcbd_posthoc_cld(d_pdr_tr, y_pdr, alpha=alpha_int, prefer_small_is_A=True)
                                            A_pdr = set(cld_pdr.loc[cld_pdr["CLD"].str.contains("a", na=False),"rule"]) if not cld_pdr.empty else set()
                                            disp_pdr = d_pdr.groupby("rule")["value"].mean().rename("PDR_mean(orig)")

                                        # --- Show set & intersection results ---
                                        st.markdown(f"- **A(Reward)**: {len(A_rew)}, **A(Time)**: {len(A_time)}, **A(PDR)**: {len(A_pdr)}")

                                        inter_RT  = sorted(A_rew.intersection(A_time))
                                        inter_RP  = sorted(A_rew.intersection(A_pdr))
                                        inter_TP  = sorted(A_time.intersection(A_pdr))
                                        inter_RTP = sorted(A_rew.intersection(A_time).intersection(A_pdr))

                                        def _show_table(title, rules):
                                            st.markdown(f"**{title}** — {len(rules)} rules")
                                            if len(rules) == 0:
                                                st.caption("N/A")
                                                return
                                            out = (pd.DataFrame({"rule": rules})
                                                    .merge(disp_rew, on="rule", how="left")
                                                    .merge(disp_time, on="rule", how="left")
                                                    .merge(disp_pdr, on="rule", how="left"))
                                            # Sort: triple intersection by Reward desc, tie-breaker Time asc, PDR asc
                                            if "Triple" in title:
                                                out = out.sort_values(["Reward_mean(orig)", "Time_mean(orig)", "PDR_mean(orig)"],
                                                                    ascending=[False, True, True], kind="mergesort")
                                            elif "Reward∩Time" in title:
                                                out = out.sort_values(["Reward_mean(orig)", "Time_mean(orig)"],
                                                                    ascending=[False, True], kind="mergesort")
                                            elif "Reward∩PDR" in title:
                                                out = out.sort_values(["Reward_mean(orig)", "PDR_mean(orig)"],
                                                                    ascending=[False, True], kind="mergesort")
                                            elif "Time∩PDR" in title:
                                                out = out.sort_values(["Time_mean(orig)", "PDR_mean(orig)"],
                                                                    ascending=[True, True], kind="mergesort")
                                            st.dataframe(out, width='stretch')

                                        _show_table("Triple Intersection (Reward ∩ Time ∩ PDR)", inter_RTP)
                                        _show_table("Double Intersection (Reward∩Time)", inter_RT)
                                        _show_table("Double Intersection (Reward∩PDR)", inter_RP)
                                        _show_table("Double Intersection (Time∩PDR)", inter_TP)

                            else:
                                st.caption("No post-hoc test results.")

            # ══════════════════════════════════════════════════════════════
            # TAB 4: Pareto Dominance Analysis
            # ══════════════════════════════════════════════════════════════
            with analytics_tabs[3]:
                st.subheader("Multi-Objective Dominance Analysis")
                st.caption("Statistical Pareto efficiency: rule A dominates B iff A is significantly better on at least one metric AND not significantly worse on any metric.")

                if not rpath or not os.path.exists(rpath):
                    st.info("Load RAW data first (results_*.txt required).")
                else:
                    pareto_alpha = st.slider("Significance level (alpha)", 0.001, 0.1, 0.05, 0.001, key="pareto_alpha")
                    pareto_alpha = float(pareto_alpha)

                    # Prepare data for all 3 metrics
                    _pareto_metrics = {"Reward": False, "Time": True, "PDR": True}  # True = smaller is better
                    _pareto_clds = {}
                    _pareto_posthocs = {}
                    _pareto_means = {}
                    _pareto_ok = True

                    for _pm, _pm_asc in _pareto_metrics.items():
                        _pd_raw = _prep_metric(dfraw, _pm) if 'dfraw' in dir() else pd.DataFrame()
                        if _pd_raw.empty:
                            _pareto_ok = False
                            break
                        _pd_tr, _py, _ = _transform_for_metric(_pd_raw, _pm)
                        try:
                            _p_means, _p_cld, _p_explain = _rcbd_posthoc_cld(_pd_tr, _py, alpha=pareto_alpha, prefer_small_is_A=_pm_asc)
                            # Also get the post-hoc table
                            _rcbd_m = smf.ols(f"{_py} ~ C(rule) + C(run)", data=_pd_tr).fit()
                            _p_ph, _p_emm = _emm_pairwise(_rcbd_m, _pd_tr, "rule", "run", _py, pareto_alpha)
                            _pareto_posthocs[_pm] = _p_ph
                            _pareto_means[_pm] = _pd_raw.groupby("rule")["value"].mean()
                        except Exception as e_pareto:
                            st.warning(f"Could not compute post-hoc for {_pm}: {e_pareto}")
                            _pareto_ok = False
                            break

                    if not _pareto_ok or len(_pareto_posthocs) < 3:
                        st.warning("All 3 metrics (Reward, Time, PDR) are required for Pareto analysis.")
                    else:
                        rules_all = sorted(_pareto_means["Reward"].index.tolist())
                        n_rules = len(rules_all)

                        # Build pairwise significance lookup
                        def _sig_lookup(ph_df, alpha_v):
                            """Returns dict: (g1,g2) -> p_adj"""
                            lookup = {}
                            for _, row in ph_df.iterrows():
                                lookup[(row["group1"], row["group2"])] = row["p-adj"]
                                lookup[(row["group2"], row["group1"])] = row["p-adj"]
                            return lookup

                        sig_tables = {m: _sig_lookup(ph, pareto_alpha) for m, ph in _pareto_posthocs.items()}

                        # Statistical dominance check
                        dominance_matrix = np.zeros((n_rules, n_rules), dtype=int)  # 1 if row dominates col
                        for i, ri in enumerate(rules_all):
                            for j, rj in enumerate(rules_all):
                                if i == j:
                                    continue
                                sig_better_any = False
                                sig_worse_any = False
                                for _m, _m_asc in _pareto_metrics.items():
                                    p_val = sig_tables[_m].get((ri, rj), 1.0)
                                    mean_i = _pareto_means[_m][ri]
                                    mean_j = _pareto_means[_m][rj]
                                    if _m_asc:  # smaller is better
                                        is_better = mean_i < mean_j
                                    else:  # larger is better
                                        is_better = mean_i > mean_j
                                    if p_val < pareto_alpha:
                                        if is_better:
                                            sig_better_any = True
                                        else:
                                            sig_worse_any = True
                                if sig_better_any and not sig_worse_any:
                                    dominance_matrix[i, j] = 1

                        # Compute dominance counts and Pareto layers
                        dom_count = dominance_matrix.sum(axis=1)   # how many others I dominate
                        dom_by_count = dominance_matrix.sum(axis=0)  # how many dominate me

                        # Non-dominated sorting (NSGA-II style)
                        remaining = set(range(n_rules))
                        layers = []
                        while remaining:
                            front = []
                            for i in remaining:
                                dominated = False
                                for j in remaining:
                                    if i != j and dominance_matrix[j, i] == 1:
                                        dominated = True
                                        break
                                if not dominated:
                                    front.append(i)
                            layers.append(front)
                            remaining -= set(front)
                            if len(layers) > n_rules:
                                break

                        layer_map = {}
                        for li, layer in enumerate(layers):
                            for idx in layer:
                                layer_map[idx] = li + 1

                        # Display Pareto summary table
                        pareto_df = pd.DataFrame({
                            "rule": rules_all,
                            "Pareto_Layer": [layer_map[i] for i in range(n_rules)],
                            "Dominates": dom_count.tolist(),
                            "Dominated_By": dom_by_count.tolist(),
                            "Reward_mean": [_pareto_means["Reward"][r] for r in rules_all],
                            "Time_mean": [_pareto_means["Time"][r] for r in rules_all],
                            "PDR_mean": [_pareto_means["PDR"][r] for r in rules_all],
                        }).sort_values(["Pareto_Layer", "Dominates"], ascending=[True, False]).reset_index(drop=True)

                        st.markdown("#### Pareto Layers (Layer 1 = non-dominated front)")
                        st.dataframe(
                            pareto_df.style.format({
                                "Reward_mean": "{:.4f}", "Time_mean": "{:.2f}", "PDR_mean": "{:.4f}"
                            }),
                            width='stretch', hide_index=True,
                        )

                        n_front1 = len([x for x in layers[0]]) if layers else 0
                        st.markdown(f"**Pareto Front (Layer 1):** {n_front1} rules | "
                                    f"**Total Layers:** {len(layers)}")

                        # 3D Scatter with Pareto layers
                        import plotly.graph_objects as go_pareto
                        fig_pareto = go_pareto.Figure()
                        layer_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#95a5a6"]
                        for li, layer_idxs in enumerate(layers[:6]):
                            layer_rules = [rules_all[i] for i in layer_idxs]
                            fig_pareto.add_trace(go_pareto.Scatter3d(
                                x=[_pareto_means["PDR"][r] for r in layer_rules],
                                y=[_pareto_means["Time"][r] for r in layer_rules],
                                z=[_pareto_means["Reward"][r] for r in layer_rules],
                                mode="markers+text",
                                marker=dict(size=8 if li == 0 else 5, color=layer_colors[li % len(layer_colors)],
                                            opacity=0.9 if li == 0 else 0.5),
                                text=[r.split(",")[-1].strip()[:15] for r in layer_rules],
                                textposition="top center",
                                textfont=dict(size=8),
                                name=f"Layer {li+1} ({len(layer_idxs)})",
                                hovertext=layer_rules,
                            ))
                        fig_pareto.update_layout(
                            scene=dict(xaxis_title="PDR (lower=better)", yaxis_title="Time (lower=better)",
                                       zaxis_title="Reward (higher=better)"),
                            height=600, margin=dict(l=0, r=0, t=30, b=0),
                            title="Statistical Pareto Dominance (3D)",
                        )
                        st.plotly_chart(fig_pareto, width='stretch')

                        # Dominance heatmap (compact)
                        with st.expander("Dominance Matrix (interactive)"):
                            short_names = [r.split(",")[-1].strip()[:20] for r in rules_all]
                            fig_heat = go_pareto.Figure(data=go_pareto.Heatmap(
                                z=dominance_matrix, x=short_names, y=short_names,
                                colorscale=[[0, "#2c3e50"], [1, "#e74c3c"]],
                                showscale=False,
                                hovertemplate="Row %{y} dominates Col %{x}: %{z}<extra></extra>",
                            ))
                            fig_heat.update_layout(height=600, xaxis_tickangle=45,
                                                   title="Dominance Matrix (1 = row statistically dominates column)")
                            st.plotly_chart(fig_heat, width='stretch')

            # ══════════════════════════════════════════════════════════════
            # TAB 5: Bootstrap / Non-Parametric Alternatives
            # ══════════════════════════════════════════════════════════════
            with analytics_tabs[4]:
                st.subheader("Bootstrap CI & Non-Parametric Tests")
                st.caption("When ANOVA normality assumptions are violated, use these robust alternatives.")

                if not rpath or not os.path.exists(rpath):
                    st.info("Load RAW data first (results_*.txt required).")
                else:
                    _bs_metric = st.selectbox("Metric", ["Reward", "Time", "PDR", "Reward w.o.G", "PDR w.o.G"],
                                              key="bs_metric_sel")
                    _bs_method = st.radio("Method", ["BCa Bootstrap CI", "Friedman Test (RCBD alternative)",
                                                     "Kruskal-Wallis (One-way alternative)"], key="bs_method")

                    _bs_raw = _prep_metric(dfraw, _bs_metric) if 'dfraw' in dir() else pd.DataFrame()
                    if _bs_raw.empty:
                        st.warning(f"No data for {_bs_metric}.")
                    else:
                        if _bs_method == "BCa Bootstrap CI":
                            st.markdown("#### BCa Bootstrap Confidence Intervals")
                            _bs_nboot = st.number_input("Bootstrap iterations", 1000, 50000, 9999, 1000, key="bs_nboot")
                            _bs_conf = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="bs_conf")

                            with st.spinner("Computing bootstrap CIs..."):
                                from scipy.stats import bootstrap as scipy_bootstrap
                                _bs_rules = sorted(_bs_raw["rule"].unique())
                                _bs_results = []
                                for _br in _bs_rules:
                                    _bdata = _bs_raw.loc[_bs_raw["rule"] == _br, "value"].values
                                    try:
                                        _bci = scipy_bootstrap(
                                            (_bdata,), statistic=np.mean,
                                            n_resamples=int(_bs_nboot),
                                            confidence_level=float(_bs_conf),
                                            method="BCa",
                                        )
                                        _bs_results.append({
                                            "rule": _br, "mean": np.mean(_bdata),
                                            "CI_low": _bci.confidence_interval.low,
                                            "CI_high": _bci.confidence_interval.high,
                                            "CI_width": _bci.confidence_interval.high - _bci.confidence_interval.low,
                                            "std_error": _bci.standard_error,
                                        })
                                    except Exception as _be:
                                        _bs_results.append({
                                            "rule": _br, "mean": np.mean(_bdata),
                                            "CI_low": np.nan, "CI_high": np.nan,
                                            "CI_width": np.nan, "std_error": np.nan,
                                        })
                                _bs_df = pd.DataFrame(_bs_results)
                                _is_asc = _small_is_better(_bs_metric) if '_small_is_better' in dir() else False
                                _bs_df = _bs_df.sort_values("mean", ascending=_is_asc).reset_index(drop=True)

                            st.dataframe(
                                _bs_df.style.format({"mean": "{:.4f}", "CI_low": "{:.4f}", "CI_high": "{:.4f}",
                                                     "CI_width": "{:.4f}", "std_error": "{:.4f}"}),
                                width='stretch', hide_index=True,
                            )
                            st.caption(f"BCa Bootstrap ({int(_bs_nboot)} resamples, {_bs_conf*100:.0f}% CI). "
                                       "No normality assumption required.")

                            # Bootstrap CI comparison chart
                            import plotly.graph_objects as go_bs
                            fig_bs = go_bs.Figure()
                            fig_bs.add_trace(go_bs.Scatter(
                                x=_bs_df["mean"], y=list(range(len(_bs_df))),
                                error_x=dict(type="data",
                                             symmetric=False,
                                             array=(_bs_df["CI_high"] - _bs_df["mean"]).tolist(),
                                             arrayminus=(_bs_df["mean"] - _bs_df["CI_low"]).tolist()),
                                mode="markers",
                                marker=dict(size=6, color="#1abc9c"),
                                text=_bs_df["rule"],
                                hovertemplate="%{text}<br>Mean: %{x:.4f}<extra></extra>",
                            ))
                            fig_bs.update_layout(
                                yaxis=dict(tickmode="array", tickvals=list(range(len(_bs_df))),
                                           ticktext=[r[:30] for r in _bs_df["rule"]], autorange="reversed"),
                                xaxis_title=_bs_metric,
                                height=max(400, len(_bs_df) * 18),
                                margin=dict(l=250, r=20, t=30, b=30),
                                title=f"BCa Bootstrap CI ({_bs_metric})",
                            )
                            st.plotly_chart(fig_bs, width='stretch')

                        elif _bs_method == "Friedman Test (RCBD alternative)":
                            st.markdown("#### Friedman Test (non-parametric RCBD)")
                            st.caption("Tests whether rule rankings are consistent across runs. No normality assumption.")

                            # Pivot to wide format: rows=run, cols=rule
                            if "run" not in _bs_raw.columns:
                                st.warning("Run (block) information not available in raw data.")
                            else:
                                _fr_pivot = _bs_raw.pivot_table(index="run", columns="rule", values="value", aggfunc="mean")
                                _fr_pivot = _fr_pivot.dropna(axis=1)
                                if _fr_pivot.shape[1] < 2:
                                    st.warning("Need at least 2 rules for Friedman test.")
                                else:
                                    from scipy.stats import friedmanchisquare
                                    _fr_groups = [_fr_pivot[c].values for c in _fr_pivot.columns]
                                    _fr_stat, _fr_p = friedmanchisquare(*_fr_groups)
                                    _fr_n = _fr_pivot.shape[0]
                                    _fr_k = _fr_pivot.shape[1]
                                    # Kendall's W
                                    _fr_W = _fr_stat / (_fr_n * (_fr_k - 1)) if (_fr_n * (_fr_k - 1)) > 0 else 0

                                    col_fr1, col_fr2, col_fr3, col_fr4 = st.columns(4)
                                    col_fr1.metric("Chi-square", f"{_fr_stat:.3f}")
                                    col_fr2.metric("p-value", f"{_fr_p:.3g}")
                                    col_fr3.metric("Kendall's W", f"{_fr_W:.4f}")
                                    col_fr4.metric("Effect", "Large" if _fr_W > 0.5 else "Medium" if _fr_W > 0.3 else "Small")

                                    if _fr_p < 0.05:
                                        st.success(f"Significant (p={_fr_p:.3g}): Rule rankings differ across runs.")
                                    else:
                                        st.info(f"Not significant (p={_fr_p:.3g}): No evidence of consistent ranking differences.")

                                    # Post-hoc: Conover-Friedman or Nemenyi
                                    with st.expander("Post-hoc: Conover-Friedman pairwise comparisons"):
                                        try:
                                            from scipy.stats import rankdata
                                            from statsmodels.stats.multitest import multipletests as _mt
                                            # Rank within each block
                                            _fr_ranks = _fr_pivot.rank(axis=1)
                                            _fr_mean_ranks = _fr_ranks.mean()
                                            _fr_rule_names = _fr_pivot.columns.tolist()
                                            # Conover test statistic
                                            _A2 = (_fr_ranks.values ** 2).sum()
                                            _C = _fr_n * _fr_k * (_fr_k + 1)**2 / 4
                                            _T1 = (_fr_n - 1) * _fr_stat / (_fr_n * (_fr_k - 1) - _fr_stat) if (_fr_n * (_fr_k - 1) - _fr_stat) > 0 else 0
                                            _se = np.sqrt(2 * _fr_n * (_A2 - _C) / ((_fr_n - 1) * (_fr_k - 1))) if (_fr_n - 1) * (_fr_k - 1) > 0 else 1

                                            from scipy.stats import t as t_dist
                                            _con_pairs, _con_pvals = [], []
                                            for i in range(len(_fr_rule_names)):
                                                for j in range(i+1, len(_fr_rule_names)):
                                                    _diff = abs(_fr_mean_ranks[_fr_rule_names[i]] - _fr_mean_ranks[_fr_rule_names[j]])
                                                    _t_val = _diff / (_se / np.sqrt(_fr_n)) if _se > 0 else 0
                                                    _df_con = (_fr_n - 1) * (_fr_k - 1)
                                                    _p_con = 2 * (1 - t_dist.cdf(abs(_t_val), _df_con))
                                                    _con_pairs.append((_fr_rule_names[i], _fr_rule_names[j]))
                                                    _con_pvals.append(_p_con)

                                            _rej, _padj, _, _ = _mt(_con_pvals, method="holm")
                                            _con_df = pd.DataFrame({
                                                "group1": [p[0] for p in _con_pairs],
                                                "group2": [p[1] for p in _con_pairs],
                                                "p-adj": _padj, "reject": _rej,
                                            })
                                            st.dataframe(_con_df, width='stretch')
                                        except Exception as e_con:
                                            st.warning(f"Conover post-hoc failed: {e_con}")

                        else:  # Kruskal-Wallis
                            st.markdown("#### Kruskal-Wallis Test (non-parametric one-way)")
                            st.caption("Non-parametric alternative to one-way ANOVA. No normality assumption.")

                            from scipy.stats import kruskal
                            _kw_rules = sorted(_bs_raw["rule"].unique())
                            _kw_groups = [_bs_raw.loc[_bs_raw["rule"] == r, "value"].values for r in _kw_rules]
                            _kw_stat, _kw_p = kruskal(*_kw_groups)
                            _kw_n = len(_bs_raw)
                            # Epsilon-squared effect size
                            _kw_eps2 = (_kw_stat - len(_kw_rules) + 1) / (_kw_n - len(_kw_rules)) if (_kw_n - len(_kw_rules)) > 0 else 0

                            col_kw1, col_kw2, col_kw3 = st.columns(3)
                            col_kw1.metric("H-statistic", f"{_kw_stat:.3f}")
                            col_kw2.metric("p-value", f"{_kw_p:.3g}")
                            col_kw3.metric("Epsilon-sq", f"{_kw_eps2:.4f}")

                            if _kw_p < 0.05:
                                st.success(f"Significant (p={_kw_p:.3g}): At least one rule differs.")
                                with st.expander("Post-hoc: Dunn's test with Holm correction"):
                                    try:
                                        import scikit_posthocs as sp
                                        _dunn = sp.posthoc_dunn(_bs_raw, val_col="value", group_col="rule", p_adjust="holm")
                                        st.dataframe(_dunn.style.format("{:.4f}"), width='stretch')
                                    except ImportError:
                                        # Manual Dunn's approximation
                                        from scipy.stats import rankdata as _rd
                                        from statsmodels.stats.multitest import multipletests as _mt2
                                        _all_vals = _bs_raw["value"].values
                                        _all_ranks = _rd(_all_vals)
                                        _bs_raw_c = _bs_raw.copy()
                                        _bs_raw_c["_rank"] = _all_ranks
                                        _mean_ranks = _bs_raw_c.groupby("rule")["_rank"].mean()
                                        _ns = _bs_raw_c.groupby("rule").size()
                                        _N = len(_all_vals)
                                        _tie_corr = 1  # simplified
                                        _dunn_pairs, _dunn_pvals = [], []
                                        for ki in range(len(_kw_rules)):
                                            for kj in range(ki+1, len(_kw_rules)):
                                                _ri, _rj = _kw_rules[ki], _kw_rules[kj]
                                                _z = abs(_mean_ranks[_ri] - _mean_ranks[_rj]) / np.sqrt(
                                                    _N * (_N + 1) / 12 * (1/_ns[_ri] + 1/_ns[_rj])) if (_ns[_ri] > 0 and _ns[_rj] > 0) else 0
                                                from scipy.stats import norm
                                                _p_d = 2 * (1 - norm.cdf(abs(_z)))
                                                _dunn_pairs.append((_ri, _rj))
                                                _dunn_pvals.append(_p_d)
                                        _rej_d, _padj_d, _, _ = _mt2(_dunn_pvals, method="holm")
                                        _dunn_df = pd.DataFrame({
                                            "group1": [p[0] for p in _dunn_pairs],
                                            "group2": [p[1] for p in _dunn_pairs],
                                            "p-adj": _padj_d, "reject": _rej_d,
                                        })
                                        st.dataframe(_dunn_df, width='stretch')
                            else:
                                st.info(f"Not significant (p={_kw_p:.3g}): No evidence of differences between rules.")

            # ══════════════════════════════════════════════════════════════
            # TAB 6: Power Analysis
            # ══════════════════════════════════════════════════════════════
            with analytics_tabs[5]:
                st.subheader("Power Analysis & Sample Size Recommendation")
                st.caption("Assess whether the current sample size provides adequate statistical power to detect meaningful differences.")

                if not rpath or not os.path.exists(rpath):
                    st.info("Load RAW data first (results_*.txt required).")
                else:
                    _pw_metric = st.selectbox("Metric for power analysis", ["Reward", "Time", "PDR"],
                                              key="pw_metric_sel")
                    _pw_raw = _prep_metric(dfraw, _pw_metric) if 'dfraw' in dir() else pd.DataFrame()
                    if _pw_raw.empty:
                        st.warning(f"No data for {_pw_metric}.")
                    else:
                        _pw_tr, _pw_y, _ = _transform_for_metric(_pw_raw, _pw_metric)
                        try:
                            _pw_model = smf.ols(f"{_pw_y} ~ C(rule) + C(run)", data=_pw_tr).fit()
                            _pw_anova = sm.stats.anova_lm(_pw_model, typ=2)

                            # Extract effect sizes
                            _pw_ss_rule = float(_pw_anova.loc["C(rule)", "sum_sq"])
                            _pw_ss_resid = float(_pw_anova.loc["Residual", "sum_sq"])
                            _pw_ss_total = _pw_ss_rule + _pw_ss_resid
                            _pw_df_rule = int(_pw_anova.loc["C(rule)", "df"])
                            _pw_df_resid = int(_pw_anova.loc["Residual", "df"])
                            _pw_ms_resid = _pw_ss_resid / _pw_df_resid if _pw_df_resid > 0 else 1
                            _pw_eta2 = _pw_ss_rule / _pw_ss_total if _pw_ss_total > 0 else 0
                            _pw_f2 = _pw_eta2 / (1 - _pw_eta2) if _pw_eta2 < 1 else 0  # Cohen's f-squared

                            _pw_k = _pw_df_rule + 1  # number of groups
                            _pw_n_per = len(_pw_raw) // _pw_k if _pw_k > 0 else 30  # samples per group
                            _pw_F_obs = float(_pw_anova.loc["C(rule)", "F"])

                            # Post-hoc power using non-central F distribution
                            from scipy.stats import ncf, f as fdist
                            _pw_ncp = _pw_f2 * _pw_n_per * _pw_k  # non-centrality parameter
                            _pw_f_crit = fdist.ppf(0.95, _pw_df_rule, _pw_df_resid)
                            _pw_power = 1 - ncf.cdf(_pw_f_crit, _pw_df_rule, _pw_df_resid, _pw_ncp)

                            st.markdown("#### Post-hoc Power Analysis (Current Data)")
                            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                            col_p1.metric("eta-squared", f"{_pw_eta2:.4f}")
                            col_p2.metric("Cohen's f", f"{np.sqrt(_pw_f2):.4f}")
                            col_p3.metric("n per group", f"{_pw_n_per}")
                            col_p4.metric("Observed Power", f"{_pw_power:.3f}")

                            _pw_interp = "Adequate" if _pw_power >= 0.8 else "Marginal" if _pw_power >= 0.5 else "Low"
                            if _pw_power >= 0.8:
                                st.success(f"Power = {_pw_power:.3f} ({_pw_interp}). Current sample size is sufficient to detect the observed effect.")
                            elif _pw_power >= 0.5:
                                st.warning(f"Power = {_pw_power:.3f} ({_pw_interp}). Consider increasing totalSamples for reliable detection.")
                            else:
                                st.error(f"Power = {_pw_power:.3f} ({_pw_interp}). Sample size is insufficient. Increase totalSamples significantly.")

                            # Prospective: power curve
                            st.markdown("#### Prospective Power Curve")
                            st.caption("How many samples per group are needed to achieve a target power?")
                            _pw_target = st.slider("Target power", 0.50, 0.99, 0.80, 0.05, key="pw_target")

                            _pw_ns = np.arange(5, 201, 5)
                            _pw_powers = []
                            for _n in _pw_ns:
                                _ncp_n = _pw_f2 * _n * _pw_k
                                _df_resid_n = (_n - 1) * _pw_k  # approximate
                                _f_crit_n = fdist.ppf(0.95, _pw_df_rule, max(_df_resid_n, 1))
                                _pow_n = 1 - ncf.cdf(_f_crit_n, _pw_df_rule, max(_df_resid_n, 1), _ncp_n)
                                _pw_powers.append(_pow_n)

                            # Find required n
                            _pw_req_n = None
                            for _ni, _pi in zip(_pw_ns, _pw_powers):
                                if _pi >= float(_pw_target):
                                    _pw_req_n = _ni
                                    break

                            import plotly.graph_objects as go_pw
                            fig_pw = go_pw.Figure()
                            fig_pw.add_trace(go_pw.Scatter(
                                x=_pw_ns.tolist(), y=_pw_powers,
                                mode="lines+markers", marker=dict(size=4, color="#1abc9c"),
                                line=dict(width=2, color="#1abc9c"),
                                name="Power",
                            ))
                            fig_pw.add_hline(y=float(_pw_target), line_dash="dash", line_color="#e74c3c",
                                             annotation_text=f"Target={_pw_target:.2f}")
                            fig_pw.add_vline(x=_pw_n_per, line_dash="dot", line_color="#3498db",
                                             annotation_text=f"Current n={_pw_n_per}")
                            if _pw_req_n:
                                fig_pw.add_vline(x=_pw_req_n, line_dash="dash", line_color="#2ecc71",
                                                 annotation_text=f"Required n={_pw_req_n}")
                            fig_pw.update_layout(
                                xaxis_title="Samples per group (totalSamples)",
                                yaxis_title="Statistical Power",
                                yaxis_range=[0, 1.05],
                                height=400,
                                title=f"Power Curve ({_pw_metric}, Cohen's f = {np.sqrt(_pw_f2):.4f})",
                            )
                            st.plotly_chart(fig_pw, width='stretch')

                            if _pw_req_n:
                                st.info(f"To achieve power >= {_pw_target:.2f}, set **totalSamples >= {_pw_req_n}** (currently {_pw_n_per}).")
                            else:
                                st.info(f"Power >= {_pw_target:.2f} not achievable within n=200. Effect may be too small to detect reliably.")

                            # Effect size interpretation table
                            with st.expander("Effect Size Reference (Cohen 1988)"):
                                _eff_ref = pd.DataFrame({
                                    "Effect Size": ["Small", "Medium", "Large"],
                                    "Cohen's f": ["0.10", "0.25", "0.40"],
                                    "eta-squared": ["0.01", "0.06", "0.14"],
                                    "Your value": [f"f={np.sqrt(_pw_f2):.4f}", f"eta2={_pw_eta2:.4f}", ""],
                                })
                                st.table(_eff_ref)

                        except Exception as e_pw:
                            st.warning(f"Power analysis failed: {e_pw}")

            # ══════════════════════════════════════════════════════════════
            # TAB 7: Publication Export
            # ══════════════════════════════════════════════════════════════
            with analytics_tabs[6]:
                st.subheader("Publication-Quality Export")
                st.caption("Export ANOVA tables, CLD results, and figures in publication-ready formats.")

                if not rpath or not os.path.exists(rpath):
                    st.info("Load RAW data first.")
                else:
                    _ex_alpha = st.slider("Alpha for export analysis", 0.001, 0.1, 0.05, 0.001, key="ex_alpha")
                    _ex_alpha = float(_ex_alpha)

                    # Generate all analyses for export
                    _ex_metrics = ["Reward", "Time", "PDR"]
                    _ex_latex_parts = []
                    _ex_cld_parts = []

                    for _em in _ex_metrics:
                        _ex_raw = _prep_metric(dfraw, _em) if 'dfraw' in dir() else pd.DataFrame()
                        if _ex_raw.empty:
                            continue
                        _ex_tr, _ey, _ = _transform_for_metric(_ex_raw, _em)
                        try:
                            _ex_model = smf.ols(f"{_ey} ~ C(rule) + C(run)", data=_ex_tr).fit()
                            _ex_anova = sm.stats.anova_lm(_ex_model, typ=2)

                            # Add effect sizes
                            _ex_ss_total = _ex_anova["sum_sq"].sum()
                            _ex_anova["eta_sq"] = _ex_anova["sum_sq"] / _ex_ss_total
                            _ms_r = float(_ex_anova.loc["Residual", "sum_sq"] / _ex_anova.loc["Residual", "df"]) if _ex_anova.loc["Residual", "df"] > 0 else 0
                            _ex_anova["omega_sq"] = (_ex_anova["sum_sq"] - _ex_anova["df"] * _ms_r) / (_ex_ss_total + _ms_r)
                            _ex_anova.loc[_ex_anova["omega_sq"] < 0, "omega_sq"] = 0.0

                            # Format ANOVA table for LaTeX
                            _ex_at = _ex_anova.copy()
                            _ex_at.index = _ex_at.index.str.replace("C(rule)", "Rule", regex=False)
                            _ex_at.index = _ex_at.index.str.replace("C(run)", "Block (Run)", regex=False)
                            _ex_at = _ex_at.rename(columns={
                                "sum_sq": "SS", "df": "df", "F": "F", "PR(>F)": "p",
                                "eta_sq": "$\\eta^2$", "omega_sq": "$\\omega^2$"
                            })
                            _ex_at["MS"] = _ex_at["SS"] / _ex_at["df"]
                            _ex_at = _ex_at[["SS", "df", "MS", "F", "p", "$\\eta^2$", "$\\omega^2$"]]

                            _latex = _ex_at.to_latex(
                                float_format="%.4f", na_rep="--",
                                caption=f"ANOVA Table for {_em} (RCBD, $\\alpha$={_ex_alpha})",
                                label=f"tab:anova_{_em.lower()}",
                                escape=False,
                            )
                            _ex_latex_parts.append(f"% === {_em} ===\n{_latex}")

                            # CLD table
                            _is_asc = _em == "Time" or _em.startswith("PDR")
                            _ex_ph, _ex_emm = _emm_pairwise(_ex_model, _ex_tr, "rule", "run", _ey, _ex_alpha)
                            _ex_emm_sorted = _ex_emm.sort_values(ascending=_is_asc)
                            _ex_cld = cld_from_pairs(_ex_emm_sorted, _ex_ph, alpha=_ex_alpha)
                            if not _ex_cld.empty:
                                # Add original-scale means
                                _orig_means = _ex_raw.groupby("rule")["value"].mean()
                                _ex_cld["orig_mean"] = _ex_cld["rule"].map(_orig_means)
                                _cld_latex = _ex_cld.to_latex(
                                    float_format="%.4f", index=False,
                                    caption=f"CLD for {_em} (shared letter = no significant difference at $\\alpha$={_ex_alpha})",
                                    label=f"tab:cld_{_em.lower()}",
                                )
                                _ex_cld_parts.append(f"% === CLD: {_em} ===\n{_cld_latex}")
                        except Exception as e_ex:
                            _ex_latex_parts.append(f"% === {_em}: FAILED ({e_ex}) ===")

                    if _ex_latex_parts:
                        st.markdown("#### ANOVA Tables (LaTeX)")
                        _full_latex = "\n\n".join(_ex_latex_parts)
                        st.code(_full_latex, language="latex")
                        st.download_button("Download ANOVA LaTeX", _full_latex,
                                           file_name="anova_tables.tex", mime="text/plain", key="dl_anova_tex")

                    if _ex_cld_parts:
                        st.markdown("#### CLD Tables (LaTeX)")
                        _full_cld_latex = "\n\n".join(_ex_cld_parts)
                        st.code(_full_cld_latex, language="latex")
                        st.download_button("Download CLD LaTeX", _full_cld_latex,
                                           file_name="cld_tables.tex", mime="text/plain", key="dl_cld_tex")

                    # CSV export of all results
                    st.markdown("#### Combined CSV Export")
                    _csv_parts = []
                    for _em in _ex_metrics:
                        _ex_raw = _prep_metric(dfraw, _em) if 'dfraw' in dir() else pd.DataFrame()
                        if _ex_raw.empty:
                            continue
                        _summary = _ex_raw.groupby("rule")["value"].agg(["mean", "std", "count"]).reset_index()
                        _summary["metric"] = _em
                        _csv_parts.append(_summary)
                    if _csv_parts:
                        _csv_all = pd.concat(_csv_parts, ignore_index=True)
                        _csv_str = _csv_all.to_csv(index=False)
                        st.download_button("Download Summary CSV", _csv_str,
                                           file_name="analysis_summary.csv", mime="text/csv", key="dl_summary_csv")

                    # Raw data export
                    st.markdown("#### Raw Data Export (all metrics, long format)")
                    if 'dfraw' in dir() and not dfraw.empty:
                        _raw_csv = dfraw.to_csv(index=False)
                        st.download_button("Download Raw Data CSV", _raw_csv,
                                           file_name="raw_data_long.csv", mime="text/csv", key="dl_raw_csv")


# ------------------------------
# Data Tables tab (edit/read separation + filename labels)
# ------------------------------
with tabs[3]:
    st.subheader("CSV Tables (Edit/Save)")
    bp = st.session_state.base_path
    exp = st.session_state.selected_exp
    coord = st.session_state.selected_coord
    if not (bp and exp and coord):
        st.info("Select a scenario from the sidebar first.")
    else:
        coord_folder = Path(bp) / "scenarios" / exp / coord
        st.caption(str(coord_folder))
        csvs = list_coord_csvs(bp, exp, coord)
        # Exclude fire station master file from editable list
        csvs_editable = [p for p in csvs if os.path.basename(p) != "fire_stations.csv"]
        if not csvs_editable:
            st.info("No editable CSV files found.")
        else:
            labels = {p: os.path.basename(p) for p in csvs_editable}
            target = st.selectbox("CSV to Edit", options=list(labels.keys()), format_func=lambda p: labels[p])
            df = read_csv_smart(target)
            edit = st.data_editor(df, width='stretch', num_rows="dynamic", height=400)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("💾 Save (auto backup)"):
                    write_csv_smart(edit, target)
                    st.success("Save complete")
            with c2:
                if st.button("🔄 Refresh"):
                    st.rerun()
            with c3:
                yaml_path = find_yaml_in_coord(bp, exp, coord)
                if yaml_path and st.button("▶️ Re-run with Modified Values"):
                    try:
                        from orchestrator import Orchestrator
                        with st.spinner("Running simulation..."):
                            orc = Orchestrator(base_path=bp)
                            result = orc.run_simulation(config_path=yaml_path)
                        if result["ok"]:
                            st.success("✅ Simulation complete!")
                            st.write(f"• Log file: `{result['log_file']}`")
                        else:
                            st.error(f"❌ Execution failed (code: {result['returncode']})")
                            with st.expander("stdout"):
                                st.text(result.get("stdout", ""))
                            with st.expander("stderr"):
                                st.text(result.get("stderr", ""))
                    except Exception as e:
                        st.error("Simulation execution error")
                        st.exception(e)

        st.markdown("#### Hospital Master (Excel, read-only)")
        hdf = read_excel_hospital(bp)
        if hdf is not None and not hdf.empty:
            st.dataframe(hdf.head(200), width='stretch', height=280)
        else:
            st.caption("Hospital Excel data not found or load failed")

        st.markdown("#### Fire Station Master (read-only)")
        global_center_csv = Path(bp) / "scenarios" / "fire_stations.csv"
        if global_center_csv.is_file():
            cdf = read_csv_smart(str(global_center_csv))
            st.dataframe(cdf.head(200), width='stretch', height=260)
        else:
            st.caption("Fire station CSV not found")

# ------------------------------
# Generate tab
# ------------------------------
def parse_env_kv(text: str):
    env = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

if "gen_state" not in st.session_state:
    st.session_state.gen_state = {}
if "env_txt" not in st.session_state:
    st.session_state.env_txt = ""
if "env_txt2" not in st.session_state:
    st.session_state.env_txt2 = ""

# ------------------------------
# Generate tab (independent, last tab)
# ------------------------------
# ──────────────────────────────────────────────────────────────────────────────
# Rerun tab (re-run existing scenario)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Re-run Existing Scenario")
    st.info("💡 This tab operates **independently** from the sidebar. Select an existing scenario to modify parameters and re-run.")

    # ─────────────────────────────────────────────────────────────────
    # Rerun tab dedicated base_path input
    # ─────────────────────────────────────────────────────────────────
    if "rerun_base_path" not in st.session_state:
        st.session_state.rerun_base_path = CLOUD_BASE_PATH if IS_CLOUD else DEFAULT_LOCAL_BASE_PATH
    else:
        if IS_CLOUD and st.session_state.rerun_base_path != CLOUD_BASE_PATH:
            st.session_state.rerun_base_path = CLOUD_BASE_PATH


    st.markdown("---")
    st.markdown("### Project Path Setup")

    col_path, col_btn = st.columns([4, 1])
    with col_path:
        rerun_bp_input = st.text_input(
            "🗂️ Project Path (base_path)",
            value=st.session_state.rerun_base_path,
            placeholder="e.g. C:\\Users\\USER\\MCI_ADV",
            help="Enter the project root path containing the scenarios folder",
            key="rerun_bp_input",
            disabled=IS_CLOUD,
        )
        if IS_CLOUD:
            st.caption(f"☁️ Cloud: fixed to `{CLOUD_BASE_PATH}`.")

    with col_btn:
        st.write("")  # alignment spacer
        st.write("")  # alignment spacer
        if (not IS_CLOUD) and st.button("✅ Confirm Path", key="rerun_check_path"):
            st.session_state.rerun_base_path = rerun_bp_input


    bp_rerun = st.session_state.rerun_base_path

    # Path validation
    if not bp_rerun:
        st.warning("⚠️ Enter the project path above and click **✅ Confirm Path**.")
        st.stop()

    if not base_ok(bp_rerun):
        st.error(f"❌ Invalid path: `{bp_rerun}`")
        st.caption("• Check if the path exists\n• Check if the `scenarios` folder is present")
        st.stop()

    st.success(f"✅ Valid path: `{bp_rerun}`")

    # ─────────────────────────────────────────────────────────────────
    # Load Orchestrator
    # ─────────────────────────────────────────────────────────────────
    try:
        from orchestrator import Orchestrator
    except Exception as e:
        st.error("❌ `src/sce_src/orchestrator.py` not found.")
        st.exception(e)
        st.stop()

    # ─────────────────────────────────────────────────────────────────
    # Select experiment folder and coordinate folder
    # ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Select Scenario")

    # Build experiment/coordinate lists
    exps_rerun = list_experiments_any(bp_rerun)
    if not exps_rerun:
        st.warning("⚠️ No experiments found in scenarios folder.")
        st.stop()

    sel_exp_rerun = st.selectbox(
        "📂 Select Experiment Folder",
        options=exps_rerun,
        key="sel_exp_rerun",
        help="Select an experiment folder from scenarios"
    )

    coords_rerun = list_coords_from_scenarios(bp_rerun, sel_exp_rerun) if sel_exp_rerun else []
    if not coords_rerun:
        st.warning(f"⚠️ No coordinate folders in experiment `{sel_exp_rerun}`.")
        st.stop()

    sel_coord_rerun = st.selectbox(
        "📍 Select Coordinate Folder",
        options=coords_rerun,
        key="sel_coord_rerun",
        help="Select a coordinate folder from the experiment"
    )

    # ─────────────────────────────────────────────────────────────────
    # Read YAML file and parameter editing UI
    # ─────────────────────────────────────────────────────────────────
    if sel_exp_rerun and sel_coord_rerun:
        cfg_path_rerun = os.path.join(bp_rerun, "scenarios", sel_exp_rerun, sel_coord_rerun, f"config_{sel_coord_rerun}.yaml")

        if not os.path.exists(cfg_path_rerun):
            st.error(f"❌ CONFIG file not found: `{cfg_path_rerun}`")
            st.stop()

        st.success(f"✅ CONFIG file: `{os.path.basename(cfg_path_rerun)}`")

        try:
            with open(cfg_path_rerun, "r", encoding="utf-8") as f:
                yaml_data_rerun = yaml.safe_load(f)

            # ─────────────────────────────────────────────────────────────────
            # Display current settings
            # ─────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Current Config")

            with st.expander("View Current Scenario Config", expanded=False):
                st.json(yaml_data_rerun)

            # ─────────────────────────────────────────────────────────────────
            # Parameter editing UI
            # ─────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Edit Parameters")
            st.caption("⚠️ Changing departure time, incident size, or coordinates requires scenario regeneration (API re-call)")

            col1, col2 = st.columns(2)

            # Ambulance parameters
            with col1:
                st.markdown("**🚑 Ambulance**")
                amb_cfg_rerun = yaml_data_rerun.get('entity_info', {}).get('ambulance', {})
                is_use_time_amb_rerun = st.checkbox(
                    "Use API Duration",
                    value=amb_cfg_rerun.get('is_use_time', True),
                    key="rerun_is_use_time",
                    help="True: use API duration. False: compute time from distance / velocity."
                )
                amb_velocity_rerun = st.number_input(
                    "Ambulance Speed (km/h)",
                    value=float(amb_cfg_rerun.get('velocity', 40)),
                    min_value=1.0,
                    step=1.0,
                    key="rerun_amb_velocity"
                )
                amb_handover_rerun = st.number_input(
                    "Patient Handover Time (min)",
                    value=float(amb_cfg_rerun.get('handover_time', 10.0)),
                    min_value=0.0,
                    step=0.5,
                    key="rerun_amb_handover"
                )
                duration_coeff_rerun = st.number_input(
                    "API Duration Weight",
                    value=float(amb_cfg_rerun.get('duration_coeff', 1.0)),
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    format="%.1f",
                    key="rerun_duration_coeff",
                    help="Coefficient multiplied with the API duration (default: 1.0)."
                )

            # UAV parameters
            with col2:
                st.markdown("**🛩️ UAV**")
                uav_cfg_rerun = yaml_data_rerun.get('entity_info', {}).get('uav', {})
                uav_velocity_rerun = st.number_input(
                    "UAV Speed (km/h)",
                    value=float(uav_cfg_rerun.get('velocity', 80)),
                    min_value=1.0,
                    step=1.0,
                    key="rerun_uav_velocity"
                )
                uav_handover_rerun = st.number_input(
                    "Patient Handover Time (min)",
                    value=float(uav_cfg_rerun.get('handover_time', 15.0)),
                    min_value=0.0,
                    step=0.5,
                    key="rerun_uav_handover"
                )

            st.markdown("**🏥 Hospital**")
            col3, col4 = st.columns(2)
            with col3:
                hosp_cfg_rerun = yaml_data_rerun.get('entity_info', {}).get('hospital', {})
                max_send_coeff_rerun = st.text_input(
                    "hospital_max_send_coeff",
                    value=str(hosp_cfg_rerun.get('max_send_coeff', [1, 1])).strip('[]'),
                    key="rerun_max_send_coeff",
                    help="e.g. 1.1, 1.0"
                )

            with col4:
                run_cfg_rerun = yaml_data_rerun.get('run_setting', {})
                total_samples_rerun = st.number_input(
                    "Simulation Iterations",
                    value=int(run_cfg_rerun.get('totalSamples', 30)),
                    min_value=1,
                    step=1,
                    key="rerun_total_samples"
                )

            # ─────────────────────────────────────────────────────────────────
            # Execute button
            # ─────────────────────────────────────────────────────────────────
            st.markdown("---")
            if st.button("▶️ Apply Changes & Run Simulation", key="btn_rerun_execute"):
                try:
                    # Create YAML backup (with timestamp) - before modification
                    import shutil
                    backup_path_rerun = cfg_path_rerun.replace(".yaml", f"_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml")
                    shutil.copy(cfg_path_rerun, backup_path_rerun)
                    st.info(f"📦 Original YAML backup: `{os.path.basename(backup_path_rerun)}`")

                    # Read YAML file as string and modify directly (preserving comments and format)
                    with open(cfg_path_rerun, "r", encoding="utf-8") as f:
                        yaml_text_rerun = f.read()

                    # Parse max_send_coeff
                    try:
                        coeff_list_rerun = [float(x.strip()) for x in max_send_coeff_rerun.split(',')]
                        coeff_str_rerun = "[" + ", ".join(str(c) for c in coeff_list_rerun) + "]"
                    except:
                        st.warning("max_send_coeff format error, keeping original value")
                        coeff_str_rerun = None

                    # Replace values only via regex (preserving comments and format)
                    # ambulance velocity
                    yaml_text_rerun = re.sub(
                        r'(ambulance:.*?velocity:\s*)[\d.]+',
                        rf'\g<1>{amb_velocity_rerun}',
                        yaml_text_rerun, flags=re.DOTALL
                    )
                    # ambulance handover_time
                    yaml_text_rerun = re.sub(
                        r'(ambulance:.*?handover_time:\s*)[\d.]+',
                        rf'\g<1>{amb_handover_rerun}',
                        yaml_text_rerun, flags=re.DOTALL
                    )
                    # ambulance is_use_time
                    yaml_text_rerun = re.sub(
                        r'(ambulance:.*?is_use_time:\s*)(True|False|true|false)',
                        rf'\g<1>{"True" if is_use_time_amb_rerun else "False"}',
                        yaml_text_rerun, flags=re.DOTALL
                    )
                    # ambulance duration_coeff
                    # Update duration_coeff if exists in YAML, otherwise add it
                    if re.search(r'ambulance:.*?duration_coeff:', yaml_text_rerun, flags=re.DOTALL):
                        # Update existing field
                        yaml_text_rerun = re.sub(
                            r'(ambulance:.*?duration_coeff:\s*)[\d.]+',
                            rf'\g<1>{duration_coeff_rerun}',
                            yaml_text_rerun, flags=re.DOTALL
                        )
                    else:
                        # Add field after is_use_time if not present
                        yaml_text_rerun = re.sub(
                            r'(ambulance:.*?is_use_time:\s*(?:True|False|true|false)[^\n]*\n)',
                            rf'\g<1>    duration_coeff: {duration_coeff_rerun} # API duration weight (default: 1.0, adjust for environmental factors)\n',
                            yaml_text_rerun, flags=re.DOTALL
                        )
                    # uav velocity
                    yaml_text_rerun = re.sub(
                        r'(uav:.*?velocity:\s*)[\d.]+',
                        rf'\g<1>{uav_velocity_rerun}',
                        yaml_text_rerun, flags=re.DOTALL
                    )
                    # uav handover_time
                    yaml_text_rerun = re.sub(
                        r'(uav:.*?handover_time:\s*)[\d.]+',
                        rf'\g<1>{uav_handover_rerun}',
                        yaml_text_rerun, flags=re.DOTALL
                    )

                    if coeff_str_rerun:
                        yaml_text_rerun = re.sub(
                            r'max_send_coeff:\s*\[[\d.,\s]+\]',
                            f'max_send_coeff: {coeff_str_rerun}',
                            yaml_text_rerun
                        )

                    yaml_text_rerun = re.sub(
                        r'(totalSamples:\s*)[\d]+',
                        rf'\g<1>{total_samples_rerun}',
                        yaml_text_rerun
                    )

                    # Save YAML - preserve original format perfectly
                    with open(cfg_path_rerun, "w", encoding="utf-8") as f:
                        f.write(yaml_text_rerun)

                    st.success("✅ YAML file updated!")

                    # Run simulation
                    with st.spinner("Running simulation..."):
                        orc_rerun = Orchestrator(base_path=bp_rerun)
                        res_rerun = orc_rerun.run_simulation(config_path=cfg_path_rerun)

                    if res_rerun["ok"]:
                        st.success("✅ Simulation complete!")
                        st.write(f"• Exp ID: `{res_rerun['exp_id']}`")
                        st.write(f"• Coord: `{res_rerun['coord']}`")
                        st.write(f"• Log file: `{res_rerun['log_file']}`")
                        st.write(f"• Summary CSV has been auto-updated")
                        st.caption("💡 Check results in the Scenarios/Maps tabs.")
                    else:
                        st.error(f"❌ Simulation failed (code: {res_rerun['returncode']})")
                        with st.expander("stdout"):
                            st.text(res_rerun.get("stdout", ""))
                        with st.expander("stderr"):
                            st.text(res_rerun.get("stderr", ""))

                except Exception as e_rerun:
                    st.error("❌ Simulation execution error")
                    st.exception(e_rerun)

        except Exception as e_yaml_rerun:
            st.error(f"❌ YAML file read failed: {e_yaml_rerun}")
            st.exception(e_yaml_rerun)
