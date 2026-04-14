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
    st.header("⚙️ Settings")

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

            # -- Basemap selection (light only) + theme toggle --
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                basemap_choice_ui = st.selectbox(
                    "Light Tile",
                    ["OpenStreetMap","CartoDB Positron"],
                    index=0,
                    key="mini_basemap_light",
                )
            with col_m2:
                theme_choice = st.radio(
                    "Theme",
                    ["Light", "Dark"],
                    index=0,
                    horizontal=True,
                    key="mini_theme",
                )

            # -- folium-based render (same display logic: single point) --
            try:
                import folium
                from streamlit_folium import st_folium

                # UI label -> folium tile name mapping
                tile_map_light = {
                    "CartoDB Positron": "CartoDB positron",
                    "OpenStreetMap": "OpenStreetMap",
                }
                # Dark is fixed
                tile_dark = "CartoDB dark_matter"

                chosen_tile = tile_dark if theme_choice == "Dark" else tile_map_light[basemap_choice_ui]

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Pretendard', -apple-system, sans-serif !important;
}

/* -- Background gradient -- */
.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #111827 40%, #0f172a 100%);
}

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111d35 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.08) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
    font-size: 0.85rem;
}

/* -- Main title -- */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    padding-bottom: 4px;
    color: #e2e8f0 !important;
}
.gradient-text {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* -- Subheader -- */
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(56, 189, 248, 0.15);
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}

/* -- Tab bar -- */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid rgba(56, 189, 248, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 500;
    color: #94a3b8 !important;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #1e40af) !important;
    color: #e0f2fe !important;
    box-shadow: 0 2px 12px rgba(59, 130, 246, 0.25);
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(30, 58, 95, 0.4);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* -- Button -- */
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%) !important;
    color: #e0f2fe !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
}
.stButton > button:active { transform: translateY(0); }

/* -- Input fields -- */
[data-baseweb="input"],
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}
[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(59, 130, 246, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
}

/* -- Dropdown menu -- */
[data-baseweb="popover"] {
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    overflow: hidden;
}
[data-baseweb="menu"] { background: #111827 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-radius: 14px !important;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.18) !important;
}

/* -- Metric card -- */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(56, 189, 248, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: #e0f2fe !important;
    font-weight: 700 !important;
}

/* -- DataFrame -- */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.08);
}

/* -- Divider -- */
hr {
    border-color: rgba(56, 189, 248, 0.1) !important;
    margin: 24px 0 !important;
}

/* -- Alert messages -- */
.stAlert, [data-testid="stAlert"] { border-radius: 10px !important; }

/* -- Checkbox/radio hover -- */
.stCheckbox label:hover, .stRadio label:hover { color: #38bdf8 !important; }

/* -- Scrollbar -- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.3); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #1e40af, #38bdf8);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1><span>🚑</span> <span class="gradient-text">MCI Disaster Simulation Dashboard</span></h1>', unsafe_allow_html=True)

tabs = st.tabs(["Maps", "Scenarios", "Analytics", "Data Tables", "Rerun"])

# ------------------------------
# Scenarios tab
# ------------------------------
with tabs[1]:
    st.subheader("📁 Selected Scenario")
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

        with st.expander("📝 View Experiment Summary", expanded=True):
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
            st.button("📂 Load Log Files", key="load_logs_btn_disabled", disabled=True, help="Log viewer is disabled for 101+ iterations")
            logs = None
        else:
            # Performance optimization: load logs only on button click (improves other tab loading speed)
            if st.button("📂 Load Log Files", key="load_logs_btn", help="Click to load logs"):
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


# ------------------------------
# Maps tab (multi-select + UAV dispatch/transport toggle + enhanced legend)
# ------------------------------
with tabs[0]:
    st.subheader("🗺️ Map Visualization")
    bp   = st.session_state.base_path
    exp  = st.session_state.selected_exp
    coord= st.session_state.selected_coord

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
        # Performance optimization: prevent unnecessary re-renders with key
        st_folium(m, width=None, height=690, key="main_map", returned_objects=[])


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
    st.subheader("📊 RAW Result Analysis (results_{coord}.txt)")

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

        if st.button("📊 Load Analysis Data", key="load_analytics_btn", help="Parse and analyze RAW results"):
            st.session_state.analytics_loaded = True

        if not st.session_state.get("analytics_loaded", False):
            st.caption("💡 Click 'Load Analysis Data' above to view analysis. (Disabled by default for faster tab loading)")
        else:
            # -- RAW: Toggle display of selected metrics only
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
                        st.markdown(f"#### ▶ RAW Table — **{m}** (per run)")
                        st.dataframe(raw_tables[m], width='stretch')
                else:
                    st.warning("No readable blocks found in RAW (results_*.txt).")
            else:
                st.warning("RAW (results_*.txt) file not found.")

            st.divider()

            # ===== (existing) STAT summary analysis section =====
            st.subheader("📈 STAT Summary Analysis (_stat.txt)")
            st.info(
                "📂 results/exp_YYYYMMDD_HHMMSS/(lat,lon)/results_{coord}.txt (Raw), results_{coord}_stat.txt (Stats)\n\n"
                "- **Reward**: Sum of survival probabilities\n- **Time**: Elapsed time\n- **PDR**\n- **w.o.G**: Metrics excluding Green patients"
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

                st.markdown("#### 🏆 Scenario Ranking (Sort by)")
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


            # ── ANOVA Suite (when raw file exists)
            st.markdown("#### 🧪 ANOVA (One-way / RCBD / Reduced Factorial)")

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


            def games_howell_fallback(df, grp, yvar, alpha=0.05):
                """Approximate with Welch t + Holm when pingouin is unavailable."""
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
                Monotone CLD: traverse sorted means top-to-bottom,
                assign same letter only when all current group members are 'not significant'.
                Move to next letter if any pair is 'significant'.
                - means: index=group name (rule), values=mean on same scale as post-hoc
                        (for RCBD, sort by y_adj mean recommended; see application below)
                - pair_tbl: must contain ['group1','group2'] and ['reject'] or ['p-adj']
                Result: single letter (A,B,C,...) per group -- prevents zigzag, ensures interval-type grouping.
                """
                # If empty table, assign all A
                if pair_tbl is None or pair_tbl.empty or len(means) <= 1:
                    return pd.DataFrame({"rule": means.index, "mean": means.values, "CLD": ["A"]*len(means)})

                ph = pair_tbl.copy()
                # Calculate non-significance status
                if "reject" in ph.columns:
                    ph["ns"] = ~ph["reject"].astype(bool)
                else:
                    ph["ns"] = ph["p-adj"] >= alpha

                # Fast lookup dict: (a,b)->non-significant(True/False), missing info treated conservatively as 'significant'
                key = lambda a,b: tuple(sorted((a,b)))
                ns_map = { key(r["group1"], r["group2"]): bool(r["ns"]) for _, r in ph.iterrows() }

                ordered = list(means.index)  # assumes already sorted externally
                def is_ns(a, b):
                    return ns_map.get(key(a,b), False)  # missing info treated as False(=significant) to prevent excessive merging

                letters = {}
                current_letter = "A"
                current_members = [ordered[0]]
                letters[ordered[0]] = current_letter

                for g in ordered[1:]:
                    # Keep same letter if non-significant with all current group members
                    if all(is_ns(g, m) for m in current_members):
                        letters[g] = current_letter
                        current_members.append(g)
                    else:
                        # Advance to next letter (monotone increase)
                        nxt = ord(current_letter) + 1
                        current_letter = chr(nxt) if nxt <= ord('Z') else current_letter  # keep Z if exceeded
                        letters[g] = current_letter
                        current_members = [g]

                return pd.DataFrame({
                    "rule": ordered,
                    "mean": [means[g] for g in ordered],
                    "CLD":  [letters[g] for g in ordered],
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
                                # Reduced factorial: main effects + 2-way interactions only (no 3/4-way)
                                formula = (f"{yvar} ~ C(Phase) + C(RedPolicy) + C(RedAction) + C(YellowAction)"
                                           " + C(Phase):C(RedPolicy) + C(Phase):C(RedAction) + C(Phase):C(YellowAction)"
                                           " + C(RedPolicy):C(RedAction) + C(RedPolicy):C(YellowAction)"
                                           " + C(RedAction):C(YellowAction)")
                                st.caption("Model: main effects + all 2-way interactions (3/4-way excluded for power)")

                            model = smf.ols(formula, data=d).fit()
                            anova_tbl = sm.stats.anova_lm(model, typ=2)

                            # Include Total + eta-squared
                            out = make_total_row(anova_tbl, d[yvar])
                            st.dataframe(out, width='stretch')

                            # Significant effects summary
                            alpha = st.slider("Significance Level (alpha)", 0.001, 0.1, 0.05, 0.001)
                            sig = out[(out.index!="Total") & (out["PR(>F)"] < alpha)].sort_values("PR(>F)")
                            if not sig.empty:
                                st.markdown("##### 📌 Interpretation Summary")
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
                        

                            # Homogeneity of variance (by rule). For RCBD, test with block-residualized values
                            try:
                                if mode.startswith("One-way"):
                                    df_lev = d
                                else:
                                    df_lev = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                groups = [g[yvar].values for _, g in df_lev.groupby("rule")]
                                p_lev = sps.levene(*groups, center="median").pvalue
                                st.write(f"Levene(Brown–Forsythe): p={p_lev:.3g}")
                            except Exception:
                                p_lev = np.nan

                            # Added right after p_shap, p_lev computation
                            p_shap = _scalar(p_shap, default=1.0)  # on failure, default to 'normality pass'
                            p_lev  = _scalar(p_lev,  default=np.nan)
                            alpha  = _scalar(alpha,  default=0.05)

                            # ===== Post-hoc tests & CLD =====
                            st.markdown("##### Post-hoc Tests")
                            posthoc = pd.DataFrame(); explain = ""

                            def _small_is_better(m: str) -> bool:
                                # Time, PDR(incl. woG)=smaller is better / Reward types=larger is better
                                return (m == "Time") or m.startswith("PDR")

                            # --- Finalize post-hoc input and CLD mean (Series) ---
                            if mode == "One-way + Block(run) (RCBD recommended)":
                                dd = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                dd_work = _make_dd_work(dd, yvar)               # ensure analysis column '__y__' is 1D
                                y_post  = dd_work["__y__"]
                                grp_post= dd_work["rule"]
                                means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                    ascending=_small_is_better(metric)
                                )
                                # Levene also based on dd_work (safer)
                                lev_groups = [g["__y__"].values for _, g in dd_work.groupby("rule")]
                            else:
                                dd_work = _make_dd_work(d, yvar)                # ensure 1D for One-way as well
                                y_post  = dd_work["__y__"]
                                grp_post= dd_work["rule"]
                                means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                    ascending=_small_is_better(metric)
                                )
                                lev_groups = [g["__y__"].values for _, g in dd_work.groupby("rule")]

                            try:
                                if len(lev_groups) >= 2 and all(len(x) > 1 for x in lev_groups):
                                    p_lev = sps.levene(*lev_groups, center="median").pvalue
                                else:
                                    p_lev = np.nan
                                st.write(f"Levene(Brown–Forsythe): p={_scalar(p_lev):.3g}")
                            except Exception:
                                p_lev = np.nan


                            # --- Post-hoc: Games-Howell (robust to non-normality & heteroscedasticity) ---
                            try:
                                import pingouin as pg
                                gh = pg.pairwise_gameshowell(dv="__y__", between="rule", data=dd_work)
                                posthoc = gh.rename(columns={"A":"group1","B":"group2","pval":"p-adj"})
                                posthoc["reject"] = posthoc["p-adj"] < alpha
                                explain = "Games–Howell"
                            except Exception:
                                posthoc = games_howell_fallback(
                                    pd.DataFrame({"rule": grp_post.values, "y": y_post.values}),
                                    "rule", "y", alpha=alpha
                                )
                                explain = "Welch t-tests + Holm correction (Games–Howell fallback)"

                            # --- Output results & CLD ---
                            if not posthoc.empty:
                                st.caption(explain)
                                st.dataframe(posthoc, width='stretch')

                                ph = posthoc.copy()
                                if ("p-adj" not in ph.columns) and ("reject" not in ph.columns):
                                    st.info("No p-value information available for CLD.")
                                else:
                                    cld = cld_from_pairs(means_for_cld, ph, alpha=alpha)
                                    st.markdown("##### CLD (same letter = no significant difference, A = best)")
                                    st.caption("⚠️ CLD uses a greedy monotone algorithm; letter assignments may vary with group ordering. "
                                               "Interpret letters as approximate groupings — always check pairwise p-values for precise conclusions.")
                                    st.dataframe(cld, width='stretch')

                                    # Top candidates (‘A’ group) -- sorted by metric direction
                                    st.markdown(f"#### ✅ Top Candidates (**{metric}, A=Best**)")
                                    top = cld[cld["CLD"]=="A"].sort_values("mean", ascending=_small_is_better(metric))
                                    st.dataframe(top, width='stretch')

                                    # ================== A-Group Intersection (Reward, Time, PDR, RCBD-based) ==================
                                    st.markdown("### 🔗 A-Group Intersection (Reward ∩ Time(asc) ∩ PDR(asc), RCBD)")

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
                                        RCBD: y ~ C(rule) + C(run), then Games-Howell on block-adjusted y*.
                                        CLD via cld_from_pairs (monotone letter algorithm).
                                        prefer_small_is_A=True  → ascending sort (A = smallest = Best)
                                        prefer_small_is_A=False → descending sort (A = largest = Best)
                                        """
                                        from scipy import stats as sps

                                        posthoc = pd.DataFrame()
                                        explain = ""

                                        # --- Block-adjust y* for post-hoc input (1D guaranteed)
                                        dd = block_adjust(d, yvar, block_col="run").rename(columns={"y_adj": yvar})
                                        dd_work = _make_dd_work(dd, yvar)
                                        y_post, grp_post = dd_work["__y__"], dd_work["rule"]
                                        means_for_cld = _means_series(dd_work, "rule", "__y__").sort_values(
                                            ascending=prefer_small_is_A
                                        )

                                        # --- Post-hoc: Games-Howell (robust to non-normality & heteroscedasticity)
                                        try:
                                            import pingouin as pg
                                            gh = pg.pairwise_gameshowell(dv="__y__", between="rule", data=dd_work)
                                            posthoc = gh.rename(columns={"A":"group1","B":"group2","pval":"p-adj"})
                                            posthoc["reject"] = posthoc["p-adj"] < alpha
                                            explain = "Games–Howell (RCBD, y*)"
                                        except Exception:
                                            posthoc = games_howell_fallback(
                                                pd.DataFrame({"rule": grp_post.values, "y": y_post.values}),
                                                "rule", "y", alpha=alpha
                                            )
                                            explain = "Welch t-tests + Holm correction (RCBD, y*)"

                                        # --- CLD computation
                                        if posthoc.empty or (("p-adj" not in posthoc.columns) and ("reject" not in posthoc.columns)):
                                            return means_for_cld, pd.DataFrame(), explain

                                        cld = cld_from_pairs(means_for_cld, posthoc, alpha=alpha)
                                        return means_for_cld, cld, explain

                                    with st.expander("🔍 A-Group Intersection (Reward↑, Time↓, PDR↓)", expanded=True):
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
                                            A_rew = set(cld_rew.loc[cld_rew["CLD"]=="A","rule"]) if not cld_rew.empty else set()
                                            disp_rew = d_rew.groupby("rule")["value"].mean().rename("Reward_mean(orig)")

                                        # --- Time (smaller = A) ---
                                        d_time = _prep_metric(dfraw, "Time")
                                        if d_time.empty:
                                            st.info("No Time data.")
                                            A_time, disp_time = set(), pd.Series(dtype=float)
                                        else:
                                            d_time_tr, y_time, _ = _transform_for_metric(d_time, "Time")
                                            means_time, cld_time, _ = _rcbd_posthoc_cld(d_time_tr, y_time, alpha=alpha_int, prefer_small_is_A=True)
                                            A_time = set(cld_time.loc[cld_time["CLD"]=="A","rule"]) if not cld_time.empty else set()
                                            disp_time = d_time.groupby("rule")["value"].mean().rename("Time_mean(orig)")

                                        # --- PDR (smaller = A; logit analysis, display uses original scale mean) ---
                                        d_pdr = _prep_metric(dfraw, "PDR")
                                        if d_pdr.empty:
                                            st.info("No PDR data.")
                                            A_pdr, disp_pdr = set(), pd.Series(dtype=float)
                                        else:
                                            d_pdr_tr, y_pdr, _ = _transform_for_metric(d_pdr, "PDR")
                                            means_pdr, cld_pdr, _ = _rcbd_posthoc_cld(d_pdr_tr, y_pdr, alpha=alpha_int, prefer_small_is_A=True)
                                            A_pdr = set(cld_pdr.loc[cld_pdr["CLD"]=="A","rule"]) if not cld_pdr.empty else set()
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


# ------------------------------
# Data Tables tab (edit/read separation + filename labels)
# ------------------------------
with tabs[3]:
    st.subheader("🧾 CSV Tables (Edit/Save)")
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
                    st.experimental_rerun()
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
    st.subheader("🔄 Re-run Existing Scenario")
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
    st.markdown("### 📁 Project Path Setup")

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
            st.markdown("### ⚙️ Current Config")

            with st.expander("📋 View Current Scenario Config", expanded=False):
                st.json(yaml_data_rerun)

            # ─────────────────────────────────────────────────────────────────
            # Parameter editing UI
            # ─────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔧 Edit Parameters")
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
