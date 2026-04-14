"""
BatchLab: batch scenario generator + multi-coordinate result visualizer.
- Does not touch the existing Streamlit dashboard.
- Reuses Orchestrator for generation/run, adds batch + comparison UX.
"""
import os
import re
import json
import math
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import product

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests
import folium
from streamlit_folium import st_folium


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

try:
    from orchestrator import Orchestrator
except Exception as e:  # pragma: no cover
    st.error("Failed to import Orchestrator. Check `src/sce_src/orchestrator.py`.")
    st.exception(e)
    st.stop()


# -------------------------------------------------
# Constants and helpers
# -------------------------------------------------
KST = timezone(timedelta(hours=9))
CLOUD_BASE_PATH = _detect_cloud_base_path()
IS_CLOUD = bool(CLOUD_BASE_PATH)
DEFAULT_LOCAL_BASE_PATH = str(REPO_ROOT) if (REPO_ROOT / "scenarios").is_dir() else ""

RAW_BLOCK_NAMES = ["Reward", "Time", "PDR", "Reward_woG", "PDR_woG"]
_RAW_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def base_ok(bp: str) -> bool:
    return bool(bp) and os.path.isdir(os.path.join(bp, "scenarios"))


def parse_env_kv(text: str) -> dict:
    env = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def get_kakao_key_from_secrets() -> str:
    try:
        for k in ("KAKAO_REST_API_KEY", "KAKAO_API_KEY", "KAKAO_KEY"):
            if k in st.secrets:
                return str(st.secrets[k]).strip()
        if "kakao" in st.secrets:
            sec = st.secrets["kakao"]
            for k in ("rest_api_key", "api_key", "key"):
                if k in sec:
                    return str(sec[k]).strip()
    except Exception:
        pass
    return ""


def normalize_search_result(doc, search_type: str):
    if search_type == "keyword":
        return {
            "place_name": doc.get("place_name", ""),
            "address_name": doc.get("address_name", ""),
            "x": float(doc["x"]),
            "y": float(doc["y"]),
            "search_type": "keyword",
        }
    display_name = ""
    if "road_address" in doc and doc["road_address"]:
        display_name = doc["road_address"].get("building_name", "")
    display_name = display_name or doc.get("address_name", "")
    return {
        "place_name": f"{display_name} (address)",
        "address_name": doc.get("address_name", ""),
        "x": float(doc["x"]),
        "y": float(doc["y"]),
        "search_type": "address",
    }


def perform_address_search(search_query: str, api_key: str):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key.strip()}"}
    params = {"query": search_query, "analyze_type": "similar", "size": 10}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        return False, [], f"HTTP {resp.status_code}: {msg}"
    docs = resp.json().get("documents", [])
    normalized = [normalize_search_result(doc, "address") for doc in docs]
    return True, normalized, ""


def _split_key_vals(line: str):
    m = re.search(_RAW_FLOAT, line)
    if not m:
        return line.strip(), []
    key = line[: m.start()].strip().rstrip(",")
    vals = [float(x) for x in re.findall(_RAW_FLOAT, line[m.start() :])]
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


@st.cache_data(ttl=600, show_spinner=False)
def parse_raw_results(raw_path: str) -> pd.DataFrame:
    if not (raw_path and os.path.exists(raw_path)):
        return pd.DataFrame(
            columns=[
                "rule",
                "Phase",
                "RedPolicy",
                "RedAction",
                "YellowAction",
                "run",
                "metric",
                "value",
            ]
        )

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
        return pd.DataFrame(columns=["rule", "Phase", "RedPolicy", "RedAction", "YellowAction", "run", "metric", "value"])

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


@st.cache_data(ttl=300, show_spinner=False)
def collect_results(results_root: str, selected_exps: list[str]) -> pd.DataFrame:
    root = Path(results_root)
    if not root.exists():
        return pd.DataFrame()
    exp_dirs = [d for d in root.iterdir() if d.is_dir() and (not selected_exps or d.name in selected_exps)]
    frames = []
    for exp_dir in exp_dirs:
        for coord_dir in exp_dir.iterdir():
            if not coord_dir.is_dir():
                continue
            raw_path = coord_dir / f"results_{coord_dir.name}.txt"
            if not raw_path.exists():
                continue
            df = parse_raw_results(str(raw_path))
            if df.empty:
                continue
            df["exp_id"] = exp_dir.name
            df["coord"] = coord_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ts_now(fmt="%Y%m%d_%H%M%S") -> str:
    return datetime.now(KST).strftime(fmt)


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="BatchLab (multi-run + analytics)", page_icon="📊", layout="wide")
st.title("BatchLab · multi-coordinate generator & result analyzer")
st.caption("Run multiple coords/parameter sets in batch, then compare results side by side. Existing dashboard stays untouched.")


# -------------------------------------------------
# Session state defaults
# -------------------------------------------------
if "batch_base_path" not in st.session_state:
    st.session_state.batch_base_path = CLOUD_BASE_PATH if IS_CLOUD else DEFAULT_LOCAL_BASE_PATH
if "batch_env_txt" not in st.session_state:
    st.session_state.batch_env_txt = ""
if "batch_coords" not in st.session_state:
    st.session_state.batch_coords = []
if "batch_params" not in st.session_state:
    st.session_state.batch_params = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "batch_api_key" not in st.session_state:
    st.session_state.batch_api_key = get_kakao_key_from_secrets()
if "batch_departure_date" not in st.session_state:
    st.session_state.batch_departure_date = datetime.now(KST).date()
if "batch_departure_time" not in st.session_state:
    st.session_state.batch_departure_time = datetime.now(KST).time()
if "batch_search_results" not in st.session_state:
    st.session_state.batch_search_results = []


# -------------------------------------------------
# Section: base path + env
# -------------------------------------------------
st.markdown("### 1) Base path & runtime env")
col_bp, col_env = st.columns([2, 1])
with col_bp:
    bp_input = st.text_input(
        "base_path (MCI_ADV root)",
        value=st.session_state.batch_base_path,
        placeholder="e.g., C:\\Users\\USER\\MCI_ADV",
        disabled=IS_CLOUD,
    )
    if st.button("Use this base_path", key="set_base_path"):
        st.session_state.batch_base_path = bp_input.strip()
    if IS_CLOUD:
        st.info(f"Cloud mode detected; base_path fixed to `{CLOUD_BASE_PATH}`.")
    if not base_ok(st.session_state.batch_base_path):
        st.warning("base_path must contain `scenarios/`.")
        st.stop()
with col_env:
    st.text_area(
        "Extra env (KEY=VALUE per line, passed to make/run)",
        value=st.session_state.batch_env_txt,
        key="batch_env_txt",
        height=120,
    )


# -------------------------------------------------
# Section: API key + departure time
# -------------------------------------------------
st.markdown("### 2) Kakao API + departure time")
col_key, col_time = st.columns([1, 1])
with col_key:
    cloud_key = get_kakao_key_from_secrets() if IS_CLOUD else ""
    st.session_state.batch_api_key = st.text_input(
        "Kakao REST API key",
        value=cloud_key or st.session_state.batch_api_key,
        type="password",
        help="Used for address lookup and reverse geocoding in generator.",
    )
    if cloud_key:
        st.caption("Loaded from Streamlit secrets.")

    # ── Road data provider (Kakao API ↔ OSRM) ─────────────
    is_use_time = st.checkbox(
        "Use Kakao Mobility API duration (real-time traffic)",
        value=True,
        key="batch_is_use_time",
        help=(
            "✅ Checked → Calls the Kakao Mobility API. Duration (minutes) for the given "
            "departure time is saved to CSV and used by the simulator as "
            "'duration × duration_coeff'. **Requires a Kakao REST API key.**\n\n"
            "⬜ Unchecked → Calls the open-source OSRM service. Both distance and duration "
            "are saved with the same schema, but the **first simulation runs in "
            "distance/velocity mode**. No API key needed — recommended for external "
            "reviewers and public code environments. If you later re-run the same "
            "scenario folder with is_use_time=True, the stored OSRM duration is used."
        ),
    )
    if not is_use_time:
        st.caption("ℹ️ OSRM mode: no Kakao key required. The OSRM URL comes from the "
                   "`MCI_OSRM_URL` environment variable, or the default demo server.")

with col_time:
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.batch_departure_date = st.date_input("Departure date", value=st.session_state.batch_departure_date)
    with c2:
        st.session_state.batch_departure_time = st.time_input("Departure time", value=st.session_state.batch_departure_time)
    departure_time_str = f"{st.session_state.batch_departure_date.strftime('%Y%m%d')}{st.session_state.batch_departure_time.strftime('%H%M')}"
    st.caption(f"API param: departure_time = {departure_time_str}")

duration_coeff = st.number_input(
    "duration_coeff (API travel time scaler)",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
    help="Coefficient multiplied with the API duration when is_use_time=True. Ignored in OSRM mode (is_use_time=False)."
)


# -------------------------------------------------
# Section: search & coordinate list
# -------------------------------------------------
st.markdown("### 3) Search coordinates and build a list")
search_type = st.radio("Search type", ["Keyword", "Address"], horizontal=True)
col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    placeholder = "e.g., Gimpo Airport" if search_type == "Keyword" else "e.g., 110 Sejong-daero, Jung-gu, Seoul"
    search_kw = st.text_input("Search term", placeholder=placeholder)
with col_s2:
    st.write("")
    st.write("")
    do_search = st.button("Search", key="btn_search")

if do_search and search_kw:
    if not st.session_state.batch_api_key:
        st.error("Provide Kakao REST API key first.")
    else:
        try:
            if search_type == "Keyword":
                url = "https://dapi.kakao.com/v2/local/search/keyword.json"
                headers = {"Authorization": f"KakaoAK {st.session_state.batch_api_key.strip()}"}
                params = {"query": search_kw, "size": 10}
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code != 200:
                    st.error(f"HTTP {resp.status_code}: {resp.text}")
                    st.session_state.batch_search_results = []
                else:
                    docs = resp.json().get("documents", [])
                    st.session_state.batch_search_results = [normalize_search_result(doc, "keyword") for doc in docs]
            else:
                ok, docs, msg = perform_address_search(search_kw, st.session_state.batch_api_key)
                if ok:
                    st.session_state.batch_search_results = docs
                else:
                    st.error(msg)
                    st.session_state.batch_search_results = []
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state.batch_search_results = []

if st.session_state.batch_search_results:
    first = st.session_state.batch_search_results[0]
    fmap = folium.Map(location=[first["y"], first["x"]], zoom_start=13, tiles="OpenStreetMap")
    for idx, place in enumerate(st.session_state.batch_search_results):
        popup_html = f"<b>{place.get('place_name','')}</b><br>{place.get('address_name','')}<br>lat: {place['y']:.6f}<br>lon: {place['x']:.6f}"
        folium.Marker(
            location=[place["y"], place["x"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=place.get("place_name", ""),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(fmap)
    st_folium(fmap, width=720, height=380, returned_objects=[])

    st.markdown("**Search results (click to add to list)**")
    for idx, place in enumerate(st.session_state.batch_search_results):
        cols = st.columns([4, 1])
        with cols[0]:
            st.write(f"{idx+1}. {place.get('place_name','')}")
            st.caption(place.get("address_name", ""))
            st.caption(f"({place['y']:.6f}, {place['x']:.6f}) · {place.get('search_type')}")
        with cols[1]:
            if st.button("Add", key=f"add_place_{idx}"):
                st.session_state.batch_coords.append(
                    {
                        "label": place.get("place_name", f"place_{idx+1}"),
                        "lat": float(place["y"]),
                        "lon": float(place["x"]),
                        "address": place.get("address_name", ""),
                    }
                )
                st.success("Added to coordinate list.")

st.markdown("**Manual add**")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    manual_label = st.text_input("Label", value="coord")
with mc2:
    manual_lat = st.number_input("Latitude", value=37.5665, format="%.6f")
with mc3:
    manual_lon = st.number_input("Longitude", value=126.9780, format="%.6f")
if st.button("Add manual coord"):
    st.session_state.batch_coords.append({"label": manual_label or "coord", "lat": manual_lat, "lon": manual_lon, "address": ""})

if st.session_state.batch_coords:
    st.markdown("**Coordinate list**")
    for idx, c in enumerate(st.session_state.batch_coords):
        cols = st.columns([5, 1])
        with cols[0]:
            st.write(f"{idx+1}. {c['label']}  ({c['lat']:.6f}, {c['lon']:.6f})")
            if c.get("address"):
                st.caption(c["address"])
        with cols[1]:
            if st.button("Remove", key=f"rm_coord_{idx}"):
                st.session_state.batch_coords.pop(idx)
                st.rerun()
else:
    st.info("No coordinates in list yet.")


# -------------------------------------------------
# Section: parameter presets
# -------------------------------------------------
st.markdown("### 4) Parameter presets")
p1, p2, p3 = st.columns(3)
with p1:
    incident_size = st.number_input("incident_size", min_value=1, value=30, step=1)
    amb_count = st.number_input("amb_count", min_value=1, value=30, step=1)
    uav_count = st.number_input("uav_count", min_value=0, value=3, step=1)
    buffer_ratio = st.number_input("buffer_ratio", min_value=1.0, value=1.5, step=0.1, format="%.2f")
with p2:
    amb_velocity = st.number_input("ambulance velocity (km/h)", min_value=1, value=40, step=1)
    amb_handover_time = st.number_input("ambulance handover_time (min)", min_value=0.0, value=10.0, step=0.1)
    hospital_max_send_coeff = st.text_input("hospital max_send_coeff", value="1,1", help="Comma separated")
    total_samples = st.number_input("totalSamples", min_value=1, value=30, step=1)
with p3:
    uav_velocity = st.number_input("UAV velocity (km/h)", min_value=1, value=80, step=1)
    uav_handover_time = st.number_input("UAV handover_time (min)", min_value=0.0, value=15.0, step=0.1)
    random_seed = st.number_input("random_seed", min_value=0, value=0, step=1)
    preset_name = st.text_input("Preset name", value="preset-1")

if st.button("Add preset"):
    st.session_state.batch_params.append(
        {
            "name": preset_name or f"preset-{len(st.session_state.batch_params)+1}",
            "incident_size": int(incident_size),
            "amb_count": int(amb_count),
            "uav_count": int(uav_count),
            "amb_velocity": int(amb_velocity),
            "uav_velocity": int(uav_velocity),
            "amb_handover_time": float(amb_handover_time),
            "uav_handover_time": float(uav_handover_time),
            "total_samples": int(total_samples),
            "random_seed": int(random_seed),
            "buffer_ratio": float(buffer_ratio),
            "hospital_max_send_coeff": hospital_max_send_coeff.strip(),
            "is_use_time": is_use_time,
            "duration_coeff": float(duration_coeff),
        }
    )
    st.success("Preset added.")

if st.session_state.batch_params:
    st.dataframe(pd.DataFrame(st.session_state.batch_params), width='stretch', hide_index=True)
    for idx, _ in enumerate(st.session_state.batch_params):
        if st.button(f"Remove preset {idx+1}", key=f"rm_preset_{idx}"):
            st.session_state.batch_params.pop(idx)
            st.rerun()
else:
    st.info("No presets. Add at least one to enable batch run.")


# -------------------------------------------------
# Section: run batch
# -------------------------------------------------
st.markdown("### 5) Run batch")
col_run1, col_run2, col_run3 = st.columns([2, 1, 1])
with col_run1:
    exp_prefix = st.text_input("Experiment id prefix", value="batch")
    append_ts = st.checkbox("Append timestamp to prefix", value=True)
with col_run2:
    action = st.radio("Action", ["Generate only", "Generate + run simulation"], horizontal=False)
with col_run3:
    dryrun_show = st.checkbox("Show commands only (no execution)", value=False)

if st.button("Run batch now", type="primary"):
    if not st.session_state.batch_coords:
        st.error("Add at least one coordinate.")
        st.stop()
    if not st.session_state.batch_params:
        st.error("Add at least one preset.")
        st.stop()
    prefix = exp_prefix.strip() or "batch"
    if append_ts:
        prefix = f"{prefix}_{ts_now()}"
    env = parse_env_kv(st.session_state.batch_env_txt)
    run_log = []
    orc = Orchestrator(base_path=st.session_state.batch_base_path)
    run_idx = 0
    for c_idx, coord in enumerate(st.session_state.batch_coords, start=1):
        for p_idx, preset in enumerate(st.session_state.batch_params, start=1):
            run_idx += 1
            exp_id = f"{prefix}_c{c_idx}p{p_idx}"
            extra_args = {
                "buffer_ratio": preset["buffer_ratio"],
                "hospital_max_send_coeff": preset["hospital_max_send_coeff"],
                "kakao_api_key": st.session_state.batch_api_key.strip(),
                "departure_time": departure_time_str,
                "is_use_time": str(preset["is_use_time"]).lower(),
                "amb_handover_time": preset["amb_handover_time"],
                "uav_handover_time": preset["uav_handover_time"],
                "duration_coeff": preset["duration_coeff"],
            }
            record = {
                "exp_id": exp_id,
                "coord_label": coord["label"],
                "coord": (coord["lat"], coord["lon"]),
                "preset": preset["name"],
                "config_path": None,
                "log_file": None,
                "status": "pending",
                "error": "",
            }
            if dryrun_show:
                record["status"] = "dryrun"
                run_log.append(record)
                continue
            try:
                gen = orc.generate_scenario(
                    latitude=coord["lat"],
                    longitude=coord["lon"],
                    incident_size=preset["incident_size"],
                    amb_count=preset["amb_count"],
                    uav_count=preset["uav_count"],
                    amb_velocity=preset["amb_velocity"],
                    uav_velocity=preset["uav_velocity"],
                    total_samples=preset["total_samples"],
                    random_seed=preset["random_seed"],
                    exp_id=exp_id,
                    extra_env=env,
                    extra_args=extra_args,
                )
                record["config_path"] = gen.get("config_path")
                record["log_file"] = gen.get("log_file")
                record["status"] = "generated"
                if action == "Generate + run simulation" and gen.get("config_path"):
                    sim = orc.run_simulation(config_path=gen["config_path"], extra_env=env)
                    record["status"] = "simulated" if sim.get("ok") else f"sim failed ({sim.get('returncode')})"
                    record["log_file"] = sim.get("log_file") or record["log_file"]
                run_log.append(record)
            except Exception as e:
                record["status"] = "error"
                record["error"] = str(e)
                run_log.append(record)
    st.session_state.batch_results = run_log
    st.success(f"Batch finished: {len(run_log)} runs queued.")

if st.session_state.batch_results:
    st.markdown("**Run log**")
    st.dataframe(pd.DataFrame(st.session_state.batch_results), width='stretch', hide_index=True)


# -------------------------------------------------
# Section: results analytics
# -------------------------------------------------
st.markdown("### 6) Results analytics")
results_root = os.path.join(st.session_state.batch_base_path, "results")
available_exps = []
if os.path.isdir(results_root):
    available_exps = sorted([d.name for d in Path(results_root).iterdir() if d.is_dir()])
sel_exps = st.multiselect("Experiments to load", options=available_exps, default=available_exps[-5:] if available_exps else [])
load_btn = st.button("Load results")

if load_btn and sel_exps:
    df_raw = collect_results(results_root, sel_exps)
    if df_raw.empty:
        st.warning("No raw results found in selected experiments.")
    else:
        st.success(f"Loaded {len(df_raw)} rows.")
        metric_choices = ["Reward", "PDR", "Time", "Reward_woG", "PDR_woG"]
        metric_sel = st.selectbox("Metric", metric_choices, index=0)
        df_m = df_raw[df_raw["metric"] == metric_sel]
        if df_m.empty:
            st.warning("No rows for that metric.")
        else:
            agg = (
                df_m.groupby(["exp_id", "coord", "rule"])
                .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
                .reset_index()
            )
            st.markdown("**Summary (mean/std per rule)**")
            st.dataframe(agg, width='stretch', hide_index=True)

            topN = st.slider("Top N by mean", min_value=5, max_value=50, value=15, step=1)
            top_rules = agg.sort_values("mean", ascending=False).head(topN)
            chart = (
                alt.Chart(top_rules)
                .mark_bar()
                .encode(
                    x=alt.X("mean:Q", title=f"{metric_sel} (mean)"),
                    y=alt.Y("rule:N", sort="-x"),
                    color="coord:N",
                    tooltip=["exp_id", "coord", "rule", "mean", "std", "n"],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, width='stretch')

            # Reward vs Time trade-off scatter (mean per rule)
            pivot = (
                agg.pivot_table(index=["exp_id", "coord", "rule"], columns="metric", values="mean")
                if "metric" in agg.columns
                else agg
            )
            if {"Reward", "Time"}.issubset(set(df_raw["metric"].unique())):
                pivot_rt = (
                    df_raw[df_raw["metric"].isin(["Reward", "Time"])]
                    .groupby(["exp_id", "coord", "rule", "metric"])
                    .agg(mean=("value", "mean"))
                    .reset_index()
                    .pivot_table(index=["exp_id", "coord", "rule"], columns="metric", values="mean")
                    .reset_index()
                )
                st.markdown("**Reward vs Time (mean per rule)**")
                scat = (
                    alt.Chart(pivot_rt)
                    .mark_circle(size=80, opacity=0.7)
                    .encode(
                        x=alt.X("Time:Q", title="Time (mean)"),
                        y=alt.Y("Reward:Q", title="Reward (mean)"),
                        color="coord:N",
                        shape="exp_id:N",
                        tooltip=["exp_id", "coord", "rule", "Reward", "Time"],
                    )
                    .properties(height=400)
                )
                st.altair_chart(scat, width='stretch')

            st.download_button(
                "Download summary CSV",
                agg.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"batchlab_summary_{ts_now()}.csv",
                mime="text/csv",
            )
else:
    st.caption("Select experiments and click 'Load results' to visualize.")
