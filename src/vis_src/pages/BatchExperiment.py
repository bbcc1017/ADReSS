# BatchExperiment.py — Batch Experiment Pipeline Dashboard
# Step-by-step workflow: Coordinate Generation → Scenario → Simulation → Visualization
# Reuses existing functions from experiment_1/ scripts and orchestrator.
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _detect_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "src" / "sce_src" / "orchestrator.py").is_file():
            return parent
    return here


REPO_ROOT = _detect_repo_root()
EXP1_DIR = REPO_ROOT / "experiment_1"
KST = timezone(timedelta(hours=9))

# Add experiment_1 and sce_src to sys.path for imports
for _dir in [str(EXP1_DIR), str(REPO_ROOT / "src" / "sce_src")]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# ---------------------------------------------------------------------------
# Lazy imports from existing modules
# ---------------------------------------------------------------------------

_IMPORTS_OK = True
_IMPORT_ERR = ""

try:
    from generate_coords import (
        load_korea_boundary, generate_points, save_csv, save_map,
    )
    from batch_runner import (
        load_coords as br_load_coords,
        load_progress, save_progress,
        calc_stats, select_pending, process_coord,
        now_kst, today_kst,
    )
    from visualize_coords import (
        load_coords as viz_load_coords,
        collect_data, compute_ranges, build_map, build_histograms,
        find_results_dir, build_rule_analysis,
    )
    from orchestrator import Orchestrator
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERR = str(e)


# ---------------------------------------------------------------------------
# Boundary cache
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Korea boundary shapefile...")
def _cached_load_boundary(shp_path: str):
    return load_korea_boundary(shp_path)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "batch_stop_requested": False,
    "batch_running": False,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_shp(shp_name: str) -> Path:
    p = Path(shp_name)
    if p.exists():
        return p
    for base in (EXP1_DIR, REPO_ROOT):
        candidate = base / shp_name
        if candidate.exists():
            return candidate
    return EXP1_DIR / shp_name


def _embed_html(html_path: Path, height: int = 600):
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        components.html(html_content, height=height, scrolling=True)
    else:
        st.warning(f"HTML file not found: {html_path}")


def _now_kst_departure() -> str:
    """Current KST time as YYYYMMDDHHmm for departure_time default."""
    return datetime.now(KST).strftime("%Y%m%d%H%M")


def _build_full_exp_id(base_name: str, departure_time: str, is_use_time_kakao: bool = True) -> str:
    """Compute the full experiment folder name with the appropriate suffix.
    Mirrors make_csv_yaml_dynamic.py logic so the folder name matches exactly.
    Kakao mode (is_use_time_kakao=True) → exp_<base>_dep_<YYYYMMDDHHMM>
    OSRM  mode (is_use_time_kakao=False) → exp_<base>_osrm
    """
    import re as _re
    base = base_name.strip()
    if base.startswith("exp_"):
        base = base[4:]
    base = _re.sub(r"\s+", "_", base).strip("_")
    if not base:
        base = datetime.now(KST).strftime("%Y%m%d%H%M")
    # Strip any pre-existing _dep_ / _osrm suffix so we never double-append.
    base = _re.sub(r"_dep_\d{12}$", "", base)
    base = _re.sub(r"_osrm$", "", base)
    if not is_use_time_kakao:
        return f"exp_{base}_osrm"
    if departure_time and departure_time.strip():
        return f"exp_{base}_dep_{departure_time.strip()}"
    return f"exp_{base}"


def _make_args(**kwargs) -> SimpleNamespace:
    """Create an args namespace compatible with process_coord()."""
    defaults = dict(
        incident_size=30, amb_count=30, uav_count=3,
        amb_velocity=40, uav_velocity=80,
        total_samples=30, random_seed=0,
        experiment_id="exp_batch_research",
        kakao_api_key="", departure_time="",
        daily_limit=5000, calls_per_coord=40, max_retries=2,
        amb_handover_time=10.0, uav_handover_time=15.0,
        is_use_time="true", duration_coeff=1.0,
        hospital_max_send_coeff=None, buffer_ratio=None, util_by_tier=None,
        osrm_url=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _scenario_dir(experiment_id: str) -> Path:
    """Return scenarios/<experiment_id>/ path."""
    return REPO_ROOT / "scenarios" / experiment_id


def _list_scenario_folders() -> list[str]:
    """List existing experiment folders under scenarios/, sorted newest first."""
    sce_root = REPO_ROOT / "scenarios"
    if not sce_root.exists():
        return []
    folders = [
        d.name for d in sorted(sce_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if d.is_dir()
    ]
    return folders


def _experiment_selectbox(label: str, key: str) -> str | None:
    """Render a selectbox of existing scenario folders. Returns selected name or None."""
    folders = _list_scenario_folders()
    if not folders:
        st.info("No experiment folders found in `scenarios/`.")
        return None
    return st.selectbox(label, folders, key=key)


# Sentinel used to skip the rest of an expander when no folder is selected.
_SKIP = "__SKIP__"


# ===========================================================================
# PAGE START
# ===========================================================================

st.set_page_config(page_title="Batch Experiment", page_icon="🔬", layout="wide")
st.title("Batch Experiment Pipeline")
st.caption("End-to-end workflow: coordinate generation → scenario → simulation → visualization")

if not _IMPORTS_OK:
    st.error(
        f"Failed to import required modules: `{_IMPORT_ERR}`\n\n"
        "Make sure `geopandas`, `shapely`, `folium`, `matplotlib`, `numpy` are installed."
    )
    st.stop()

# ===========================================================================
# STEP 1 — Generate Random Coordinates
# ===========================================================================

with st.expander("Step 1: Generate Random Coordinates", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        n_coords = st.number_input("Number of coordinates", min_value=1, max_value=50000,
                                   value=1000, step=100, key="s1_n")
        seed_val = st.number_input("Random seed", min_value=0, value=0, key="s1_seed")
    with col2:
        shp_input = st.text_input("SHP file", value="ctprvn.shp", key="s1_shp")
        s1_base_name = st.text_input("Experiment name", value="batch_research",
                                     key="s1_expid")
    with col3:
        s1_use_kakao = st.checkbox(
            "Use Kakao Mobility API (is_use_time)",
            value=True,
            key="s1_use_kakao",
            help=(
                "✅ Checked → Kakao API mode. Folder name gets a `_dep_<YYYYMMDDHHMM>` suffix "
                "from the departure time below.\n\n"
                "⬜ Unchecked → OSRM (open-source) mode. Departure time is meaningless for OSRM, "
                "so the folder gets an `_osrm` suffix instead."
            ),
        )
        if s1_use_kakao:
            s1_dep_time = st.text_input("Departure time (YYYYMMDDHHmm)",
                                        value=_now_kst_departure(), key="s1_dep",
                                        help="Kakao API routing time. Included in folder name.")
        else:
            s1_dep_time = ""
            st.caption("Departure time disabled (OSRM mode).")

    s1_exp_id = _build_full_exp_id(s1_base_name, s1_dep_time, is_use_time_kakao=s1_use_kakao)

    # CSV path is derived from full experiment_id: scenarios/<exp_id>/coords.csv
    csv_path = _scenario_dir(s1_exp_id) / "coords.csv"
    st.caption(f"Folder: `scenarios/{s1_exp_id}/`")

    if csv_path.exists():
        st.info(f"`coords.csv` already exists in this experiment folder. "
                f"Generation will overwrite it.")

    if st.button("Generate Coordinates", key="s1_generate"):
        shp_resolved = _resolve_shp(shp_input)
        if not shp_resolved.exists():
            st.error(f"SHP file not found: {shp_resolved}")
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            boundary = _cached_load_boundary(str(shp_resolved))
            with st.spinner(f"Generating {n_coords} random points (seed={seed_val})..."):
                pts = generate_points(boundary, n_coords, seed_val)
            save_csv(pts, str(csv_path))
            map_out = csv_path.parent / "coords_preview.html"
            save_map(pts, str(map_out))
            st.success(f"Generated **{len(pts)}** coordinates → `scenarios/{s1_exp_id}/`")

# ===========================================================================
# STEP 2 — View Coordinates
# ===========================================================================

with st.expander("Step 2: View Coordinates", expanded=False):
    s2_exp_id = _experiment_selectbox("Experiment Folder", key="s2_expid")
    if s2_exp_id is not None:
        coords_csv_path = _scenario_dir(s2_exp_id) / "coords.csv"

        if not coords_csv_path.exists():
            st.info(f"No `coords.csv` found in `scenarios/{s2_exp_id}/`. Complete **Step 1** first.")
        else:
            tab_table, tab_map = st.tabs(["Table", "Map"])
            with tab_table:
                df = pd.read_csv(coords_csv_path)
                st.metric("Total coordinates", len(df))
                st.dataframe(df, use_container_width=True, height=400)
            with tab_map:
                preview_html = coords_csv_path.parent / "coords_preview.html"
                if not preview_html.exists():
                    pts_for_map = list(zip(df["latitude"].tolist(), df["longitude"].tolist()))
                    save_map(pts_for_map, str(preview_html))
                _embed_html(preview_html, height=550)


# ===========================================================================
# STEP 3 — Generate Scenarios & Run Simulations
# ===========================================================================

with st.expander("Step 3: Generate Scenarios & Run Simulations", expanded=False):
    st.markdown("Runs scenario generation + simulation per coordinate.")

    # --- Experiment folder selection ---
    st.subheader("Experiment & API Settings")
    experiment_id = _experiment_selectbox("Experiment Folder", key="s3_expid")

    # ── Road data provider auto-detected from folder suffix ────────
    # `*_dep_<12digits>` → kakao mode, `*_osrm` → osrm mode.
    import re as _re
    _dep_match = _re.search(r"_dep_(\d{12})$", experiment_id or "")
    _is_osrm_folder = bool(_re.search(r"_osrm$", experiment_id or ""))

    if _is_osrm_folder:
        is_use_time_bool = False
        departure_time = ""
        st.info("🛣️ OSRM mode (folder ends with `_osrm`). Kakao API key not required.")
    elif _dep_match:
        is_use_time_bool = True
        departure_time = _dep_match.group(1)
        st.info(f"🗾 Kakao mode (folder has `_dep_` suffix). Departure time: `{departure_time}`.")
    else:
        # Legacy folder with no recognizable suffix — let the user pick.
        is_use_time_bool = st.checkbox(
            "Use Kakao Mobility API duration (real-time traffic)",
            value=True,
            key="s3_usetime_chk_legacy",
            help="Folder has no `_dep_` or `_osrm` suffix; choose the mode manually.",
        )
        departure_time = st.text_input("Departure time (YYYYMMDDHHmm)",
                                       value=_now_kst_departure(), key="s3_dep",
                                       help="Folder has no dep suffix. Enter manually.")
    is_use_time = "true" if is_use_time_bool else "false"

    # --- API settings (only shown when Kakao mode is active) ---
    _requires_kakao = is_use_time_bool
    if _requires_kakao:
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            kakao_key = st.text_input("Kakao API Key", type="password", key="s3_kakao")
        with col_e2:
            daily_limit = st.number_input("Daily API limit", value=5000, key="s3_dlimit")
            calls_per_coord = st.number_input("Est. API calls per coord", value=40, key="s3_cpc")
        with col_e3:
            max_retries = st.number_input("Max retries per coord", value=2, key="s3_retry")
    else:
        kakao_key = ""
        daily_limit = 999999
        calls_per_coord = 0
        max_retries = st.number_input("Max retries per coord", value=2, key="s3_retry")

    st.divider()

    # --- Simulation parameters ---
    st.subheader("Simulation Parameters")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        incident_size = st.number_input("Incident size (patients)", value=30, key="s3_inc")
        amb_count = st.number_input("Ambulance count", value=30, key="s3_amb")
        uav_count = st.number_input("UAV count", value=3, key="s3_uav")
    with col_s2:
        amb_velocity = st.number_input("AMB velocity (km/h)", value=40, key="s3_ambv")
        uav_velocity = st.number_input("UAV velocity (km/h)", value=80, key="s3_uavv")
        total_samples = st.number_input("Simulation samples", value=30, key="s3_samples")
    with col_s3:
        random_seed = st.number_input("Random seed", value=0, key="s3_seed")
        amb_handover = st.number_input("AMB handover time (min)", value=10.0, key="s3_ambh")
        uav_handover = st.number_input("UAV handover time (min)", value=15.0, key="s3_uavh")

    with st.expander("Advanced Transport Parameters", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            duration_coeff = st.number_input(
                "Duration coefficient", value=1.0, key="s3_dcoeff",
                help="Coefficient multiplied with the API duration when is_use_time=True. Ignored in OSRM mode (is_use_time=False)."
            )
        with col_t2:
            st.caption("Hospital allocation params (leave empty for defaults)")
            hospital_max_send = st.text_input("Hospital max send coeff (default: 1,1)",
                                              value="", key="s3_hcoeff")
            buffer_ratio_str = st.text_input("Buffer ratio", value="", key="s3_buffer")
            util_by_tier_str = st.text_input("Util by tier (e.g. 1:0.90,11:0.75)",
                                             value="", key="s3_util")

    # --- Progress file & coords file paths (derived from experiment_id) ---
    if experiment_id is None:
        st.info("Select an experiment folder above.")
        experiment_id = "__none__"  # prevent NameError below; won't match any path
    sce_dir = _scenario_dir(experiment_id)
    progress_path = str(sce_dir / "progress.json")
    coords_csv_path = sce_dir / "coords.csv"
    progress_file = Path(progress_path)

    st.divider()
    st.caption(f"Coords: `{coords_csv_path}` — Progress: `{progress_file}`")

    if not coords_csv_path.exists():
        st.info(f"No `coords.csv` in `scenarios/{experiment_id}/`. Complete **Step 1** first "
                f"with the same Experiment ID.")
    else:
        coords_dict = br_load_coords(str(coords_csv_path))
        n_total = len(coords_dict)

        if progress_file.exists():
            with open(progress_file, encoding="utf-8") as f:
                progress_data = json.load(f)
            stats = calc_stats(progress_data)
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Done", f"{stats['done']}/{stats['total']}")
            col_m2.metric("Sim OK", stats["sim_ok"])
            col_m3.metric("Sim Fail", stats["sim_fail"])
            col_m4.metric("Failed", stats["failed"])
            col_m5.metric("Pending", stats["pending"])
            st.progress(stats["done"] / max(stats["total"], 1))
            st.caption(f"Today API usage: {stats['today_calls']} / {daily_limit}")
        else:
            st.caption(f"No progress file yet. Will create on first run. ({n_total} coords loaded)")

        # --- Run / Stop buttons ---
        col_run, col_stop = st.columns([3, 1])
        with col_run:
            start_btn = st.button("Start Batch Processing", key="s3_start",
                                  disabled=st.session_state.batch_running)
        with col_stop:
            if st.button("Stop", key="s3_stop"):
                st.session_state.batch_stop_requested = True

        if start_btn:
            # OSRM mode (is_use_time=false) does not require a Kakao key
            requires_kakao = (str(is_use_time).lower() == "true")
            if requires_kakao and not kakao_key:
                st.error("Kakao API Key is required when is_use_time=true. "
                         "If you don't have a key, uncheck the box above to use the OSRM backend instead.")
            else:
                st.session_state.batch_running = True
                st.session_state.batch_stop_requested = False

                # Parse optional advanced params
                hcoeff = hospital_max_send if hospital_max_send.strip() else None
                buf = float(buffer_ratio_str) if buffer_ratio_str.strip() else None
                util = util_by_tier_str if util_by_tier_str.strip() else None

                args_ns = _make_args(
                    incident_size=incident_size, amb_count=amb_count, uav_count=uav_count,
                    amb_velocity=amb_velocity, uav_velocity=uav_velocity,
                    total_samples=total_samples, random_seed=random_seed,
                    experiment_id=experiment_id, kakao_api_key=kakao_key,
                    departure_time=departure_time, daily_limit=daily_limit,
                    calls_per_coord=calls_per_coord, max_retries=max_retries,
                    amb_handover_time=amb_handover, uav_handover_time=uav_handover,
                    is_use_time=is_use_time, duration_coeff=duration_coeff,
                    hospital_max_send_coeff=hcoeff, buffer_ratio=buf, util_by_tier=util,
                )

                orch = Orchestrator(base_path=str(REPO_ROOT))
                progress_data = load_progress(str(progress_file), experiment_id, n_total)
                pending = select_pending(progress_data, max_retries)

                stats = calc_stats(progress_data)
                if calls_per_coord > 0:
                    remaining_budget = daily_limit - stats["today_calls"]
                    can_process = max(0, remaining_budget // calls_per_coord)
                    targets = pending[:can_process] if can_process > 0 else []
                else:
                    # OSRM mode: no API calls, process all pending
                    remaining_budget = 0
                    can_process = len(pending)
                    targets = pending

                if not targets:
                    st.warning("No coordinates to process (all done, or daily API limit reached).")
                    st.session_state.batch_running = False
                else:
                    st.info(f"Processing **{len(targets)}** coordinates "
                            f"({len(pending)} pending)")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    session_calls = 0
                    results_summary = []

                    for i, cid in enumerate(targets):
                        if st.session_state.batch_stop_requested:
                            status_text.warning(f"Stopped by user after {i} coordinates.")
                            break

                        cid_str = str(cid)
                        c = coords_dict[cid_str]
                        status_text.text(
                            f"Processing coord {cid_str} ({i+1}/{len(targets)}) — "
                            f"({c['latitude']:.4f}, {c['longitude']:.4f})"
                        )
                        progress_bar.progress((i + 1) / len(targets))

                        progress_data, api_used, sim_ok = process_coord(
                            orch=orch,
                            coord_id=cid_str,
                            lat=c["latitude"],
                            lon=c["longitude"],
                            args=args_ns,
                            progress=progress_data,
                            session_calls=session_calls,
                            session_idx=i + 1,
                            session_total=len(targets),
                        )
                        session_calls += api_used
                        save_progress(progress_data, str(progress_file))

                        status_label = progress_data["statuses"][cid_str].get("status", "?")
                        sim_label = "sim_ok" if sim_ok else "sim_fail"
                        results_summary.append({
                            "coord_id": cid_str,
                            "status": status_label,
                            "sim": sim_label,
                            "api_calls": api_used,
                        })

                        if calls_per_coord > 0 and session_calls >= remaining_budget:
                            status_text.warning("Daily API budget exhausted.")
                            break

                    # Update api_log
                    today = today_kst()
                    api_log = progress_data.get("api_log", [])
                    today_entry = next((e for e in api_log if e.get("date") == today), None)
                    if today_entry:
                        today_entry["calls_used"] = today_entry.get("calls_used", 0) + session_calls
                        today_entry["coords_processed"] = (
                            today_entry.get("coords_processed", 0) + len(results_summary)
                        )
                    else:
                        api_log.append({
                            "date": today,
                            "calls_used": session_calls,
                            "coords_processed": len(results_summary),
                        })
                    progress_data["api_log"] = api_log
                    save_progress(progress_data, str(progress_file))

                    st.session_state.batch_running = False

                    if results_summary:
                        status_text.success(
                            f"Batch complete: {len(results_summary)} coords processed, "
                            f"{session_calls} API calls used."
                        )
                        st.dataframe(pd.DataFrame(results_summary), use_container_width=True)


# ===========================================================================
# STEP 4 — Progress Dashboard
# ===========================================================================

with st.expander("Step 4: Progress Dashboard", expanded=False):
    s4_exp_id = _experiment_selectbox("Experiment Folder", key="s4_expid")
    if s4_exp_id is None:
        st.info("No experiment folders found.")
    else:
        progress_file = _scenario_dir(s4_exp_id) / "progress.json"

        if not progress_file.exists():
            st.info(f"No progress file found at `scenarios/{s4_exp_id}/progress.json`.")
        else:
            with open(progress_file, encoding="utf-8") as f:
                prog = json.load(f)

            stats = calc_stats(prog)
            total = max(stats["total"], 1)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total", stats["total"])
            col2.metric("Done", stats["done"])
            col3.metric("Sim OK", stats["sim_ok"])
            col4.metric("Sim Fail", stats["sim_fail"])
            col5.metric("Failed", stats["failed"])
            col6.metric("Pending", stats["pending"])

            st.progress(stats["done"] / total)
            st.caption(f"Completion: {stats['done']/total*100:.1f}% — "
                       f"Today API: {stats['today_calls']} calls")

            # Status filter
            filter_status = st.multiselect(
                "Filter by status", ["done", "failed", "pending", "running", "abandoned"],
                default=["done", "failed", "pending"], key="s4_filter"
            )

            rows = []
            for cid, v in sorted(prog.get("statuses", {}).items(), key=lambda x: int(x[0])):
                st_val = v.get("status", "pending")
                if st_val in filter_status:
                    rows.append({
                        "coord_id": int(cid),
                        "status": st_val,
                        "sim_ok": v.get("sim_ok", ""),
                        "attempts": v.get("attempts", 0),
                        "finished_at": v.get("finished_at", ""),
                        "error": v.get("error", v.get("sim_error", ""))[:80],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)
            else:
                st.caption("No entries match the selected filter.")


# ===========================================================================
# STEP 5 — Visualize Results
# ===========================================================================

with st.expander("Step 5: Visualize Results", expanded=False):
    s5_exp_id = _experiment_selectbox("Experiment Folder", key="s5_expid")
    if s5_exp_id is None:
        st.info("No experiment folders found.")
    else:
        s5_sce_dir = _scenario_dir(s5_exp_id)
        progress_file = s5_sce_dir / "progress.json"
        coords_csv_path = s5_sce_dir / "coords.csv"

        if not progress_file.exists() or not coords_csv_path.exists():
            st.info(f"Need both `coords.csv` and `progress.json` in `scenarios/{s5_exp_id}/`. "
                    "Complete **Steps 1-3** first.")
        else:
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                clip_pct = st.slider("Clip percentile", 0.0, 20.0, 5.0, 0.5, key="s5_clip")
            with col_v2:
                outlier_n = st.number_input("Outlier count (each side)", value=3, min_value=0,
                                            key="s5_outlier")
            with col_v3:
                hist_fmt = st.radio("Histogram format", ["pdf", "png"], horizontal=True,
                                    key="s5_fmt")

            viz_dir = s5_sce_dir
            viz_out_path = viz_dir / "coords_map.html"

            if st.button("Generate Visualization", key="s5_gen"):
                with st.spinner("Collecting data and building visualizations..."):
                    coords = viz_load_coords(coords_csv_path)
                    with open(progress_file, encoding="utf-8") as f:
                        prog = json.load(f)
                    results_dir = find_results_dir(prog)
                    if results_dir is None:
                        st.error("Could not auto-detect results directory. "
                                 "Make sure simulations have completed successfully.")
                    else:
                        data = collect_data(coords, prog, results_dir)
                        ranges = compute_ranges(data, clip_pct)
                        build_map(data, viz_out_path, ranges, clip_pct, outlier_n)
                        build_histograms(data, viz_out_path, ranges, clip_pct, outlier_n,
                                         hist_fmt)
                        build_rule_analysis(prog, results_dir, viz_out_path)
                        st.success("Visualization generated!")

            # Display results
            tab_map, tab_hist, tab_heatmap, tab_effects = st.tabs([
                "Results Map", "Histogram", "Rule Heatmap", "Factor Main Effects"
            ])
            with tab_map:
                if viz_out_path.exists():
                    _embed_html(viz_out_path, height=700)
                else:
                    st.caption("No map generated yet. Click **Generate Visualization** above.")
            with tab_hist:
                hist_png = viz_dir / "coords_map_hist.png"
                hist_pdf = viz_dir / "coords_map_hist.pdf"
                if hist_png.exists():
                    st.image(str(hist_png), use_container_width=True)
                if hist_pdf.exists():
                    try:
                        from PIL import Image
                        import fitz  # PyMuPDF

                        doc = fitz.open(str(hist_pdf))
                        page = doc[0]
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        st.image(img, caption="Histogram (PDF preview)",
                                 use_container_width=True)
                        doc.close()
                    except ImportError:
                        st.info("Install `PyMuPDF` (`pip install pymupdf`) for inline PDF "
                                "preview.")
                    st.download_button(
                        "Download Histogram (PDF)",
                        hist_pdf.read_bytes(),
                        file_name="coords_map_hist.pdf",
                        mime="application/pdf",
                    )
                if not hist_png.exists() and not hist_pdf.exists():
                    st.caption("No histogram generated yet. Click **Generate Visualization** "
                               "above.")
            with tab_heatmap:
                heatmap_png = viz_dir / "coords_map_rule_heatmap.png"
                heatmap_pdf = viz_dir / "coords_map_rule_heatmap.pdf"
                if heatmap_png.exists():
                    st.image(str(heatmap_png), use_container_width=True)
                if heatmap_pdf.exists():
                    st.download_button(
                        "Download Rule Heatmap (PDF)",
                        heatmap_pdf.read_bytes(),
                        file_name="rule_heatmap.pdf",
                        mime="application/pdf",
                    )
                if not heatmap_png.exists() and not heatmap_pdf.exists():
                    st.caption("No rule heatmap yet. Click **Generate Visualization** above.")
            with tab_effects:
                effects_png = viz_dir / "coords_map_rule_effects.png"
                effects_pdf = viz_dir / "coords_map_rule_effects.pdf"
                if effects_png.exists():
                    st.image(str(effects_png), use_container_width=True)
                if effects_pdf.exists():
                    st.download_button(
                        "Download Main Effects (PDF)",
                        effects_pdf.read_bytes(),
                        file_name="rule_effects.pdf",
                        mime="application/pdf",
                    )
                if not effects_png.exists() and not effects_pdf.exists():
                    st.caption("No main effects chart yet. Click **Generate Visualization** "
                               "above.")
