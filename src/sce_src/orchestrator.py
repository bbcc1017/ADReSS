# -*- coding: utf-8 -*-
"""
orchestrator.py (robust unpack patch)
- Prevents "too many values to unpack" by normalizing summary path return to exactly two strings.
"""

import os
import csv
import json
import time
import shutil
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta

try:
    import yaml  # pyyaml
except Exception:
    yaml = None
try:
    import pandas as pd
except Exception:
    pd = None

KST = timezone(timedelta(hours=9))

def now_kst_iso() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def ts_short_now() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def exists_file(path: Optional[str]) -> bool:
    try:
        return bool(path) and os.path.isfile(path)
    except Exception:
        return False

def write_text(path: str, text: str, encoding: str = "utf-8"):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding=encoding, errors="ignore") as f:
        f.write(text)

def append_text(path: str, text: str, encoding: str = "utf-8"):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding=encoding, errors="ignore") as f:
        f.write(text)

def to_coord_str(lat: float, lon: float) -> str:
    return f"({lat},{lon})"

def parse_coord_from_config_path(config_path: str) -> Optional[str]:
    try:
        parts = os.path.normpath(config_path).split(os.sep)
        for i in range(len(parts)-1, -1, -1):
            if parts[i].startswith("(") and parts[i].endswith(")"):
                return parts[i]
        return None
    except Exception:
        return None

def parse_exp_from_config_path(config_path: str) -> Optional[str]:
    try:
        parts = os.path.normpath(config_path).split(os.sep)
        if "scenarios" in parts:
            ix = parts.index("scenarios")
            if ix+1 < len(parts):
                return parts[ix+1]
        return None
    except Exception:
        return None

def parse_make_generator_stdout(stdout_text: str):
    coord_info = None
    config_path = None
    api_call_count = None
    for line in stdout_text.splitlines():
        s = line.strip()
        if s.startswith("COORDINATE_INFO:"):
            try:
                js = s.split("COORDINATE_INFO:",1)[1].strip()
                coord_info = json.loads(js)
            except Exception:
                pass
        elif s.startswith("CONFIG_PATH:"):
            config_path = s.split("CONFIG_PATH:",1)[1].strip()
        elif s.startswith("API_CALL_COUNT:"):
            try:
                api_call_count = int(s.split("API_CALL_COUNT:",1)[1].strip())
            except Exception:
                pass
    return coord_info, config_path, api_call_count

# ------------------------------------------------------------------
# Reverse Geocoding (Kakao API)
# ------------------------------------------------------------------

def reverse_geocode_kakao(lat: float, lon: float, api_key: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Kakao Local API coordinate-to-address conversion (reverse geocoding)

    Args:
        lat: latitude
        lon: longitude
        api_key: Kakao REST API key
        max_retries: max retry count

    Returns:
        {
            "full_address": "lot-based address",
            "road_address": "road name address",
            "area1": "province/city",
            "area2": "district",
            "area3": "sub-district",
            "area4": "village",
            "is_valid": True/False,
            "latitude": lat,
            "longitude": lon,
            "api_response_code": 0 (success) or -999 (failure)
        }
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    params = {
        "x": str(lon),  # longitude
        "y": str(lat),  # latitude
        "input_coord": "WGS84"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])

                if documents and len(documents) > 0:
                    doc = documents[0]

                    # Lot-based address
                    address = doc.get("address", {})
                    area1 = address.get("region_1depth_name", "")  # Province/city
                    area2 = address.get("region_2depth_name", "")  # District
                    area3 = address.get("region_3depth_name", "")  # Sub-district
                    area4 = address.get("region_3depth_h_name", "")  # Village (legal district basis)

                    # Lot-based address composition
                    full_address = address.get("address_name", "")

                    # Road name address
                    road_address_obj = doc.get("road_address")
                    road_address = ""
                    if road_address_obj:
                        road_address = road_address_obj.get("address_name", "")
                    else:
                        print(f"  ℹ️ No road address (only lot-based address exists): {full_address}")

                    return {
                        "full_address": full_address,
                        "road_address": road_address,
                        "area1": area1,
                        "area2": area2,
                        "area3": area3,
                        "area4": area4,
                        "is_valid": True,
                        "latitude": lat,
                        "longitude": lon,
                        "api_response_code": 0
                    }

                # No address info (sea, mountain, etc.)
                return {
                    "full_address": "No address info",
                    "road_address": "",
                    "area1": "sea/unknown",
                    "area2": "",
                    "area3": "",
                    "area4": "",
                    "is_valid": False,
                    "latitude": lat,
                    "longitude": lon,
                    "api_response_code": -1
                }

            elif response.status_code == 401:
                print(f"❌ Kakao API auth failed (401): check API key")
                break

            elif response.status_code == 429:
                print(f"⚠️ API rate limit exceeded (429). {attempt + 1}/{max_retries} retrying...")
                time.sleep(2)

            else:
                print(f"❌ API error: {response.status_code}")
                break

        except Exception as e:
            print(f"❌ Reverse geocoding error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # On failure
    return {
        "full_address": "API error",
        "road_address": "",
        "area1": "error",
        "area2": "",
        "area3": "",
        "area4": "",
        "is_valid": False,
        "latitude": lat,
        "longitude": lon,
        "api_response_code": -999
    }

# ============================================================
# [DEPRECATED] Naver API reverse geocoding (for reference)
# ============================================================
# def reverse_geocode_naver(lat: float, lon: float, client_id: str, client_secret: str, max_retries: int = 3) -> Dict[str, Any]:
#     """
#     Naver Reverse Geocoding API (legacy code - for reference)
#
#     This function is no longer used. Migrated to Kakao API.
#     Reference: https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc
#     """
#     url = "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc"
#     headers = {
#         "X-NCP-APIGW-API-KEY-ID": client_id,
#         "X-NCP-APIGW-API-KEY": client_secret
#     }
#     params = {
#         "coords": f"{lon},{lat}",
#         "orders": "legalcode,admcode,addr,roadaddr",
#         "output": "json"
#     }
#     # ... (legacy logic omitted)

# ------------------------------------------------------------------
# Summary CSV helpers
# ------------------------------------------------------------------

SUMMARY_COLS = [
    "exp_id","coord","trial_no","note",
    "latitude","longitude","address","road_address",
    "scenario_gen_start","scenario_gen_elapsed_sec",
    "sim_start","sim_elapsed_sec","exp_completed_at","coord_total_elapsed_sec","success","log_file",
    "num_patients","max_send_coeff","num_ambulances","num_uavs","amb_speed","uav_speed","amb_handover_time","uav_handover_time","api_duration_used","duration_coeff","sim_repeats","random_seed",
]
# (additional) stable dtype schema
SUMMARY_DTYPES = {
    "exp_id":"object","coord":"object","trial_no":"Int64","note":"object",
    "latitude":"float64","longitude":"float64","address":"object","road_address":"object",
    "scenario_gen_start":"object","scenario_gen_elapsed_sec":"float64",
    "sim_start":"object","sim_elapsed_sec":"float64",
    "exp_completed_at":"object","coord_total_elapsed_sec":"float64",
    "success":"object","log_file":"object",
    "num_patients":"Int64","max_send_coeff":"object","num_ambulances":"Int64","num_uavs":"Int64",
    "amb_speed":"float64","uav_speed":"float64","amb_handover_time":"float64","uav_handover_time":"float64","api_duration_used":"object","duration_coeff":"float64","sim_repeats":"Int64","random_seed":"Int64",
}


def _summary_paths(base_path: str, exp_id: str):
    """(NOTE) Some older copies might have returned a single string or 3 items.
    We keep this signature for backward-compat, but normalize it via _summary_paths_pair().
    """
    base = os.path.join(base_path, "scenarios", exp_id)
    main = os.path.join(base, f"{exp_id}_summary.csv")
    legacy = os.path.join(base, "exp_id_summary.csv")
    return (main, legacy)

def _summary_paths_pair(base_path: str, exp_id: str) -> Tuple[str, str]:
    """Always return exactly two strings, no matter what _summary_paths returns in user's local file."""
    p = _summary_paths(base_path, exp_id)
    base = os.path.join(base_path, "scenarios", exp_id)
    main_default = os.path.join(base, f"{exp_id}_summary.csv")
    legacy_default = os.path.join(base, "exp_id_summary.csv")

    # tuple/list
    if isinstance(p, (tuple, list)):
        if len(p) >= 2:
            return str(p[0]), str(p[1])
        elif len(p) == 1:
            return str(p[0]), legacy_default
        else:
            return main_default, legacy_default
    # single string?
    if isinstance(p, str):
        # if it's already "<exp>_summary.csv", compute legacy alongside
        if p.endswith("_summary.csv") and os.path.basename(p).startswith(exp_id):
            return p, legacy_default
        # else assume it's legacy-like, compute main
        return main_default, p
    # fallback
    return main_default, legacy_default

def _load_summary_df(path_main: str, path_legacy: str):
    if pd is None:
        return None
    for pth in (path_main, path_legacy):
        if exists_file(pth):
            for enc in ("utf-8-sig","cp949","utf-8"):
                try:
                    df = pd.read_csv(pth, encoding=enc)
                    # Fill missing columns + reorder
                    for c in SUMMARY_COLS:
                        if c not in df.columns:
                            df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
                    df = df.reindex(columns=SUMMARY_COLS)
                    # Force dtype (bulk -> fallback to per-column correction on failure)
                    try:
                        df = df.astype(SUMMARY_DTYPES)
                    except Exception:
                        for col, dt in SUMMARY_DTYPES.items():
                            try:
                                df[col] = df[col].astype(dt)
                            except Exception:
                                df[col] = df[col].astype("object")
                    return df
                except Exception:
                    continue
    import pandas as _pd
    # Ensure dtype even for empty DF
    empty = {c: _pd.Series(dtype=SUMMARY_DTYPES.get(c, "object")) for c in SUMMARY_COLS}
    return _pd.DataFrame(empty)


def _save_summary(path_main: str, df):
    ensure_dir(os.path.dirname(path_main))
    df[SUMMARY_COLS].to_csv(path_main, index=False, encoding="utf-8-sig")

def upsert_summary_row_dual(base_path: str, row: Dict[str,Any]):
    exp_id = row.get("exp_id")
    if not exp_id:
        return
    path_main, path_legacy = _summary_paths_pair(base_path, exp_id)
    if pd is None:
        exists = exists_file(path_main)
        ensure_dir(os.path.dirname(path_main))
        with open(path_main, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k,"") for k in SUMMARY_COLS})
        return

    df = _load_summary_df(path_main, path_legacy)
    key = (row.get("exp_id"), row.get("coord"))

    # Auto-calculate trial_no: max trial_no of same exp_id + coord pair + 1
    existing = df[(df["exp_id"]==key[0]) & (df["coord"]==key[1])]
    if not existing.empty:
        max_trial = existing["trial_no"].max()
        next_trial = 1 if pd.isna(max_trial) else int(max_trial) + 1
    else:
        next_trial = 1

    # Auto-assign if trial_no is not specified
    if "trial_no" not in row or row.get("trial_no") is None or row.get("trial_no") == 0:
        row["trial_no"] = next_trial

    # Always append as new row (no update)
    for c in SUMMARY_COLS:
        if c not in df.columns:
            df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
    df = df.reindex(columns=SUMMARY_COLS)

    next_idx = len(df)
    for k, v in row.items():
        df.loc[next_idx, k] = v

    _save_summary(path_main, df)

# ------------------------------------------------------------------
# YAML parsing
# ------------------------------------------------------------------

def _yaml_safe_load(path: str) -> Dict[str,Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def extract_params_from_yaml(config_path: str) -> Dict[str,Any]:
    meta = {
        "num_patients": None,
        "num_ambulances": None,
        "num_uavs": None,
        "amb_speed": None,
        "uav_speed": None,
        "amb_handover_time": None,
        "uav_handover_time": None,
        "api_duration_used": None,
        "duration_coeff": None,
        "sim_repeats": None,
        "random_seed": None,
        "max_send_coeff": None,
    }
    y = _yaml_safe_load(config_path) or {}
    run = y.get("run_setting", {}) or {}
    meta["sim_repeats"] = run.get("totalSamples")
    meta["random_seed"] = run.get("random_seed")

    ent = y.get("entity_info", {}) or {}
    patient = ent.get("patient", {}) or {}
    amb = ent.get("ambulance", {}) or {}
    uav = ent.get("uav", {}) or {}
    hosp = ent.get("hospital", {}) or {}

    # incident_size / speeds / handover_time / is_use_time / duration_coeff
    meta["num_patients"] = patient.get("incident_size")
    meta["amb_speed"] = amb.get("velocity")
    meta["uav_speed"] = uav.get("velocity")
    meta["amb_handover_time"] = amb.get("handover_time")
    meta["uav_handover_time"] = uav.get("handover_time")

    # Whether API duration is used (stored as string True/False)
    is_use_time_val = amb.get("is_use_time")
    if is_use_time_val is not None:
        meta["api_duration_used"] = str(is_use_time_val)
    else:
        meta["api_duration_used"] = None

    # Road data provider (kakao | osrm) - recorded at scenario generation
    # Older scenarios always used Kakao, so default to kakao if unspecified
    meta["road_provider"] = amb.get("road_provider") or "kakao"

    # duration_coeff (API duration time weight)
    duration_coeff_val = amb.get("duration_coeff")
    if duration_coeff_val is not None:
        meta["duration_coeff"] = float(duration_coeff_val)
    else:
        meta["duration_coeff"] = 1.0  # default value

    # max_send_coeff (handles both list and string)
    msc = hosp.get("max_send_coeff")
    if isinstance(msc, (list, tuple)):
        # Also handles case where list contains a single comma-separated string like "1.1, 1"
        if len(msc) == 1 and isinstance(msc[0], str) and "," in msc[0]:
            meta["max_send_coeff"] = msc[0]
        else:
            try:
                meta["max_send_coeff"] = ",".join(str(x) for x in msc)
            except Exception:
                meta["max_send_coeff"] = str(msc)
    elif isinstance(msc, str):
        meta["max_send_coeff"] = msc

    # CSV path resolution helper (./scenarios/... relative to project root, otherwise relative to config folder)
    def _resolve(p: Optional[str]) -> Optional[str]:
        if not p: return None
        p = str(p).strip()
        if os.path.isabs(p):
            return p
        if p.startswith("./"):
            parts = os.path.normpath(config_path).split(os.sep)
            if "scenarios" in parts:
                root = os.sep.join(parts[:parts.index("scenarios")])  # project root
                return os.path.normpath(os.path.join(root, p[2:]))
        # Regular relative paths are relative to the config folder
        return os.path.normpath(os.path.join(os.path.dirname(config_path), p))

    # num_ambulances/num_uavs from CSV row count
    amb_csv = _resolve(amb.get("dispatch_distance_info"))
    uav_csv = _resolve(uav.get("dispatch_distance_info"))
    try:
        if amb_csv and os.path.isfile(amb_csv) and pd is not None:
            meta["num_ambulances"] = len(pd.read_csv(amb_csv, encoding="utf-8-sig"))
    except Exception:
        pass
    try:
        if uav_csv and os.path.isfile(uav_csv) and pd is not None:
            # Current uav_info.csv schema stores exactly one row per UAV.
            # Use the row count directly for summary reporting.
            uav_df = pd.read_csv(uav_csv, encoding="utf-8-sig")
            meta["num_uavs"] = len(uav_df)
    except Exception:
        pass

    return meta


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------
import sys

def _pick_first_file(*candidates: Union[str, Path]) -> str:
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_file():
            return str(p)
    # Fallback: first candidate string for useful error messages.
    return str(candidates[0]) if candidates else ""


def _resolve_runtime_paths(base_path: str) -> Dict[str, str]:
    root = Path(base_path).resolve()
    this_file = Path(__file__).resolve()
    this_sce_dir = this_file.parent
    this_src_dir = this_sce_dir.parent

    make_script = _pick_first_file(
        root / "src" / "sce_src" / "make_csv_yaml_dynamic.py",
        root / "make_csv_yaml_dynamic.py",
        this_sce_dir / "make_csv_yaml_dynamic.py",
    )
    main_py = _pick_first_file(
        root / "src" / "sim_src" / "main.py",
        root / "main.py",
        this_src_dir / "sim_src" / "main.py",
    )

    return {
        "make_script": make_script,
        "main_py": main_py,
    }


class Orchestrator:
    def __init__(self, base_path: str, python_cmd: Optional[str] = None):
        self.base_path = os.path.abspath(base_path)
        self.python_cmd = python_cmd or sys.executable
        runtime_paths = _resolve_runtime_paths(self.base_path)
        self.paths = {
            "make_script": runtime_paths["make_script"],
            "main_py": runtime_paths["main_py"],
            "scenarios":   os.path.join(self.base_path, "scenarios"),
            "results":     os.path.join(self.base_path, "results"),
            "logs":        os.path.join(self.base_path, "experiment_logs"),
        }

    # ---------- scenario generation ----------
    def generate_scenario(self, latitude: float, longitude: float,
                          incident_size: int = 30, amb_count: int = 30, uav_count: int = 3,
                          amb_velocity: int = 40, uav_velocity: int = 80,
                          total_samples: int = 30, random_seed: int = 0,
                          exp_id: Optional[str] = None,
                          extra_env: Optional[Dict[str,str]] = None,
                          extra_args: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:

        if not exists_file(self.paths["make_script"]):
            raise FileNotFoundError(f"make_csv_yaml_dynamic.py not found: {self.paths['make_script']}")
        # exp_YYYYMMDDHHMM format (no underscore)
        exp_id = exp_id or ("exp_" + datetime.now(KST).strftime("%Y%m%d%H%M"))

        cmd = [
            self.python_cmd, "-X", "utf8", self.paths["make_script"],
            "--base_path", self.base_path,
            "--latitude", str(latitude),
            "--longitude", str(longitude),
            "--incident_size", str(incident_size),
            "--amb_count", str(amb_count),
            "--uav_count", str(uav_count),
            "--amb_velocity", str(amb_velocity),
            "--uav_velocity", str(uav_velocity),
            "--total_samples", str(total_samples),
            "--random_seed", str(random_seed),
            "--experiment_id", exp_id,
        ]

        if extra_args:
            for k, v in extra_args.items():
                if v is None: 
                    continue
                cmd.extend([f"--{k}", str(v)])

        env = os.environ.copy()
        if extra_env:
            env.update({str(k):str(v) for k,v in extra_env.items()})

        t1 = time.time()
        started_at = now_kst_iso()
        proc = subprocess.run(cmd, cwd=self.base_path, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="ignore")
        t2 = time.time()
        elapsed = round(t2 - t1, 3)

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        _, config_path, api_call_count = parse_make_generator_stdout(stdout)
        if not config_path:
            raise RuntimeError(f"CONFIG_PATH not found in generator stdout.\n[stdout]\n{stdout}\n[stderr]\n{stderr}")

        coord = parse_coord_from_config_path(config_path) or to_coord_str(latitude, longitude)
        exp_id2 = parse_exp_from_config_path(config_path) or exp_id

        # Perform reverse geocoding directly (Kakao API)
        coord_info = None
        kakao_api_key = (extra_args or {}).get("kakao_api_key")
        if kakao_api_key:
            try:
                coord_info = reverse_geocode_kakao(latitude, longitude, kakao_api_key)
                print(f"✅ Reverse geocoding success: {coord_info.get('full_address', '')}")
            except Exception as e:
                print(f"⚠️ Reverse geocoding failed: {e}")
                coord_info = {
                    "full_address": "reverse geocoding failed",
                    "road_address": "",
                    "area1": "", "area2": "", "area3": "", "area4": "",
                    "is_valid": False,
                    "latitude": latitude,
                    "longitude": longitude,
                    "api_response_code": -999
                }
        else:
            print("⚠️ No Kakao API key. Skipping reverse geocoding")
            coord_info = {
                "full_address": "",
                "road_address": "",
                "area1": "", "area2": "", "area3": "", "area4": "",
                "is_valid": False,
                "latitude": latitude,
                "longitude": longitude,
                "api_response_code": -999
            }

        # Summary path (main + legacy) — robust
        summary_main, summary_legacy = _summary_paths_pair(self.base_path, exp_id2)

        # Log file name: (coord)_%y%m%d_%H%M%S.txt
        log_file = os.path.join(self.paths["logs"], f"{coord}_{ts_short_now()}.txt")
        ensure_dir(os.path.dirname(log_file))
        log_head = []
        log_head.append(f"=== SCENARIO_GEN_START {started_at} ===\n")
        if coord_info is not None:
            log_head.append(f"COORDINATE_INFO:{json.dumps(coord_info, ensure_ascii=False)}\n")
        log_head.append(f"CONFIG_PATH:{config_path}\n")
        log_head.append(f"--- stdout ---\n{stdout}\n")
        if stderr.strip():
            log_head.append(f"--- stderr ---\n{stderr}\n")
        log_head.append(f"=== SCENARIO_GEN_END {now_kst_iso()} (elapsed: {elapsed}s) ===\n\n")
        write_text(log_file, "".join(log_head), encoding="utf-8")

        # Initial summary row (attempt not assigned yet → 0)
        row = {
            "exp_id": exp_id2,
            "coord": coord,
            "trial_no": 0,
            "note": "scenario generated",
            "latitude": coord_info.get("latitude") if coord_info else latitude,
            "longitude": coord_info.get("longitude") if coord_info else longitude,
            "address": (coord_info or {}).get("full_address",""),
            "road_address": (coord_info or {}).get("road_address",""),
            "scenario_gen_start": started_at,
            "scenario_gen_elapsed_sec": elapsed,
            "sim_start": None,
            "sim_elapsed_sec": None,
            "exp_completed_at": None,
            "coord_total_elapsed_sec": None,
            "success": None,
            "log_file": log_file,
        }
        meta = extract_params_from_yaml(config_path)
        row.update({
            "num_patients": meta.get("num_patients"),
            "max_send_coeff": meta.get("max_send_coeff"),
            "num_ambulances": meta.get("num_ambulances"),
            "num_uavs": meta.get("num_uavs"),
            "amb_speed": meta.get("amb_speed"),
            "uav_speed": meta.get("uav_speed"),
            "amb_handover_time": meta.get("amb_handover_time"),
            "uav_handover_time": meta.get("uav_handover_time"),
            "api_duration_used": meta.get("api_duration_used"),
            "duration_coeff": meta.get("duration_coeff"),
            "sim_repeats": meta.get("sim_repeats"),
            "random_seed": meta.get("random_seed"),
        })
        upsert_summary_row_dual(self.base_path, row)

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "exp_id": exp_id2,
            "coord": coord,
            "config_path": config_path,
            "api_call_count": api_call_count,
            "summary_csv_path": summary_main,
            "summary_csv_path_legacy": summary_legacy,
            "log_file": log_file,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_sec": elapsed,
            "started_at": started_at,
        }

    # ---------- simulation run ----------
    def run_simulation(self, config_path: str, extra_env: Optional[Dict[str,str]] = None) -> Dict[str,Any]:

        if not exists_file(self.paths["main_py"]):
            raise FileNotFoundError(f"main.py not found: {self.paths['main_py']}")
        if not exists_file(config_path):
            raise FileNotFoundError(f"config_path not found: {config_path}")

        exp_id2 = parse_exp_from_config_path(config_path)
        coord2 = parse_coord_from_config_path(config_path)
        if not exp_id2 or not coord2:
            raise ValueError("Cannot extract exp_id/coord from config_path.")

        summary_main, summary_legacy = _summary_paths_pair(self.base_path, exp_id2)

        cmd = [
            self.python_cmd, "-X", "utf8", self.paths["main_py"],
            "--config_path", config_path,
        ]
        env = os.environ.copy()
        if extra_env:
            env.update({str(k):str(v) for k,v in extra_env.items()})

        sim_started = now_kst_iso()
        t1 = time.time()

        # Use Popen + threads to prevent pipe buffer deadlock (capture_output=True deadlocks when stdout exceeds 64KB)
        import threading as _threading
        _stdout_buf: list = []
        _stderr_buf: list = []

        proc = subprocess.Popen(cmd, cwd=self.base_path, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="ignore")

        def _read_pipe(pipe, buf):
            for line in pipe:
                buf.append(line)

        _t_out = _threading.Thread(target=_read_pipe, args=(proc.stdout, _stdout_buf), daemon=True)
        _t_err = _threading.Thread(target=_read_pipe, args=(proc.stderr, _stderr_buf), daemon=True)
        _t_out.start(); _t_err.start()

        proc.wait()
        _t_out.join(); _t_err.join()

        t2 = time.time()
        elapsed = round(t2 - t1, 3)
        ok = (proc.returncode == 0)

        stdout_text = "".join(_stdout_buf)
        stderr_text = "".join(_stderr_buf)

        # Per-run log file
        log_file = os.path.join(self.paths["logs"], f"{coord2}_{ts_short_now()}.txt")
        pieces = []
        pieces.append(f"=== SIM_START {sim_started} ===\n")
        if stdout_text:
            pieces.append(stdout_text)
            if not stdout_text.endswith("\n"):
                pieces.append("\n")
        if stderr_text.strip():
            pieces.append(f"--- stderr ---\n{stderr_text}\n")
        pieces.append(f"=== SIM_END {now_kst_iso()} (elapsed: {elapsed}s, rc={proc.returncode}) ===\n\n")
        write_text(log_file, "".join(pieces), encoding="utf-8")

        # Namespace for proc attribute compatibility (code below does not reference proc.stdout etc., so not strictly needed)
        class _ProcCompat:
            returncode = proc.returncode
            stdout = stdout_text
            stderr = stderr_text
        proc = _ProcCompat()

        # Summary update (attempt-aware)
        if pd is not None:
            df = _load_summary_df(summary_main, summary_legacy)
            mask_pair = (df["exp_id"]==exp_id2) & (df["coord"]==coord2)

            def _base_fields_for_pair() -> Dict[str,Any]:
                # Always read latest params from YAML (reflects modified params on re-run)
                meta = extract_params_from_yaml(config_path)

                if mask_pair.any():
                    d0 = df[mask_pair].sort_index().iloc[0].to_dict()
                    return {
                        "exp_id": exp_id2, "coord": coord2,
                        "latitude": d0.get("latitude"), "longitude": d0.get("longitude"),
                        "address": d0.get("address"), "road_address": d0.get("road_address"),
                        "scenario_gen_start": d0.get("scenario_gen_start"),
                        "scenario_gen_elapsed_sec": d0.get("scenario_gen_elapsed_sec"),
                        # Use latest params read from YAML (reflects modified values)
                        "num_patients": meta.get("num_patients"),
                        "max_send_coeff": meta.get("max_send_coeff"),
                        "num_ambulances": meta.get("num_ambulances"),
                        "num_uavs": meta.get("num_uavs"),
                        "amb_speed": meta.get("amb_speed"),
                        "uav_speed": meta.get("uav_speed"),
                        "amb_handover_time": meta.get("amb_handover_time"),
                        "uav_handover_time": meta.get("uav_handover_time"),
                        "api_duration_used": meta.get("api_duration_used"),
                        "duration_coeff": meta.get("duration_coeff"),
                        "sim_repeats": meta.get("sim_repeats"),
                        "random_seed": meta.get("random_seed"),
                    }
                else:
                    return {
                        "exp_id": exp_id2, "coord": coord2,
                        "latitude": None, "longitude": None, "address": "", "road_address": "",
                        "scenario_gen_start": None, "scenario_gen_elapsed_sec": None,
                        "num_patients": meta.get("num_patients"),
                        "max_send_coeff": meta.get("max_send_coeff"),
                        "num_ambulances": meta.get("num_ambulances"),
                        "num_uavs": meta.get("num_uavs"),
                        "amb_speed": meta.get("amb_speed"),
                        "uav_speed": meta.get("uav_speed"),
                        "amb_handover_time": meta.get("amb_handover_time"),
                        "uav_handover_time": meta.get("uav_handover_time"),
                        "api_duration_used": meta.get("api_duration_used"),
                        "duration_coeff": meta.get("duration_coeff"),
                        "sim_repeats": meta.get("sim_repeats"),
                        "random_seed": meta.get("random_seed"),
                    }

            # Always append as new row (trial_no increments even on re-experiment)
            mask_first = mask_pair & (df["trial_no"].fillna(0)==0)
            if mask_first.any():
                # First run: update trial_no from 0 to 1
                # Read and update latest params from YAML (reflects modified values)
                meta_first = extract_params_from_yaml(config_path)
                df.loc[mask_first, "trial_no"] = 1
                df.loc[mask_first, "note"] = "trial_1 " + ("success" if ok else "failure")
                df.loc[mask_first, "sim_start"] = sim_started
                df.loc[mask_first, "sim_elapsed_sec"] = elapsed
                try:
                    prev = float(df.loc[mask_first, "scenario_gen_elapsed_sec"].values[0] or 0.0)
                except Exception:
                    prev = 0.0
                df.loc[mask_first, "coord_total_elapsed_sec"] = round(prev + float(elapsed), 3)
                df.loc[mask_first, "exp_completed_at"] = now_kst_iso()
                df.loc[mask_first, "success"] = bool(ok)
                df.loc[mask_first, "log_file"] = log_file
                # Apply latest params read from YAML
                df.loc[mask_first, "num_patients"] = meta_first.get("num_patients")
                df.loc[mask_first, "max_send_coeff"] = meta_first.get("max_send_coeff")
                df.loc[mask_first, "num_ambulances"] = meta_first.get("num_ambulances")
                df.loc[mask_first, "num_uavs"] = meta_first.get("num_uavs")
                df.loc[mask_first, "amb_speed"] = meta_first.get("amb_speed")
                df.loc[mask_first, "uav_speed"] = meta_first.get("uav_speed")
                df.loc[mask_first, "amb_handover_time"] = meta_first.get("amb_handover_time")
                df.loc[mask_first, "uav_handover_time"] = meta_first.get("uav_handover_time")
                df.loc[mask_first, "api_duration_used"] = meta_first.get("api_duration_used")
                df.loc[mask_first, "duration_coeff"] = meta_first.get("duration_coeff")
                df.loc[mask_first, "sim_repeats"] = meta_first.get("sim_repeats")
                df.loc[mask_first, "random_seed"] = meta_first.get("random_seed")

            else:
                # Re-experiment: always add new row (regardless of success)
                import pandas as _pd
                prev_rows = df[mask_pair].sort_values("trial_no")
                next_try = int(_pd.to_numeric(prev_rows["trial_no"], errors="coerce").fillna(0).max()) + 1 if not prev_rows.empty else 1
                base = _base_fields_for_pair()
                newrow = {
                    **base,
                    "trial_no": next_try,
                    "note": f"trial_{next_try} " + ("success" if ok else "failure"),
                    "sim_start": sim_started,
                    "sim_elapsed_sec": elapsed,
                    "exp_completed_at": now_kst_iso(),
                    "coord_total_elapsed_sec": None,
                    "success": bool(ok),
                    "log_file": log_file,
                }
                try:
                    prev = float(base.get("scenario_gen_elapsed_sec") or 0.0)
                except Exception:
                    prev = 0.0
                newrow["coord_total_elapsed_sec"] = round(prev + float(elapsed), 3)
                # Add row without concat
                for c in SUMMARY_COLS:
                    if c not in df.columns:
                        df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
                df = df.reindex(columns=SUMMARY_COLS)

                next_idx = len(df)
                for k, v in newrow.items():
                    df.loc[next_idx, k] = v


            _save_summary(summary_main, df)

        return {
            "ok": ok,
            "returncode": proc.returncode,
            "exp_id": exp_id2,
            "coord": coord2,
            "config_path": config_path,
            "summary_csv_path": summary_main,
            "summary_csv_path_legacy": summary_legacy,
            "log_file": log_file,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "elapsed_sec": elapsed,
            "started_at": sim_started,
        }
