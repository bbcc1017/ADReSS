# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import requests
from haversine import haversine
# [ADD] ──────────────────────────────────────────────────────────────
import re
from datetime import timezone, timedelta, datetime
from typing import Optional

KST = timezone(timedelta(hours=9))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slugify(name: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\-\s]", "", str(name))
    s = re.sub(r"\s+", "_", s).strip("_")
    return (s[:maxlen] or "noname")

def save_route_json(meta: dict, payload: Optional[dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    data = {"meta": meta, "payload": {"naver_response": payload} if payload else None}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# [ADD END] ─────────────────────────────────────────────────────────


def str2bool(v):
    """String-to-bool conversion for argparse. Raises ArgumentTypeError for invalid values."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def parse_util_map(text: str):
    """
    "1:0.90,11:0.75,etc:0.60" -> {1:0.9, 11:0.75, "etc":0.6}
    """
    if not text:
        return None
    m = {}
    for part in str(text).split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            val = float(v)
        except Exception:
            continue
        if k.lower() == "etc":
            m["etc"] = val
        else:
            try:
                m[int(k)] = val
            except Exception:
                pass
    return m if m else None

class ScenarioGenerator:
    """Dynamic parameter-based scenario generation class (cross-platform compatible)"""

    def __init__(self, base_path, experiment_id=None, kakao_api_key=None, departure_time=None,
                 osrm_url=None, road_provider=None, is_use_time=True):
        # Actual road API call counter (shared by Kakao + OSRM)
        self.api_call_count = 0
        # Resolve project path to absolute
        self.base_path = os.path.abspath(base_path)

        # Determine road data provider (needed for experiment_id suffix)
        # Priority: explicit road_provider > is_use_time flag
        is_use_time = bool(is_use_time) if not isinstance(is_use_time, str) else str2bool(is_use_time)
        if road_provider is None:
            road_provider = "kakao" if is_use_time else "osrm"
        self.road_provider = road_provider

        # experiment_id generation:
        #   - kakao mode + departure_time -> exp_<base>_dep_<YYYYMMDDHHMM>
        #   - osrm  mode                 -> exp_<base>_osrm
        #   - if an appropriate suffix already exists, keep it as-is (idempotent)
        if experiment_id:
            base_exp_id = str(experiment_id).strip()
            if base_exp_id.startswith("exp_"):
                base_exp_id = base_exp_id[4:]
            # Normalize spaces to underscores; preserve existing underscores
            base_exp_id = re.sub(r"\s+", "_", base_exp_id).strip("_")
            if not base_exp_id:
                base_exp_id = datetime.now().strftime("%Y%m%d%H%M")
        else:
            base_exp_id = datetime.now().strftime("%Y%m%d%H%M")

        if self.road_provider == "osrm":
            # If the same base has a Kakao _dep_ suffix, remove it and attach _osrm
            base_exp_id = re.sub(r"_dep_\d{12}$", "", base_exp_id)
            if base_exp_id.endswith("_osrm"):
                self.experiment_id = f"exp_{base_exp_id}"
            else:
                self.experiment_id = f"exp_{base_exp_id}_osrm"
        else:  # kakao
            # Conversely, if _osrm suffix exists, remove it and attach _dep_
            base_exp_id = re.sub(r"_osrm$", "", base_exp_id)
            if departure_time and f"_dep_{departure_time}" not in base_exp_id:
                self.experiment_id = f"exp_{base_exp_id}_dep_{departure_time}"
            else:
                self.experiment_id = f"exp_{base_exp_id}"

        # Kakao API key setup
        self.kakao_api_key = kakao_api_key
        self.departure_time = departure_time  # YYYYMMDDHHMM format

        # OSRM backend setup (used when is_use_time=False)
        self.osrm_url = (osrm_url
                         or os.environ.get("MCI_OSRM_URL", "https://router.project-osrm.org"))
        
        # Data file paths (set as absolute paths)
        self.scenarios_path = os.path.join(self.base_path, "scenarios")
        self.fire_data_path = os.path.join(self.scenarios_path, "fire_stations.csv")
        self.hospital_data_path = os.path.join(self.scenarios_path, "hospital_master_data.xlsx")
        
        # Validate required data files exist
        self._validate_data_files()

        # Patient configuration (hardcoded)
        self.patient_config = {
            "ratio": {"Red": 0.1, "Yellow": 0.3, "Green": 0.5, "Black": 0.1},
            "rescue_param": {"Red": (6, 5), "Yellow": (2, 13), "Green": (1, 22), "Black": (0, 0)},
            "treat_tier3": {"Red": True, "Yellow": True, "Green": True, "Black": True},
            "treat_tier2": {"Red": False, "Yellow": True, "Green": True, "Black": True},
            "treat_tier3_mean": {"Red": 40, "Yellow": 20, "Green": 10, "Black": 0},
            "treat_tier2_mean": {"Red": float('inf'), "Yellow": 30, "Green": 15, "Black": 0}
        }
        
        # Candidate expansion multiplier (reduces AMB road distance API calls)
        self.multiplier = 1.5

        # --- ENV injection (passed from PS) ---
        # util_by_tier: e.g. "1:0.656,11:0.461,etc:0.461"
        env_util = parse_util_map(os.environ.get("MCI_UTIL_BY_TIER", ""))
        self.util_by_tier = env_util or {1: 0.656, 11: 0.461, "etc": 0.461}

        # queue_policy: "0" | "capa/2" | "0.5" etc.
        # self.queue_policy = os.environ.get("MCI_QUEUE_POLICY", "0")

        # buffer_ratio: float
        try:
            self.buffer_ratio = float(os.environ.get("MCI_BUFFER_RATIO", "1.5"))
        except Exception:
            self.buffer_ratio = 1.5
        
        # (added) max_send_coeff default input path: ENV -> default value
        self.max_send_coeff_text = os.environ.get("MCI_MAX_SEND_COEFF", "1,1")
        
        print(f"📁 Project path: {self.base_path}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
        print(f"buffer_ratio={self.buffer_ratio}")

    def _validate_data_files(self):
        """Validate that all required data files exist"""
        required_files = [
            (self.fire_data_path, "Fire station data"),
            (self.hospital_data_path, "Hospital data")
        ]
        missing_files = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing_files.append(f"{description}: {file_path}")
        if missing_files:
            print("❌ The following required files are missing:")
            for missing in missing_files:
                print(f"   • {missing}")
            raise FileNotFoundError("Please check the required data files.")
        print("✅ All required data files verified")

    def get_road_distance(self, start, end, **kwargs):
        """Road distance/time dispatcher. Delegates to kakao/osrm based on self.road_provider.

        Returns:
            (distance_km, duration_min) tuple
        """
        provider = (self.road_provider or "kakao").lower()
        if provider == "osrm":
            return self.get_road_distance_osrm(start, end, **kwargs)
        return self.get_road_distance_kakao(start, end, **kwargs)

    def get_road_distance_kakao(self, start, end, max_retries=3, save_json_dir=None, route_type=None, source_index=None, name=None, start_label="start", goal_label="goal"):
        """Road distance and time calculation using Kakao Mobility API (with retry logic)

        Args:
            start: (lat, lon) tuple
            end: (lat, lon) tuple
            save_json_dir: Directory to save JSON responses
            route_type: "center2site" or "hos2site"

        Returns:
            (distance_km, duration_min) tuple - distance (km) and transport time (min)
        """
        if not self.kakao_api_key:
            raise RuntimeError(
                "Kakao API key is missing (is_use_time=True mode). "
                "If you don't have a key, use is_use_time=False to use the OSRM backend."
            )

        url = "https://apis-navi.kakaomobility.com/v1/future/directions"
        headers = {
            "Authorization": f"KakaoAK {self.kakao_api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "origin": f"{start[1]},{start[0]}",  # lon,lat order
            "destination": f"{end[1]},{end[0]}",
            "priority": "TIME",  # shortest time priority
            "car_fuel": "GASOLINE",
            "car_hipass": "false",
            "alternatives": "false",
            "road_details": "false"
        }

        # Add departure_time parameter (realtime or future time)
        if self.departure_time:
            params["departure_time"] = self.departure_time

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    # Kakao API response structure: routes[0].summary
                    if not data.get("routes") or len(data["routes"]) == 0:
                        raise RuntimeError(
                            f"Kakao API no route ({start} -> {end}): "
                            f"Coordinates may be on sea/island or road-disconnected segment."
                        )

                    route = data["routes"][0]
                    result_code = route.get("result_code", 0)
                    if result_code != 0:
                        raise RuntimeError(
                            f"Kakao API no route (result_code={result_code}, {start} -> {end}): "
                            f"Segment unreachable without ferry."
                        )
                    summary = route.get("summary", {})

                    # Convert distance (m) -> km
                    distance_km = summary.get("distance", 0) / 1000.0

                    # Convert time (sec) -> min
                    duration_sec = summary.get("duration", 0)
                    duration_min = duration_sec / 60.0

                    # Save JSON
                    if save_json_dir:
                        now = datetime.now(KST).isoformat()
                        meta = {
                            "api_provider": "kakao",
                            "route_type": route_type,
                            "source_index": source_index,
                            "name": name,
                            # Coordinates stored in [lon, lat] format
                            start_label: [start[1], start[0]],
                            goal_label: [end[1], end[0]],
                            "departure_time": self.departure_time or "realtime",
                            "priority": params.get("priority"),
                            "saved_at": now,
                            # Summary fields
                            "distance_km": round(distance_km, 3),
                            "duration_min": round(duration_min, 2),
                            "duration_sec": duration_sec,
                            "toll_fare": summary.get("fare", {}).get("toll", 0),
                            "taxi_fare": summary.get("fare", {}).get("taxi", 0),
                            "direction_note": f"{start_label}->{goal_label}"
                        }
                        fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                        out_path = os.path.join(save_json_dir, fname)

                        # Save Kakao response
                        ensure_dir(os.path.dirname(out_path))
                        json_data = {
                            "meta": meta,
                            "payload": {"kakao_response": data}
                        }
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)

                        print(f"  📦 [{route_type}] idx={source_index:03d} {name} → {distance_km:.2f}km, {duration_min:.1f}min")

                    self.api_call_count += 1
                    return distance_km, duration_min

                elif response.status_code == 401:
                    raise RuntimeError(
                        f"Kakao API authentication failed (401): Please check your API key."
                    )
                elif response.status_code == 429:
                    print(f"  ⚠️ API rate limit exceeded (429): waiting 3 seconds...")
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"Kakao API call failed (status {response.status_code}): {start} -> {end}"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ API call error ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Kakao API call failed (exceeded {max_retries} retries): {e}"
                    ) from e

        raise RuntimeError(
            f"Kakao API rate limit exceeded (429): failed after {max_retries} retries. "
            f"Daily quota has been exhausted."
        )

    def get_road_distance_osrm(self, start, end, max_retries=3, save_json_dir=None,
                                route_type=None, source_index=None, name=None,
                                start_label="start", goal_label="goal"):
        """Road distance and time calculation using OSRM HTTP API (/route/v1/driving).

        Uses the same signature/return values/JSON save schema as the Kakao function.
        OSRM is based on a static road graph, so real-time congestion/toll info is unavailable.

        Args:
            start: (lat, lon) tuple
            end: (lat, lon) tuple

        Returns:
            (distance_km, duration_min)
        """
        base = (self.osrm_url or "https://router.project-osrm.org").rstrip("/")
        url = f"{base}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "annotations": "false",
            "alternatives": "false",
        }
        headers = {"Accept": "application/json"}

        last_err = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    code = data.get("code")
                    routes = data.get("routes") or []
                    if code != "Ok" or not routes:
                        raise RuntimeError(
                            f"OSRM no route ({start} -> {end}, code={code}): "
                            f"Road-disconnected segment or OSRM server does not cover this area."
                        )

                    route = routes[0]
                    distance_m = float(route.get("distance", 0))
                    duration_s = float(route.get("duration", 0))
                    distance_km = distance_m / 1000.0
                    duration_min = duration_s / 60.0
                    duration_sec = int(round(duration_s))

                    if save_json_dir:
                        now = datetime.now(KST).isoformat()
                        meta = {
                            "api_provider": "osrm",
                            "route_type": route_type,
                            "source_index": source_index,
                            "name": name,
                            # Coordinates in [lon, lat] format, same as Kakao
                            start_label: [start[1], start[0]],
                            goal_label: [end[1], end[0]],
                            "departure_time": "static",   # OSRM has no time-of-day concept
                            "priority": "shortest_time",
                            "saved_at": now,
                            "distance_km": round(distance_km, 3),
                            "duration_min": round(duration_min, 2),
                            "duration_sec": duration_sec,
                            "toll_fare": 0,                # Not provided by OSRM
                            "taxi_fare": 0,                # Not provided by OSRM
                            "direction_note": f"{start_label}->{goal_label}",
                        }
                        fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                        out_path = os.path.join(save_json_dir, fname)
                        ensure_dir(os.path.dirname(out_path))
                        json_data = {"meta": meta, "payload": {"osrm_response": data}}
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)

                        print(f"  📦 [{route_type}] idx={(source_index or 0):03d} {name} → "
                              f"{distance_km:.2f}km, {duration_min:.1f}min (OSRM)")

                    self.api_call_count += 1
                    return distance_km, duration_min

                elif response.status_code == 429:
                    print(f"  ⚠️ OSRM rate limit exceeded (429): waiting 3 seconds...")
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"OSRM call failed (status {response.status_code}): {start} -> {end}"
                    )

            except RuntimeError:
                raise
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    print(f"  ⚠️ OSRM call error ({attempt+1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"OSRM call failed (exceeded {max_retries} retries): {e}"
                    ) from e

        raise RuntimeError(
            f"OSRM call failed (still failing after {max_retries} retries): last_err={last_err}"
        )

    def make_amb_info(self, latitude, longitude, incident_size, amb_count, save_folder):
        """Generate ambulance information"""
        print(f"  🚑 Generating ambulance info...")
        try:
            df = pd.read_csv(self.fire_data_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"❌ Failed to load fire station data: {e}")
            return
        
        # Duplicate center rows based on 'num_vehicles' (reflects ambulance unit count)
        # ------------------------------------------------------------
        if "num_vehicles" in df.columns:
            df["num_vehicles"] = pd.to_numeric(df["num_vehicles"], errors="coerce").fillna(1).astype(int)
            df.loc[df["num_vehicles"] < 1, "num_vehicles"] = 1
        else:
            df["num_vehicles"] = 1
        df["num_vehicles_owned"] = df["num_vehicles"]

        # Duplicate centers by quantity
        df = df.loc[df.index.repeat(df["num_vehicles"])].copy()
        # ------------------------------------------------------------
        coords = list(zip(df["y_coord"], df["x_coord"]))
        euc_distances = [haversine(coord, (latitude, longitude)) for coord in coords]
        df["euclidean_distance"] = euc_distances

        # Save EUC
        # df_sorted_euc = df.sort_values("euclidean_distance").head(incident_size).copy()
        df_sorted_euc = df.sort_values("euclidean_distance").head(amb_count).copy()
        df_sorted_euc = df_sorted_euc.rename(columns={
            "euclidean_distance": "init_distance",
            "station_name": "fire_station_name"
        })
        df_sorted_euc = df_sorted_euc.reset_index(drop=True)
        df_sorted_euc = df_sorted_euc[["init_distance", "fire_station_name", "num_vehicles_owned"]]
        euc_save_path = os.path.join(save_folder, "amb_info_euc.csv")
        df_sorted_euc.to_csv(euc_save_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        # [ADD] center2site save folder
        routes_dir = os.path.join(save_folder, "routes", "center2site")
        ensure_dir(routes_dir)

        # Candidate expansion and road distance/time calculation
        df_candidates = df.sort_values("euclidean_distance").head(int(incident_size * self.multiplier)).copy()
        road_distances = []
        road_durations = []

        # for j, (_, row) in enumerate(df_candidates.iterrows()):
        #     coord = (row["y_coord"], row["x_coord"])  # (lat, lon) of center
        #     dist_km, duration_min = self.get_road_distance_kakao(
        #         start=coord, end=(latitude, longitude),  # center → site
        #         save_json_dir=routes_dir, route_type="center2site",
        #         source_index=j, name=row.get("station_name", f"center_{j}"),
        #         start_label="center", goal_label="site"
        #     )
        cache = {}  # key: (center_lat, center_lon) -> (dist_km, duration_min)
        for j, (_, row) in enumerate(df_candidates.iterrows()):
            coord = (row["y_coord"], row["x_coord"])  # (lat, lon)
            key = coord
            if key in cache:
                dist_km, duration_min = cache[key]
            else:
                dist_km, duration_min = self.get_road_distance(
                    start=coord, end=(latitude, longitude),  # center → site
                    save_json_dir=routes_dir, route_type="center2site",
                    source_index=j, name=row.get("station_name", f"center_{j}"),
                    start_label="center", goal_label="site"
                )
                cache[key] = (dist_km, duration_min)
            road_distances.append(dist_km)
            road_durations.append(duration_min)
            time.sleep(0.05)

        df_candidates["road_distance"] = road_distances
        df_candidates["road_duration"] = road_durations

        # Save ROAD (sorted by duration, select top incident_size)
        # df_sorted_road = df_candidates.sort_values("road_duration").head(incident_size).copy()
        df_sorted_road = df_candidates.sort_values("road_duration").head(amb_count).copy()
        df_sorted_road = df_sorted_road.rename(columns={
            "road_distance": "init_distance",
            "road_duration": "duration",
            "station_name": "fire_station_name"
        })
        df_sorted_road = df_sorted_road.reset_index(drop=True)
        df_sorted_road = df_sorted_road[["init_distance", "duration", "fire_station_name", "num_vehicles_owned"]]
        road_save_path = os.path.join(save_folder, "amb_info_road.csv")
        df_sorted_road.to_csv(road_save_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        print(f"  ✅ Ambulance info generation complete")

    def make_hospital_info(self, latitude, longitude, incident_size, save_folder, uav_count=0):
        """Generate hospital information (existing logic + minimum condition guarantees)

        Args:
            latitude: Incident site latitude
            longitude: Incident site longitude
            incident_size: Number of patients
            save_folder: Output folder
            uav_count: Number of UAVs (used to guarantee minimum helipad hospitals)
        """
        print(f"  🏥 Generating hospital info...")
        
        # ---------- (0) Load data ----------
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine='openpyxl')
        except Exception as e:
            print(f"❌ Failed to load hospital data: {e}")
            return

        # Use only required columns (preserve names)
        cols_needed = ["institution_name", "type_code", "num_er_beds", "x_coord", "y_coord"]
        for c in cols_needed:
            if c not in df_full.columns:
                raise KeyError(f"Required column missing: {c}")
        df = df_full[cols_needed].copy()

        # Add helipad column (include if present, fill with 0 otherwise)
        if "helipad" in df_full.columns:
            df["helipad"] = df_full["helipad"].fillna(0).astype(int)
        else:
            df["helipad"] = 0  # No helipad info available, default to 0

        # ---------- (1) Calculate Euclidean distance ----------
        coords = list(zip(df["y_coord"], df["x_coord"]))  # (lat, lon)
        df["euclidean_distance"] = [haversine((lat, lon), (latitude, longitude)) for (lat, lon) in coords]

        # ---------- (2) Parameters ----------
        util_by_tier = getattr(self, "util_by_tier", {1: 0.656, 11: 0.461, "etc": 0.461})
        # queue_policy = str(getattr(self, "queue_policy", "0")).strip()
        try:
            buffer_ratio = float(getattr(self, "buffer_ratio", 1.5))
        except Exception:
            buffer_ratio = 1.5

        ratio = self.patient_config.get("ratio", {"Red":0.1,"Yellow":0.3,"Green":0.5,"Black":0.1})
        U = int(round(incident_size * float(ratio.get("Red", 0))))
        N = int(incident_size)
        
        import math
        def _get_util(code):
            try:
                icode = int(code)
                return util_by_tier.get(icode, util_by_tier.get("etc", 0.461))
            except Exception:
                return util_by_tier.get("etc", 0.461)
            
        df["util"] = df["type_code"].apply(_get_util)
        df["capa"] = (df["num_er_beds"] * (1 - df["util"])).apply(lambda x: int(max(0, math.floor(x))))
        # Operating rooms fixed by hospital type code
        conditions = [df['type_code'] == 1, df['type_code'] == 11]; values = [3, 2]
        df['operating_rooms'] = np.select(conditions, values, default=1)
        df["eff"] = df["operating_rooms"] + df["capa"]
        df["is_tier3"] = (df["type_code"].astype(str).astype(float).astype(int) == 1).astype(int)
        
        # ---------- (3) Global Tier3 capacity check (early impossibility detection) ----------
        total_tier3_capa_all = int(df.loc[df["is_tier3"]==1, "capa"].sum())
        total_capa_all = int(df["capa"].sum())
        if total_tier3_capa_all < U:
            print(f"  ⚠️ Global Tier3 capacity insufficient: Tier3_capa_all={total_tier3_capa_all} < U={U}. Proceeding with best selection (transfer failure possible).")
        
        # --- (4) Candidate expansion: same as existing code ---
        # Generous candidate pool (df_cand) including nearby hospitals
        df_sorted = df.sort_values("euclidean_distance").reset_index(drop=True)
        sum_capa = 0; sum_capa_tier3 = 0; cand_idx = []; 
        for i, row in df_sorted.iterrows():
            cand_idx.append(i)
            sum_capa += int(row["eff"])
            if row["is_tier3"] == 1: sum_capa_tier3 += int(row["eff"]); 
            if (sum_capa >= N * buffer_ratio): break
        if not cand_idx:
            cand_idx = list(range(len(df_sorted)))
        df_cand = df_sorted.loc[cand_idx].copy()
        
        df_selected = df_cand.copy()


        # ================================================================= #
        # Check if the selected list satisfies minimum conditions; add more if insufficient
        # Rule 1: Guarantee at least 2 advanced general hospitals (Tier 3)
        final_tier3 = df_selected[df_selected["is_tier3"] == 1]
        num_to_ensure_tier3 = 2 - len(final_tier3)
        if num_to_ensure_tier3 > 0:
            print(f"  INFO: Final list has {len(final_tier3)} Tier3 hospitals. Adding to reach minimum of 2.")
            # Find the nearest unselected Tier3 hospitals from the full list and add until at least 2
            candidates = df_sorted[(df_sorted["is_tier3"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                hospitals_to_add = candidates.head(num_to_ensure_tier3)
                df_selected = pd.concat([df_selected, hospitals_to_add])

        # Rule 2: Guarantee Tier 3 hospitals can accommodate 40% of patients (considering min 10% red patients + probability distribution)
        target_capa = N * 0.4
        current_capa = df_selected[df_selected["is_tier3"] == 1]["eff"].sum()
        while current_capa < target_capa:
            print(f"  INFO: Tier3 hospital capacity is {current_capa}/{target_capa}. Adding more for capacity.")
            candidates = df_sorted[(df_sorted["is_tier3"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if candidates.empty: print("  WARNING: No more Tier3 hospitals available to add."); break
            hospital_to_add = candidates.head(1)
            df_selected = pd.concat([df_selected, hospital_to_add])
            current_capa = df_selected[df_selected["is_tier3"] == 1]["eff"].sum()

        # Rule 3: Guarantee at least 1 non-Tier3 hospital (Tier 2 etc.) (some of 64 rules fail if only Tier3 hospitals are nearest)
        if len(df_selected[df_selected["is_tier3"] == 0]) == 0:
            print("  INFO: No Tier 2 hospitals in final list. Adding to prevent simulation errors.")
            candidates = df_sorted[(df_sorted["is_tier3"] == 0) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                df_selected = pd.concat([df_selected, candidates.head(1)])

        # ================================================================= #
        # Rule 4: Guarantee minimum helipad hospitals (>= UAV count)
        if "helipad" in df_selected.columns:
            # Check UAV count (from parameter)
            uav_n = int(max(0, uav_count))

            if uav_n > 0:
                helipad_hospitals = df_selected[df_selected["helipad"] == 1]
                num_helipad = len(helipad_hospitals)

                # Add helipad hospitals if fewer than UAV count
                num_to_ensure_helipad = uav_n - num_helipad

                if num_to_ensure_helipad > 0:
                    print(f"  INFO: {num_helipad} helipad hospitals but {uav_n} UAVs. Adding {num_to_ensure_helipad} to ensure minimum {uav_n} helipad hospitals.")

                    # Find unselected helipad hospitals from the full list
                    candidates_helipad = df_sorted[
                        (df_sorted["helipad"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates_helipad.empty:
                        # Add as many helipad hospitals as needed
                        hospitals_to_add = candidates_helipad.head(num_to_ensure_helipad)
                        df_selected = pd.concat([df_selected, hospitals_to_add])
                        added_names = ", ".join(hospitals_to_add['institution_name'].values)
                        print(f"    -> Added helipad hospitals: {added_names}")
                    else:
                        print(f"  ⚠️ Warning: Only {num_helipad} helipad hospitals exist in the entire dataset. Cannot operate {uav_n} UAVs.")
                else:
                    print(f"  ✓ {num_helipad} helipad hospitals (can operate {uav_n} UAVs)")
            else:
                print("  INFO: UAV count is 0, skipping helipad hospital guarantee logic.")
        else:
            print("  ⚠️ 'helipad' column not found in source data. Skipping helipad guarantee logic.")

        # ================================================================= #
        # Rule 5: Guarantee intersection hospitals for UAV transport (helipad+Tier)
        if "helipad" in df_selected.columns:
            uav_n = int(max(0, uav_count))

            if uav_n > 0:
                # 5-1: Guarantee at least 1 helipad+Tier3 hospital for Red UAV transport
                helipad_tier3_hospitals = df_selected[
                    (df_selected["helipad"] == 1) &
                    (df_selected["is_tier3"] == 1)
                ]

                if len(helipad_tier3_hospitals) == 0:
                    print("  INFO: No helipad+Tier3 hospital for Red UAV transport. Adding...")
                    candidates = df_sorted[
                        (df_sorted["helipad"] == 1) &
                        (df_sorted["is_tier3"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['institution_name'].values[0]
                        print(f"    -> Added: {added_name}")
                    else:
                        print("  ⚠️ Warning: No helipad+Tier3 hospital in entire dataset. Red UAV transport impossible!")
                else:
                    print(f"  ✓ {len(helipad_tier3_hospitals)} helipad+Tier3 hospitals (Red UAV transport available)")

                # 5-2: Guarantee at least 1 helipad+Tier2 hospital for Yellow UAV transport
                helipad_tier2_hospitals = df_selected[
                    (df_selected["helipad"] == 1) &
                    (df_selected["is_tier3"] == 0)
                ]

                if len(helipad_tier2_hospitals) == 0:
                    print("  INFO: No helipad+Tier2 hospital for Yellow UAV transport. Adding...")
                    candidates = df_sorted[
                        (df_sorted["helipad"] == 1) &
                        (df_sorted["is_tier3"] == 0) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['institution_name'].values[0]
                        print(f"    -> Added: {added_name}")
                    else:
                        print("  ⚠️ Warning: No helipad+Tier2 hospital in entire dataset. Yellow UAV transport impossible!")
                else:
                    print(f"  ✓ {len(helipad_tier2_hospitals)} helipad+Tier2 hospitals (Yellow UAV transport available)")
            else:
                print("  INFO: UAV count is 0, skipping helipad+Tier intersection guarantee logic.")
        else:
            print("  ⚠️ 'helipad' column not found in source data. Skipping helipad+Tier intersection guarantee logic.")

        df_euc = df_selected.sort_values("euclidean_distance").reset_index(drop=True).copy()
        print(f" Final hospitals generated: {len(df_euc)} (Tier3: {df_euc['is_tier3'].sum()}, General etc.: {len(df_euc) - df_euc['is_tier3'].sum()})")

        # ---------- (6) EUC file saved later in road order (index consistency) ----------
        # CRITICAL: distance_Hos2Site_euc.csv must follow road order to match h_states indices
        # Therefore, only euc_info is saved at this point; distance is saved after road reordering.

        euc_info = df_euc[["operating_rooms", "capa", "type_code", "institution_name", "helipad"]].copy()
        euc_info.columns = ["num_or", "num_beds", "type_code", "institution_name", "helipad"]
        euc_info_path = os.path.join(save_folder, "hospital_info_euc.csv")
        euc_info.to_csv(euc_info_path, index=True, index_label="Index", encoding="utf-8-sig")
        
        routes_dir_hos = os.path.join(save_folder, "routes", "hos2site")
        ensure_dir(routes_dir_hos)
        # ---------- (7) ROAD distance & time calculation & save (selected hospitals only) ----------
        road_distances = []
        road_durations = []

        for j, (_, row) in enumerate(df_euc.iterrows()):
            end = (row["y_coord"], row["x_coord"])
            road_km, duration_min = self.get_road_distance(
                start=(latitude, longitude), end=end,  # site → hospital
                save_json_dir=routes_dir_hos, route_type="hos2site",
                source_index=j, name=row.get("institution_name", f"hospital_{j}"),
                start_label="site", goal_label="hospital"
            )
            road_distances.append(road_km)
            road_durations.append(duration_min)
            time.sleep(0.05)

        df_euc = df_euc.copy()
        df_euc["road_distance"] = road_distances
        df_euc["road_duration"] = road_durations
        df_road = df_euc.sort_values("road_duration").reset_index(drop=True).copy()

        # Add duration column to distance_Hos2Site_road.csv
        dist_road_df = pd.DataFrame({
            "distance": df_road["road_distance"],
            "duration": df_road["road_duration"]
        })
        dist_road_path = os.path.join(save_folder, "distance_Hos2Site_road.csv")
        dist_road_df.to_csv(dist_road_path, index=True, index_label="Index", encoding="utf-8-sig")

        # CRITICAL FIX: Save distance_Hos2Site_euc.csv in road order (index consistency)
        # df_road is sorted by road_duration, so it has the same index order as h_states.
        # Euclidean distance values are preserved, only the order changes to match road order.
        dist_euc_df = pd.DataFrame({"distance": df_road["euclidean_distance"]})
        dist_euc_path = os.path.join(save_folder, "distance_Hos2Site_euc.csv")
        dist_euc_df.to_csv(dist_euc_path, index=True, index_label="Index", encoding="utf-8-sig")

        road_info = df_road[["operating_rooms", "capa", "type_code", "institution_name", "helipad"]].copy()
        road_info.columns = ["num_or", "num_beds", "type_code", "institution_name", "helipad"]
        road_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        road_info.to_csv(road_info_path, index=True, index_label="Index", encoding="utf-8-sig")

        print(f"  ✅ Hospital info generation complete (distance_Hos2Site_euc.csv saved in road order)")


    
    def make_uav_info(self, latitude, longitude, incident_size, uav_count, save_folder):
        """UAV info generation - based on hospital_info_road.csv (KEY CHANGE)
        - Filter hospitals with helipad=1 from hospital_info_road.csv
        - Calculate distance from incident site and select nearest N helipad hospitals
        - Assign max 1 UAV per hospital
        - uav_info hospitals = guaranteed subset of hospital_info (index consistent)
        - CSV structure: Index, init_distance, num_or, num_beds, type_code, institution_name
        """
        print(f"  🚁 Generating UAV info (based on hospital_info_road.csv)...")

        import os
        import pandas as pd
        from haversine import haversine

        # 0) Parameter cleanup
        try:
            uav_n = int(max(0, int(uav_count)))
        except Exception:
            uav_n = 0
        if uav_n <= 0:
            print("⚠️ UAV count is 0. Skipping UAV info generation.")
            # Instead of skipping, create an empty CSV file for UAV=0 experiments (header only)
            empty_df = pd.DataFrame(columns=["Index", "init_distance", "num_or", "num_beds", "type_code", "institution_name"])
            save_path = os.path.join(save_folder, "uav_info.csv")
            empty_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"  Empty UAV info file created: {save_path}")
            return

        # 1) Load hospital_info_road.csv (instead of original Excel!)
        hospital_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        if not os.path.exists(hospital_info_path):
            print(f"❌ {hospital_info_path} file not found.")
            print("   Please run make_hospital_info() first.")
            raise FileNotFoundError(f"❌ {hospital_info_path} file not found.")

        try:
            df_hospital_pool = pd.read_csv(hospital_info_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"❌ Failed to load hospital_info_road.csv: {e}")
            return

        # 2) Check for helipad column (required)
        if "helipad" not in df_hospital_pool.columns:
            print("❌ 'helipad' column not found in hospital_info_road.csv.")
            print("   Please check the helipad column addition logic in make_hospital_info().")
            raise KeyError("❌ 'helipad' column missing from hospital_info_road.csv.")

        # 3) Filter only helipad hospitals from hospital_info
        df_helipad_in_pool = df_hospital_pool[df_hospital_pool["helipad"] == 1].copy()

        if df_helipad_in_pool.empty:
            print("❌ No helipad hospitals found in hospital_info_road.csv.")
            print("   Please check the helipad guarantee logic in make_hospital_info().")
            raise ValueError("❌ No helipad hospitals in hospital_info.")

        # 4) Validate helipad hospital count (compare with UAV count)
        if len(df_helipad_in_pool) < uav_n:
            print(f"❌ Only {len(df_helipad_in_pool)} helipad hospitals in hospital_info, cannot deploy {uav_n} UAVs.")
            print(f"   Check helipad guarantee logic in make_hospital_info() or reduce UAV count.")
            raise ValueError(
                f"❌ Only {len(df_helipad_in_pool)} helipad hospitals in hospital_info, "
                f"cannot deploy {uav_n} UAVs."
            )

        # 5) Calculate incident-to-hospital Euclidean distance (hospital_info lacks coordinates, so fetch from original Excel)
        # hospital_info_road.csv may already have distance info, but we safely fetch coordinates from the original
        try:
            df_full_excel = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ Failed to load original Excel data: {e}")
            return

        # Match coordinates by hospital name
        df_helipad_in_pool = df_helipad_in_pool.merge(
            df_full_excel[["institution_name", "x_coord", "y_coord"]],
            on="institution_name",
            how="left"
        )

        # Check for hospitals with missing coordinates
        if df_helipad_in_pool[["x_coord", "y_coord"]].isnull().any().any():
            missing_hospitals = df_helipad_in_pool[df_helipad_in_pool[["x_coord", "y_coord"]].isnull().any(axis=1)]["institution_name"].tolist()
            print(f"⚠️ Warning: The following hospitals have no coordinate info: {missing_hospitals}")
            df_helipad_in_pool = df_helipad_in_pool.dropna(subset=["x_coord", "y_coord"])

        df_helipad_in_pool["distance"] = df_helipad_in_pool.apply(
            lambda row: haversine((row["y_coord"], row["x_coord"]), (latitude, longitude)),
            axis=1
        )

        # 6) Sort by distance (nearest helipad hospital first)
        df_helipad_in_pool = df_helipad_in_pool.sort_values("distance").reset_index(drop=True)

        # 7) Select top N (max 1 UAV per hospital)
        df_selected = df_helipad_in_pool.head(uav_n).copy()

        # 8) Save CSV (using same hospitals as hospital_info, index consistency guaranteed)
        result_df = pd.DataFrame({
            "uav_id": range(len(df_selected)),                 # UAV number (0..)
            "hospital_idx": df_selected["Index"].astype(int),
            "init_distance": df_selected["distance"].round(3),
            "num_or": df_selected["num_or"],
            "num_beds": df_selected["num_beds"],
            "type_code": df_selected["type_code"],
            "institution_name": df_selected["institution_name"]
        })

        save_path = os.path.join(save_folder, "uav_info.csv")
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ UAV info generation complete: {len(result_df)} UAVs")
        print(f"     Helipad hospitals: {', '.join(df_selected['institution_name'].head(3).tolist())}{'...' if len(df_selected) > 3 else ''}")
        print(f"     Generated as subset of hospital_info (index consistency guaranteed)")



    def make_patient_info(self, save_folder):
        """Generate patient information (using hardcoded values)"""
        print(f"  👥 Generating patient info...")
        types = self.patient_config["ratio"].keys()
        rows = []
        for t in types:
            α, β = self.patient_config["rescue_param"][t]
            rows.append({
                "type": t,
                "ratio": self.patient_config["ratio"][t],
                "rescue_param_alpha": α,
                "rescue_param_beta": β,
                "treat_tier3": self.patient_config["treat_tier3"][t],
                "treat_tier2": self.patient_config["treat_tier2"][t],
                "treat_tier3_mean": self.patient_config["treat_tier3_mean"][t],
                "treat_tier2_mean": self.patient_config["treat_tier2_mean"][t]
            })
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_folder, "patient_info.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Patient info generation complete")

    def make_distance_Hos2Hos(self, save_folder):
        """Generate inter-hospital distance matrix"""
        print(f"  📐 Generating inter-hospital distance matrix...")
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ Failed to load hospital data: {e}")
            return

        # Euclidean (CRITICAL FIX: generated in road order)
        try:
            # Use hospital_info_road.csv instead of hospital_info_euc.csv (index consistency)
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road_hos = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road_hos["institution_name"].tolist()
            coords_road = []
            for name in names_road:
                row = df_full[df_full["institution_name"] == name]
                if not row.empty:
                    coords_road.append((row.iloc[0]["y_coord"], row.iloc[0]["x_coord"]))
                else:
                    coords_road.append((0, 0))
            N = len(coords_road)
            matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    if i == j:
                        dist = 0
                    else:
                        dist = haversine(coords_road[i], coords_road[j])
                    matrix[i][j] = dist
                    matrix[j][i] = dist
            save_path_euc = os.path.join(save_folder, "distance_Hos2Hos_euc.csv")
            pd.DataFrame(matrix).to_csv(save_path_euc, index=True, encoding="utf-8-sig")
            print(f"  ✅ Inter-hospital Euclidean distance matrix complete (road order)")
        except Exception as e:
            print(f"❌ Euclidean distance calculation failed: {e}")

        # Road (using Excel file - pre-calculated data)
        try:
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road["institution_name"].tolist()

            # Load pre-calculated distance matrix from Excel
            excel_path = os.path.join(self.base_path, "scenarios", "DISTANCE_MATRIX_FINAL.xlsx")
            print(f"  📂 Loading Excel distance matrix: {excel_path}")
            df_matrix = pd.read_excel(excel_path, sheet_name="Distance_Matrix", engine="openpyxl")

            # Use first column as index (hospital names)
            df_matrix_indexed = df_matrix.set_index(df_matrix.columns[0])  # Use first column as index

            # Build distance matrix by looking up values
            N = len(names_road)
            matrix = np.zeros((N, N))
            missing_hospitals = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = 0
                    else:
                        hospital_i = names_road[i]
                        hospital_j = names_road[j]

                        # Look up distance from Excel matrix
                        if hospital_i in df_matrix_indexed.index and hospital_j in df_matrix_indexed.columns:
                            dist = df_matrix_indexed.loc[hospital_i, hospital_j]
                            matrix[i][j] = float(dist) if pd.notna(dist) else 0
                        else:
                            matrix[i][j] = 0
                            if hospital_i not in missing_hospitals:
                                missing_hospitals.append(hospital_i)
                            if hospital_j not in missing_hospitals:
                                missing_hospitals.append(hospital_j)

            if missing_hospitals:
                print(f"  ⚠️ Hospitals not found in Excel ({len(missing_hospitals)}): {missing_hospitals[:5]}...")

            save_path_road = os.path.join(save_folder, "distance_Hos2Hos_road.csv")
            pd.DataFrame(matrix).to_csv(save_path_road, index=True, encoding="utf-8-sig")
            print(f"  ✅ Inter-hospital road distance matrix complete (using Excel data)")
        except Exception as e:
            print(f"❌ Road distance calculation failed: {e}")
        print(f"  ✅ Inter-hospital distance matrix generation complete")

    def _sanitize_coeff_text(self, text: str) -> str:
        """Normalize '1.1,1' or '[1.1, 1]' to '1.1, 1'"""
        if not text:
            return "1,1"
        t = text.strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        parts = [p.strip() for p in t.split(",") if p.strip() != ""]
        if len(parts) != 2:
            return "1,1"
        # Numeric validation (fallback to default on failure)
        try:
            a = float(parts[0]); b = float(parts[1])
        except Exception:
            return "1,1"
        return f"{a},{b}".replace(",", ", ")
    
    def make_config_yaml(self, latitude, longitude, incident_size, amb_velocity,
                         uav_velocity, total_samples, random_seed, save_folder, is_use_time=True,
                         amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0):
        """Generate Config YAML file"""
        print(f"  ⚙️ Generating Config YAML...")
        folder_name = f"({latitude},{longitude})"
        config_filename = f"config_{folder_name}.yaml"
        config_path = os.path.join(save_folder, config_filename)
        relative_folder = f"./scenarios/{self.experiment_id}/{folder_name}"

        # departure_time field
        departure_time_field = ""
        if self.departure_time:
            departure_time_field = f'  departure_time: "{self.departure_time}" # API query time (YYYYMMDDHHMM)\n'

        yaml_content = f"""#incident_info:
#  incident_size: {incident_size} # Incident size (total patients)
#  latitude: {latitude} # Latitude
#  longitude: {longitude} # Longitude
#  incident_type: null # Expandable for future incident type settings

entity_info:
{departure_time_field}  patient:
    incident_size: {incident_size} # Incident size (total patients)
    latitude: {latitude} # Latitude
    longitude: {longitude} # Longitude
    incident_type: null # Expandable for future incident type settings
    info_path: "{relative_folder}/patient_info.csv"
  hospital:
    load_data: True
    info_path: "{relative_folder}/hospital_info_road.csv"
    dist_Hos2Hos_euc_info: "{relative_folder}/distance_Hos2Hos_euc.csv"
    dist_Hos2Hos_road_info: "{relative_folder}/distance_Hos2Hos_road.csv"
    dist_Hos2Site_euc_info: "{relative_folder}/distance_Hos2Site_euc.csv"
    dist_Hos2Site_road_info: "{relative_folder}/distance_Hos2Site_road.csv"
    max_send_coeff: [{self._sanitize_coeff_text(self.max_send_coeff_text)}]
  ambulance:
    load_data: True
    dispatch_distance_info: "{relative_folder}/amb_info_road.csv"
    velocity: {amb_velocity} # unit: km/h
    handover_time: {amb_handover_time} # unit: minutes
    is_use_time: {('True' if is_use_time else 'False')} # True: use API duration, False: distance/velocity based calculation
    duration_coeff: {duration_coeff} # API duration time weight (default: 1.0, adjust for environmental factors)
    road_provider: {self.road_provider or ('kakao' if is_use_time else 'osrm')} # Road data provider (kakao | osrm) - recorded at scenario generation
  uav:
    load_data: True
    dispatch_distance_info: "{relative_folder}/uav_info.csv"
    velocity: {uav_velocity} # unit: km/h
    handover_time: {uav_handover_time} # unit: minutes
    is_use_time: False # UAV always uses Euclidean distance

event_info_path: "./src/sim_src/event_info.json"

rule_info:
  isFullFactorial: True
  priority_rule: ["START", "ReSTART"]
  hos_select_rule: ["RedOnly", "YellowNearest"] # hos_select_rule: ["RedOnly", "YellowHalf"]
  red_mode_rule: ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
  yellow_mode_rule: ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]

run_setting:
  totalSamples: {total_samples} # number of samples
  random_seed: {random_seed} # null, if do not want to fix
  rule_test: True
  eval_mode: True
  output_path: "./results/{self.experiment_id}"
  exp_indicator: "{folder_name}"
  save_info: True # NotImplemented"""
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(yaml_content)
        print(f"  ✅ Config YAML generation complete")
        absolute_config_path = os.path.abspath(config_path)
        print(f"CONFIG_PATH:{absolute_config_path}")
        return absolute_config_path

    def generate_scenario(self, latitude, longitude, incident_size, amb_count,
                          uav_count, amb_velocity, uav_velocity,
                          total_samples, random_seed, is_use_time=True,
                          amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0):
        """
        Generate a complete scenario (all CSV + YAML)
        Args:
            is_use_time: True for API duration, False for distance/velocity based calculation
            amb_handover_time: Ambulance patient handover time (minutes)
            uav_handover_time: UAV patient handover time (minutes)
        Returns: Path to generated config file
        """
        # Defensive bool conversion (works even if caller passes a string)
        is_use_time = bool(is_use_time) if not isinstance(is_use_time, str) else str2bool(is_use_time)

        # road_provider was already determined in __init__ based on is_use_time.
        # If the is_use_time at call time differs from __init__ assumption, update and warn.
        expected_provider = "kakao" if is_use_time else "osrm"
        if expected_provider != self.road_provider:
            print(f"  ⚠️ road_provider mismatch: __init__={self.road_provider}, "
                  f"generate_scenario(is_use_time={is_use_time}) -> updating to {expected_provider}")
            self.road_provider = expected_provider

        if self.road_provider == "kakao" and not self.kakao_api_key:
            raise RuntimeError(
                "is_use_time=True mode requires --kakao_api_key. "
                "If you don't have a key, use is_use_time=False to use the OSRM backend."
            )
        print(f"  🛣️ Road data provider: {self.road_provider}"
              f"{' (' + self.osrm_url + ')' if self.road_provider == 'osrm' else ''}")

        print(f"""\n📍 Starting scenario generation for coordinates ({latitude},{longitude})...""")
        start_time = time.time()
        folder_name = f"({latitude},{longitude})"
        save_folder = os.path.join(self.base_path, "scenarios", self.experiment_id, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        # Clean up previous run's routes/ folder (prevent JSON accumulation on retry)
        import shutil as _shutil
        routes_cleanup = os.path.join(save_folder, "routes")
        if os.path.exists(routes_cleanup):
            _shutil.rmtree(routes_cleanup)

        # Reverse geocoding is done in orchestrator.py, so only basic info is printed
        coordinate_info = {
            "latitude": latitude,
            "longitude": longitude,
            "full_address": "",
            "road_address": "",
            "area1": "",
            "area2": "",
            "area3": "",
            "area4": "",
            "is_valid": False
        }
        print(f"COORDINATE_INFO:{json.dumps(coordinate_info, ensure_ascii=False)}")
        print(f"  📍 Coordinates: ({latitude}, {longitude}) - reverse geocoding handled by orchestrator")

        # Generation pipeline
        self.make_amb_info(latitude, longitude, incident_size, amb_count, save_folder)
        self.make_hospital_info(latitude, longitude, incident_size, save_folder, uav_count)
        self.make_uav_info(latitude, longitude, incident_size, uav_count, save_folder)
        self.make_patient_info(save_folder)
        self.make_distance_Hos2Hos(save_folder)
        config_path = self.make_config_yaml(
            latitude, longitude, incident_size,
            amb_velocity, uav_velocity, total_samples,
            random_seed, save_folder, is_use_time,
            amb_handover_time, uav_handover_time, duration_coeff
        )
        
        elapsed = round(time.time() - start_time, 2)
        print(f"  ⏱️ Scenario generation complete ({elapsed}s)")
        print(f"API_CALL_COUNT:{self.api_call_count}")
        print(f"CONFIG_PATH:{config_path}")
        return config_path

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCI scenario dynamic generation (cross-platform compatible)")
    parser.add_argument("--base_path", required=True, help="Project root path")
    parser.add_argument("--latitude", type=float, required=False, help="Latitude")
    parser.add_argument("--longitude", type=float, required=False, help="Longitude")
    parser.add_argument("--incident_size", type=int, default=30, help="Number of patients")
    parser.add_argument("--amb_count", type=int, default=30, help="Number of ambulances")
    parser.add_argument("--uav_count", type=int, default=3, help="Number of UAVs")
    parser.add_argument("--amb_velocity", type=int, default=40, help="Ambulance speed")
    parser.add_argument("--uav_velocity", type=int, default=80, help="UAV speed")
    parser.add_argument("--total_samples", type=int, default=30, help="Number of simulation iterations")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--experiment_id", type=str, default=None, help="Experiment ID")
    # Advanced options (both ENV and CLI allowed)
    # parser.add_argument("--queue_policy", type=str, help='e.g. "0", "capa/2", "0.5"')
    parser.add_argument("--buffer_ratio", type=float, help="Candidate buffer multiplier (default 1.5)")
    parser.add_argument("--util_by_tier", type=str, help='e.g. "1:0.90,11:0.75,etc:0.60"')
    parser.add_argument("--hospital_max_send_coeff", type=str, default=None, help="Send coefficient 'a,b' format (e.g. 1.1,1.0). If not specified, uses ENV(MCI_MAX_SEND_COEFF) or default 1,1")

    # Kakao API parameters
    parser.add_argument("--kakao_api_key", type=str, default=None, help="Kakao Mobility REST API key (required for is_use_time=true mode)")
    parser.add_argument("--departure_time", type=str, default=None, help="Departure time (YYYYMMDDHHMM format, e.g. 202512241800)")
    parser.add_argument("--is_use_time", type=str2bool, default=True,
                        help="True: Kakao API duration based / False: OSRM static distance based (distance/velocity). OSRM duration can also be used on simulation re-runs")
    # OSRM backend (used when is_use_time=false)
    parser.add_argument("--osrm_url", type=str, default=None,
                        help="OSRM HTTP API base URL (default: env MCI_OSRM_URL or https://router.project-osrm.org)")
    parser.add_argument("--amb_handover_time", type=float, default=10.0, help="Ambulance patient handover time (minutes)")
    parser.add_argument("--uav_handover_time", type=float, default=15.0, help="UAV patient handover time (minutes)")
    parser.add_argument("--duration_coeff", type=float, default=1.0, help="API duration time weight (default: 1.0)")

    args = parser.parse_args()
    try:
        # UTF-8 output configuration
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    try:
        # args.is_use_time is already parsed to bool by str2bool
        generator = ScenarioGenerator(
            args.base_path,
            args.experiment_id,
            kakao_api_key=args.kakao_api_key,
            departure_time=args.departure_time,
            osrm_url=args.osrm_url,
            is_use_time=args.is_use_time,
        )

        # CLI arguments override ENV defaults
        if args.hospital_max_send_coeff:
            generator.max_send_coeff_text = args.hospital_max_send_coeff
        # if args.queue_policy is not None:
        #     generator.queue_policy = args.queue_policy
        if args.buffer_ratio is not None:
            generator.buffer_ratio = float(args.buffer_ratio)
        if args.util_by_tier:
            m = parse_util_map(args.util_by_tier)
            if m:
                generator.util_by_tier = m

        # Print current applied values
        print(f"buffer_ratio={generator.buffer_ratio}")

        # In current project flow, incident coordinates are passed directly from outside.
        if args.latitude is None or args.longitude is None:
            print("❌ --latitude and --longitude arguments are required.")
            sys.exit(1)
        latitude, longitude = args.latitude, args.longitude
        
        # Generate scenario
        config_path = generator.generate_scenario(
            latitude, longitude,
            args.incident_size, args.amb_count, args.uav_count,
            args.amb_velocity, args.uav_velocity,
            args.total_samples, args.random_seed,
            is_use_time=args.is_use_time,
            amb_handover_time=args.amb_handover_time,
            uav_handover_time=args.uav_handover_time,
            duration_coeff=args.duration_coeff
        )
        
        if config_path:
            print(f"\n✅ Scenario generation successful!")
            print(f"📄 Config file: {config_path}")
        else:
            print("❌ Scenario generation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
