# Generate.py
# Standalone page for new scenario generation
# -------------------------------------------------------------------------------------------------
import os, re, yaml, shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd
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

KST = timezone(timedelta(hours=9))

# Page config
st.set_page_config(
    page_title="Create New Scenario",
    page_icon="➕",
    layout="wide"
)

# ── Command-Center Theme CSS ──────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #d4d4d8;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #e2a04a 0%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #141417;
}
[data-testid="stSidebar"] {
    background-color: #1a1a1f;
}

/* ── Buttons ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #e2a04a 0%, #c7893e 100%);
    color: #141417;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    opacity: .85;
    color: #141417;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #e2a04a 0%, #c7893e 100%) !important;
}

/* ── Inputs ── */
input, textarea, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #1e1e23 !important;
    color: #d4d4d8 !important;
    border: 1px solid #2a2a30 !important;
    border-radius: 6px !important;
}
input:focus, textarea:focus,
[data-baseweb="input"] input:focus {
    border-color: #e2a04a !important;
    box-shadow: 0 0 0 1px #e2a04a33 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #d4d4d8;
    border-radius: 6px 6px 0 0;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background-color: #1e1e23;
    border-bottom: 2px solid #e2a04a;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e1e23;
    border: 1px solid #2a2a30;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
    color: #a1a1aa;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stMetricValue"] {
    color: #e2a04a;
    font-family: 'Outfit', sans-serif;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #141417; }
::-webkit-scrollbar-thumb { background: #2a2a30; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a40; }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────
def base_ok(bp: str) -> bool:
    """Check if base_path is valid (scenarios folder exists)"""
    if not bp:
        return False
    scenarios = os.path.join(bp, "scenarios")
    return os.path.isdir(scenarios)

def parse_env_kv(text: str):
    """Parse environment variable text"""
    env = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

def get_kakao_key_from_secrets():
    """Read Kakao key from Streamlit Cloud Secrets (TOML) / environment variables"""
    # 1) Streamlit Secrets (recommended)
    try:
        # Highest priority: TOP-LEVEL key
        for k in ("KAKAO_REST_API_KEY", "KAKAO_API_KEY", "KAKAO_KEY"):
            if k in st.secrets:
                return str(st.secrets[k]).strip()

        # Section format also supported: [kakao] rest_api_key="..."
        if "kakao" in st.secrets:
            sec = st.secrets["kakao"]
            for k in ("rest_api_key", "api_key", "key"):
                if k in sec:
                    return str(sec[k]).strip()
    except Exception:
        pass

    # 2) Environment variable fallback (available if needed)
    return (os.getenv("KAKAO_REST_API_KEY") or os.getenv("KAKAO_API_KEY") or "").strip()

def normalize_search_result(doc, search_type):
    """Normalize API response to unified format for both keyword and address searches"""
    if search_type == "Keyword Search":
        return {
            "place_name": doc.get("place_name", ""),
            "address_name": doc.get("address_name", ""),
            "x": float(doc["x"]),
            "y": float(doc["y"]),
            "search_type": "keyword"
        }
    else:  # Address search
        # Prefer building name for display
        building_name = ""
        if "road_address" in doc and doc["road_address"]:
            building_name = doc["road_address"].get("building_name", "")

        display_name = building_name if building_name else doc.get("address_name", "")

        return {
            "place_name": f"{display_name} (Address Search)",
            "address_name": doc.get("address_name", ""),
            "x": float(doc["x"]),
            "y": float(doc["y"]),
            "search_type": "address"
        }


def perform_address_search(search_query, api_key):
    """
    Perform address search using Kakao Local API
    API Doc: https://developers.kakao.com/docs/latest/ko/local/dev-guide#address-coord

    Returns: (success: bool, documents: list, error_msg: str, status_code: int)
    """
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {api_key.strip()}"}

        params = {
            "query": search_query,
            "analyze_type": "similar",  # Allow partial matches
            "size": 10
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            with st.expander("Debug Info", expanded=False):
                st.caption(f"API URL = {url}")
                st.caption(f"query = {search_query}")
                st.caption(f"status = {response.status_code}")
                st.caption(f"results = {len(documents)}")

            # Normalize results to match keyword search format
            normalized = [normalize_search_result(doc, "Address Search") for doc in documents]
            return (True, normalized, "", 200)

        elif response.status_code == 401:
            return (False, [], "API key auth failed (401 Unauthorized)", 401)

        elif response.status_code == 403:
            return (False, [], "Access denied (403 Forbidden)", 403)

        else:
            try:
                error_data = response.json()
                error_msg = str(error_data)
            except:
                error_msg = response.text
            return (False, [], f"API error (status: {response.status_code})", response.status_code)

    except requests.exceptions.Timeout:
        return (False, [], "Request timeout. Check network connection.", -1)

    except requests.exceptions.RequestException as e:
        error_msg = f"Search failed: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" (status: {e.response.status_code})"
        return (False, [], error_msg, -1)

    except Exception as e:
        return (False, [], f"Unexpected error: {e}", -1)

# ─────────────────────────────────────────────────────────────────
# Session State initialization
# ─────────────────────────────────────────────────────────────────
CLOUD_BASE_PATH = _detect_cloud_base_path()
IS_CLOUD = bool(CLOUD_BASE_PATH)
DEFAULT_LOCAL_BASE_PATH = str(REPO_ROOT) if (REPO_ROOT / "scenarios").is_dir() else ""

if "generate_base_path" not in st.session_state:
    st.session_state.generate_base_path = CLOUD_BASE_PATH if IS_CLOUD else DEFAULT_LOCAL_BASE_PATH
else:
    # On Cloud, always fixed (immediately reverted even if user changes it)
    if IS_CLOUD and st.session_state.generate_base_path != CLOUD_BASE_PATH:
        st.session_state.generate_base_path = CLOUD_BASE_PATH

if "gen_state" not in st.session_state:
    st.session_state.gen_state = {}
if "env_txt" not in st.session_state:
    st.session_state.env_txt = ""
# Batch generation state
if "batch_coord_rows" not in st.session_state:
    st.session_state.batch_coord_rows = []
if "batch_preset_rows" not in st.session_state:
    st.session_state.batch_preset_rows = []
if "batch_run_log" not in st.session_state:
    st.session_state.batch_run_log = []
# Preset list (default set used when adding rows)
if "batch_presets" not in st.session_state:
    st.session_state.batch_presets = [{
        "name": "Default",
        "incident_size": 30,
        "amb_count": 30,
        "uav_count": 3,
        "amb_velocity": 40,
        "uav_velocity": 80,
        "amb_handover": 10.0,
        "uav_handover": 15.0,
        "total_samples": 30,
        "random_seed": 0,
        "buffer_ratio": 1.5,
        "max_send_coeff": "1,1",
        "is_use_time": True,
        "duration_coeff": 1.0,
    }]

# Helper: preset lookup / coordinate row append
def _get_preset_by_name(name: str):
    for p in st.session_state.batch_presets:
        if str(p.get("name")) == str(name):
            return p
    return st.session_state.batch_presets[0] if st.session_state.batch_presets else {}

def _append_coord_row(label: str, lat: float, lon: float, address: str = "", preset_name: str = "Default", search_source: str = "manual"):
    p = _get_preset_by_name(preset_name)
    row = {
        "label": label or address or f"{lat},{lon}",
        "lat": lat,
        "lon": lon,
        "address": address,
        "incident_size": p.get("incident_size", 30),
        "amb_count": p.get("amb_count", 30),
        "uav_count": p.get("uav_count", 3),
        "amb_velocity": p.get("amb_velocity", 40),
        "uav_velocity": p.get("uav_velocity", 80),
        "amb_handover": p.get("amb_handover", 10.0),
        "uav_handover": p.get("uav_handover", 15.0),
        "total_samples": p.get("total_samples", 30),
        "random_seed": p.get("random_seed", 0),
        "buffer_ratio": p.get("buffer_ratio", 1.5),
        "max_send_coeff": p.get("max_send_coeff", "1,1"),
        "is_use_time": p.get("is_use_time", True),
        "duration_coeff": p.get("duration_coeff", 1.0),
        "preset": preset_name,
        "source": search_source,
    }
    st.session_state.batch_coord_rows.append(row)


def _write_label_map(base_path: str, rows: list[dict]):
    """
    rows: [{exp_id, coord, label, source}]
    Cumulatively saved to scenarios/label_map.csv (latest wins on exp_id+coord duplicates)
    """
    if not rows:
        return
    path = Path(base_path) / "scenarios" / "label_map.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["exp_id", "coord", "label", "source"]
    df_new = pd.DataFrame(rows, columns=cols)
    if path.exists():
        try:
            df_old = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df_old = pd.DataFrame(columns=cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["exp_id", "coord"], keep="last")
    else:
        df = df_new
    df.to_csv(path, index=False, encoding="utf-8-sig")

# ─────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────
st.title("🧪 Scenario Generation & Execution")
st.info("💡 This page operates **completely independently** from the main app's sidebar settings. You can generate new scenarios even without existing ones!")

# ─────────────────────────────────────────────────────────────────
# 1. Project path (base_path) input
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Project Path Setup")

col_path, col_btn = st.columns([4, 1])
with col_path:
    generate_bp_input = st.text_input(
        "🗂️ Project Path (base_path)",
        value=st.session_state.generate_base_path,
        placeholder="e.g. C:\\Users\\USER\\MCI_ADV",
        help="Enter the project root path containing the scenarios folder",
        disabled=IS_CLOUD,
    )
    if IS_CLOUD:
        st.caption(f"☁️ Cloud: base_path is fixed to `{CLOUD_BASE_PATH}`.")

with col_btn:
    st.write("")  # alignment spacer
    st.write("")  # alignment spacer
    if (not IS_CLOUD) and st.button("✅ Confirm Path", key="gen_check_path"):
        st.session_state.generate_base_path = generate_bp_input


bp = st.session_state.generate_base_path

# Path validation
if not bp:
    st.warning("⚠️ Please enter the project path above and click **✅ Confirm Path**.")
    st.stop()

if not base_ok(bp):
    st.error(f"❌ Invalid path: `{bp}`")
    st.caption("• Check that the path exists\n• Check that the `scenarios` folder is present")
    st.stop()

st.success(f"✅ Valid path: `{bp}`")

# ─────────────────────────────────────────────────────────────────
# 2. Load Orchestrator
# ─────────────────────────────────────────────────────────────────
try:
    from orchestrator import Orchestrator
except Exception as e:
    st.error("❌ `src/sce_src/orchestrator.py` not found.")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# 3. Scenario generation UI
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 1️⃣ Scenario Generation")

# API key input and save
st.markdown("#### 🔑 Kakao REST API Key")

# session_state initialization
if "kakao_api_key" not in st.session_state:
    if IS_CLOUD:
        st.session_state.kakao_api_key = get_kakao_key_from_secrets() or ""
    else:
        st.session_state.kakao_api_key = ""
else:
    # On Cloud, always sync to the Secrets value if available
    if IS_CLOUD:
        sec = get_kakao_key_from_secrets()
        if sec and st.session_state.kakao_api_key != sec:
            st.session_state.kakao_api_key = sec


col_key, col_save = st.columns([3, 1])

cloud_secret_key = get_kakao_key_from_secrets() if IS_CLOUD else ""
has_cloud_key = bool(cloud_secret_key)

with col_key:
    api_key_input = st.text_input(
        "Kakao REST API Key",
        value=st.session_state.kakao_api_key,
        type="password",
        placeholder="MCI app REST API key (Mobility + Local service activation required)",
        help="Used for scenario generation and coordinate search",
        key="api_key_input",
        disabled=IS_CLOUD and has_cloud_key,   # Lock input when Cloud+Secrets
    )
    if IS_CLOUD and has_cloud_key:
        st.caption("☁️ API key automatically loaded from Cloud Secrets.")


with col_save:
    st.write("")  # alignment spacer
    st.write("")  # alignment spacer

    # Local: use "Save" as before
    # Cloud: allow manual input only when Secrets is absent (edge case)
    if ((not IS_CLOUD) or (IS_CLOUD and not has_cloud_key)) and st.button("✅ Save", key="save_api_key"):
        if api_key_input and api_key_input.strip():
            st.session_state.kakao_api_key = api_key_input.strip()
            st.success("✅ API key saved!")
        else:
            st.error("⚠️ Please enter an API key.")


# Status display
if st.session_state.kakao_api_key:
    st.caption(f"✅ API key stored ({len(st.session_state.kakao_api_key)} chars)")
else:
    st.caption("⚠️ No API key")

# Use the same key for all purposes
kakao_api_key = st.session_state.kakao_api_key

# ─────────────────────────────────────────────────────────────
# Road data provider selection (Kakao API ↔ OSRM)
# ─────────────────────────────────────────────────────────────
st.markdown("#### 🛣️ Road Data Provider")
is_use_time = st.checkbox(
    "Use Kakao Mobility API duration (real-time traffic)",
    value=True,
    key="is_use_time_checkbox",
    help=(
        "✅ Checked → Calls the Kakao Mobility API. Real-time/predicted duration (minutes) "
        "for the given departure time is saved to CSV and used by the simulator as "
        "'duration × duration_coeff'. **Requires a Kakao REST API key.**\n\n"
        "⬜ Unchecked → Calls the open-source OSRM service "
        "(https://router.project-osrm.org). Both road distance (km) and duration (min) "
        "are saved with the same schema, but the **first simulation runs in "
        "distance/velocity mode** (ScenarioManager branch). No API key needed — "
        "recommended for external reviewers and public code environments. "
        "If you later re-run the same scenario folder with is_use_time=True, "
        "the stored OSRM duration will be used."
    ),
)
if not is_use_time:
    st.caption("ℹ️ OSRM mode: no Kakao key required. The OSRM URL comes from the "
               "`MCI_OSRM_URL` environment variable, or the default demo server.")

# Travel time mode selection
st.markdown("#### 📍 Departure Time Setup")

# session_state initialization (once only)
if "departure_date_value" not in st.session_state:
    st.session_state.departure_date_value = datetime.now(KST).date()
if "departure_time_value" not in st.session_state:
    st.session_state.departure_time_value = datetime.now(KST).time()

col_date, col_time = st.columns(2)
with col_date:
    departure_date = st.date_input(
        "Departure Date",
        value=st.session_state.departure_date_value,
        help="Expected incident date",
        key="departure_date_input"
    )
with col_time:
    departure_time = st.time_input(
        "Departure Time",
        value=st.session_state.departure_time_value,
        help="Expected incident time",
        key="departure_time_input"
    )

# Save to session_state immediately when value changes (updates on single click)
if departure_date != st.session_state.departure_date_value:
    st.session_state.departure_date_value = departure_date
if departure_time != st.session_state.departure_time_value:
    st.session_state.departure_time_value = departure_time

# Convert to YYYYMMDDHHMM format
departure_time_str = f"{st.session_state.departure_date_value.strftime('%Y%m%d')}{st.session_state.departure_time_value.strftime('%H%M')}"
st.caption(f"→ API param: `{departure_time_str}`")

# duration_coeff (is_use_time has moved to the 'Road Data Provider' section above)
duration_coeff = st.number_input(
    "API Duration Weight",
    value=1.0,
    min_value=0.1,
    max_value=10.0,
    step=0.1,
    format="%.1f",
    help="Coefficient multiplied with the API duration (default 1.0; adjust for weather/traffic). Only meaningful when is_use_time=True."
)

st.markdown("---")
st.markdown("### Coordinate Search")

# Search type selector
search_type = st.radio(
    "Search Method",
    ["Keyword Search", "Address Search"],
    horizontal=True,
    key="search_type_radio",
    help="Keyword: search by name (e.g. Incheon Airport) | Address: road/lot number"
)

# session_state initialization
if "search_type" not in st.session_state:
    st.session_state.search_type = "Keyword Search"

# Clear results when switching search types
if st.session_state.search_type != search_type:
    st.session_state.search_type = search_type
    st.session_state.search_results = []
    st.session_state.selected_place_index = -1

if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 37.465833
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = 126.443333
if "selected_place_name" not in st.session_state:
    st.session_state.selected_place_name = ""
if "selected_place_index" not in st.session_state:
    st.session_state.selected_place_index = -1  # -1 means nothing selected

# Dynamic placeholder and help text based on search type
if search_type == "Keyword Search":
    placeholder = "e.g. Incheon Airport, Seoul Station"
    help_text = "Search places via Kakao API"
    label = "Place Search"
else:
    placeholder = "e.g. Seoul Gangnam-gu Teheran-ro 152"
    help_text = "Search address via Kakao API (road/lot number)"
    label = "Address Search"

with st.form(key="search_form"):
    col_search, col_search_btn = st.columns([3, 1])
    with col_search:
        search_keyword = st.text_input(
            label,
            placeholder=placeholder,
            help=help_text,
        )
    with col_search_btn:
        st.write("")
        st.write("")
        search_button = st.form_submit_button("Search")

# Execute search
if search_button and search_keyword:
    # Check REST API key
    if not kakao_api_key or not kakao_api_key.strip():
        st.error("⚠️ Please enter the Kakao REST API key and click '✅ Save' first!")
    else:
        # Route to appropriate search based on selected type
        if search_type == "Keyword Search":
            # ─────────────────────────────────────────────────────────────────
            # Keyword search
            # ─────────────────────────────────────────────────────────────────
            try:
                # Kakao Local API - Keyword search
                # Official docs: https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-keyword
                url = "https://dapi.kakao.com/v2/local/search/keyword.json"

                # Header: Authorization: KakaoAK {REST_API_KEY}
                headers = {
                    "Authorization": f"KakaoAK {kakao_api_key.strip()}"
                }

                params = {
                    "query": search_keyword,
                    "size": 10  # Max 10 results
                }

                # API request
                response = requests.get(url, headers=headers, params=params, timeout=10)

                # Check response
                if response.status_code == 200:
                    data = response.json()
                    documents = data.get("documents", [])

                    with st.expander("Debug Info", expanded=False):
                        st.caption(f"API URL = {url}")
                        st.caption(f"header = Authorization: KakaoAK {kakao_api_key[:4]}...{kakao_api_key[-4:]}")
                        st.caption(f"query = {search_keyword}")
                        st.caption(f"status = {response.status_code}")
                        st.caption(f"response keys = {list(data.keys())}")
                        st.caption(f"results = {len(documents)}")

                    if documents:
                        # Normalize keyword results
                        normalized = [normalize_search_result(doc, "Keyword Search") for doc in documents]
                        st.session_state.search_results = normalized
                        st.success(f"✅ {len(documents)} places found!")
                    else:
                        st.warning("⚠️ No search results.")
                        st.session_state.search_results = []
                elif response.status_code == 401:
                    st.error("❌ API key auth failed (401 Unauthorized)")
                    st.caption("Check if your REST API key is correct.")
                    try:
                        error_data = response.json()
                        st.code(error_data, language="json")
                    except:
                        st.code(response.text)
                elif response.status_code == 403:
                    st.error("❌ Access denied (403 Forbidden)")
                    st.caption("Check platform settings and API key permissions.")
                    try:
                        error_data = response.json()
                        st.code(error_data, language="json")
                    except:
                        st.code(response.text)
                else:
                    st.error(f"❌ API error (status: {response.status_code})")
                    try:
                        error_data = response.json()
                        st.code(error_data, language="json")
                    except:
                        st.code(response.text)

            except requests.exceptions.Timeout:
                st.error("❌ Request timeout. Check network connection.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Search failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.caption(f"Status code: {e.response.status_code}")
                    try:
                        st.code(e.response.json(), language="json")
                    except:
                        st.code(e.response.text)
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                import traceback
                st.code(traceback.format_exc())

        else:  # Address search
            # ─────────────────────────────────────────────────────────────────
            # Address search
            # ─────────────────────────────────────────────────────────────────
            success, documents, error_msg, status_code = perform_address_search(search_keyword, kakao_api_key)

            if success:
                if documents:
                    st.session_state.search_results = documents
                    st.success(f"✅ {len(documents)} addresses found!")
                else:
                    st.warning("⚠️ No results. Please check the address.")
                    st.info("""💡 **Address Search Tips:**
- Road name: `Seoul Gangnam-gu Teheran-ro 152`
- Lot number: `Seoul Gangnam-gu Yeoksam-dong 737`
- Abbreviated: `Gangnam-gu Teheran-ro 152`
                    """)
                    st.session_state.search_results = []
            else:
                # Display error based on status code
                if status_code == 401:
                    st.error(f"❌ {error_msg}")
                    st.caption("Check if your REST API key is correct.")
                elif status_code == 403:
                    st.error(f"❌ {error_msg}")
                    st.caption("Check platform settings and API key permissions.")
                else:
                    st.error(f"❌ {error_msg}")

# ─────────────────────────────────────────────────────────────────
# Display search results
# ─────────────────────────────────────────────────────────────────
if st.session_state.search_results:
    st.markdown("#### Search Results")

    # Create map (center: first result)
    first = st.session_state.search_results[0]
    center_lat = float(first["y"])
    center_lon = float(first["x"])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Add markers
    for idx, place in enumerate(st.session_state.search_results):
        place_name = place.get("place_name", "")
        address = place.get("address_name", "")
        lat = float(place["y"])
        lon = float(place["x"])

        # Popup content
        popup_html = f"""
        <div style="width: 200px;">
            <b>{place_name}</b><br>
            {address}<br>
            <small>Lat: {lat:.6f}</small><br>
            <small>Lon: {lon:.6f}</small>
        </div>
        """

        # Selected place is red, others are blue
        is_selected = (st.session_state.selected_place_index == idx)
        marker_color = "red" if is_selected else "blue"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=place_name,
            icon=folium.Icon(color=marker_color, icon="info-sign")
        ).add_to(m)

    # Display map
    st_folium(m, width=700, height=400, returned_objects=[])

    # Results table
    preset_options = [p.get("name","") for p in st.session_state.batch_presets]
    add_preset_choice = st.selectbox("Preset to apply", options=preset_options, key="search_add_preset")
    st.markdown("**Place List (click to select / add to list)**")
    for idx, place in enumerate(st.session_state.search_results):
        place_name = place.get("place_name", "")
        address = place.get("address_name", "")
        lat = float(place["y"])
        lon = float(place["x"])

        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{idx+1}. {place_name}**")
            st.caption(f"📍 {address}")
            st.caption(f"Coords: ({lat:.6f}, {lon:.6f})")
        with col2:
            if st.button("Select", key=f"select_{idx}"):
                st.session_state.selected_lat = lat
                st.session_state.selected_lon = lon
                st.session_state.selected_place_name = place_name
                st.session_state.selected_place_index = idx  # Store selected index
                st.success(f"✅ '{place_name}' selected!")
                st.rerun()
            if st.button("Add to List", key=f"addlist_{idx}"):
                disp_label = place_name or address or f"{lat:.5f},{lon:.5f}"
                _append_coord_row(disp_label, lat, lon, address, add_preset_choice, place.get("search_type", "manual"))
                st.success(f"📌 Added to list: {disp_label}")

        if idx < len(st.session_state.search_results) - 1:
            st.markdown("---")

# Display selected coordinates
if st.session_state.selected_place_name:
    st.info(f"📌 Selected place: **{st.session_state.selected_place_name}** ({st.session_state.selected_lat:.6f}, {st.session_state.selected_lon:.6f})")

st.markdown("---")
st.markdown("### 3️⃣ Scenario Parameters")
colA, colB, colC = st.columns(3)
with colA:
    latitude  = st.number_input("Latitude", value=st.session_state.selected_lat, format="%.6f")
    incident_size = st.number_input("Patient Count (incident_size)", value=30, min_value=1, step=1)
    amb_velocity  = st.number_input("Ambulance Speed (km/h)", value=40, min_value=1, step=1)
    amb_handover_time = st.number_input("AMB Handover Time (min)", value=10.0, min_value=0.0, step=0.1, format="%.1f", help="Time to load/unload patients at scene or hospital")
    total_samples = st.number_input("Simulation Iterations (totalSamples)", value=30, min_value=1, step=1)
with colB:
    longitude = st.number_input("Longitude", value=st.session_state.selected_lon, format="%.6f")
    amb_count  = st.number_input("Ambulance Count (amb_count)", value=30, min_value=1, step=1)
    uav_velocity = st.number_input("UAV Speed (km/h)", value=80, min_value=1, step=1)
    uav_handover_time = st.number_input("UAV Handover Time (min)", value=15.0, min_value=0.0, step=0.1, format="%.1f", help="Time to load/unload patients at scene or hospital")
    random_seed  = st.number_input("Random Seed", value=0, min_value=0, step=1)
with colC:
    uav_count = st.number_input("UAV Count (uav_count)", value=3, min_value=0, step=1)
    hospital_max_send_coeff = st.text_input("max_send_coeff (e.g. 1.05,1)", value="1,1")
    buffer_ratio = st.number_input("buffer_ratio", value=1.5, min_value=1.0, step=0.1)


if st.button("📦 Generate Scenario", key="btn_generate_scenario"):
        # API key validation (Kakao key required only in is_use_time=True mode)
    if is_use_time and (not kakao_api_key or not kakao_api_key.strip()):
        st.error("⚠️ Kakao API key is required when 'Use Kakao API duration' is checked. "
                 "Uncheck it to use the OSRM backend instead.")
        st.stop()

    try:
        env = parse_env_kv(st.session_state.env_txt)
        extra_args = {
            "buffer_ratio": buffer_ratio,
            "hospital_max_send_coeff": hospital_max_send_coeff.strip(),
            "kakao_api_key": kakao_api_key.strip() if kakao_api_key else "",
            "departure_time": departure_time_str,
            "is_use_time": str(is_use_time).lower(),  # "true" or "false"
            "amb_handover_time": float(amb_handover_time),
            "uav_handover_time": float(uav_handover_time),
            "duration_coeff": float(duration_coeff),
        }
        orc = Orchestrator(base_path=bp)
        res = orc.generate_scenario(
            latitude=latitude, longitude=longitude,
            incident_size=int(incident_size),
            amb_count=int(amb_count), uav_count=int(uav_count),
            amb_velocity=int(amb_velocity), uav_velocity=int(uav_velocity),
            total_samples=int(total_samples), random_seed=int(random_seed),
            exp_id=None,  # Always auto-generated
            extra_env=env, extra_args=extra_args
        )

        # Store latest state
        st.session_state.gen_state = {
            "exp_id": res["exp_id"],
            "coord": res["coord"],
            "config_path": res["config_path"],
            "summary_csv_path": res["summary_csv_path"],
            "summary_csv_path_legacy": res["summary_csv_path_legacy"],
            "log_file": res["log_file"]
        }

        label_text = (st.session_state.selected_place_name or "").strip()
        if label_text:
            try:
                _write_label_map(bp, [{
                    "exp_id": res["exp_id"],
                    "coord": str(res["coord"]),
                    "label": label_text,
                    "source": "single",
                }])
            except Exception as lm_err:
                st.warning(f"label_map update failed: {lm_err}")

        st.success("✅ Scenario generated!")
        st.write(f"• Exp ID: `{res['exp_id']}`")
        st.write(f"• Coord: `{res['coord']}`")
        st.write(f"• CONFIG_PATH: `{res['config_path']}`")
        st.write(f"• Summary CSV: `{res['summary_csv_path']}`")
        st.write(f"• Log file: `{res['log_file']}`")

    except Exception as e:
        st.error("❌ Scenario generation error")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────
# 4. Run the just-generated scenario immediately
# ─────────────────────────────────────────────────────────────────
if st.session_state.gen_state and st.session_state.gen_state.get("config_path"):
    st.markdown("---")
    st.markdown("### 4️⃣ Run Generated Scenario Immediately")
    st.info("Run simulation for the scenario generated above.")
    st.code(f"CONFIG: {st.session_state.gen_state.get('config_path')}")

    if st.button("▶️ Run Simulation Now", key="btn_immediate_run"):
        try:
            orc_imm = Orchestrator(base_path=bp)
            res_imm = orc_imm.run_simulation(config_path=st.session_state.gen_state["config_path"])

            # Update latest state
            st.session_state.gen_state.update({
                "exp_id": res_imm["exp_id"],
                "coord": res_imm["coord"],
                "config_path": res_imm["config_path"],
                "summary_csv_path": res_imm["summary_csv_path"],
                "summary_csv_path_legacy": res_imm["summary_csv_path_legacy"],
                "log_file": res_imm["log_file"]
            })

            st.success("✅ Simulation complete!")
            st.write(f"• Log file: `{res_imm['log_file']}`")
            st.caption("Check results in the main app Scenarios/Maps tabs.")

        except Exception as e_imm:
            st.error("❌ Simulation execution error")
            st.exception(e_imm)

# ------------------------------
# 5. Batch (multi) scenario generation/execution
# ------------------------------
st.markdown("---")
st.markdown("### 4️⃣ Batch Scenario Generation/Execution")
st.caption("Generate scenarios from the coordinate list at once, and optionally run simulations.")

# Preset editing
st.markdown("#### Edit Presets (Default Sets)")
default_preset = [{
    "name": "Default",
    "incident_size": 30,
    "amb_count": 30,
    "uav_count": 3,
    "amb_velocity": 40,
    "uav_velocity": 80,
    "amb_handover": 10.0,
    "uav_handover": 15.0,
    "total_samples": 30,
    "random_seed": 0,
    "buffer_ratio": 1.5,
    "max_send_coeff": "1,1",
    "is_use_time": True,
    "duration_coeff": 1.0,
}]
preset_df = pd.DataFrame(st.session_state.batch_presets)
if preset_df.empty:
    preset_df = pd.DataFrame(default_preset)
if isinstance(st.session_state.get("preset_editor"), pd.DataFrame):
    preset_df = st.session_state.preset_editor
preset_edited = st.data_editor(
    preset_df,
    num_rows="dynamic",
    hide_index=True,
    width='stretch',
    column_config={
        "name": st.column_config.TextColumn("Preset Name", required=True),
        "incident_size": st.column_config.NumberColumn("incident_size", step=1, format="%d"),
        "amb_count": st.column_config.NumberColumn("amb_count", step=1, format="%d"),
        "uav_count": st.column_config.NumberColumn("uav_count", step=1, format="%d"),
        "amb_velocity": st.column_config.NumberColumn("AMB Speed", step=1, format="%d"),
        "uav_velocity": st.column_config.NumberColumn("UAV Speed", step=1, format="%d"),
        "amb_handover": st.column_config.NumberColumn("AMB Handover(min)", step=0.1, format="%.1f"),
        "uav_handover": st.column_config.NumberColumn("UAV Handover(min)", step=0.1, format="%.1f"),
        "total_samples": st.column_config.NumberColumn("totalSamples", step=1, format="%d"),
        "random_seed": st.column_config.NumberColumn("random_seed", step=1, format="%d"),
        "buffer_ratio": st.column_config.NumberColumn("buffer_ratio", step=0.1, format="%.2f"),
        "max_send_coeff": st.column_config.TextColumn("max_send_coeff"),
        "is_use_time": st.column_config.CheckboxColumn("Use API Duration"),
        "duration_coeff": st.column_config.NumberColumn("duration_coeff", step=0.1, format="%.1f"),
    },
    key="preset_editor"
)
if isinstance(st.session_state.get("preset_editor"), pd.DataFrame):
    preset_edited = st.session_state.preset_editor
presets_clean = preset_edited.dropna(how="all").to_dict(orient="records")
st.session_state.batch_presets = presets_clean if presets_clean else default_preset
preset_names = [p.get("name", "") for p in st.session_state.batch_presets]

# Coordinate + parameter combined table
st.markdown("#### Coordinate & Parameter Table")
coord_columns = [
    "label", "lat", "lon", "address", "preset",
    "incident_size", "amb_count", "uav_count",
    "amb_velocity", "uav_velocity",
    "amb_handover", "uav_handover",
    "total_samples", "random_seed",
    "buffer_ratio", "max_send_coeff",
    "is_use_time", "duration_coeff",
    "source"
]
if st.session_state.batch_coord_rows:
    coord_df = pd.DataFrame(st.session_state.batch_coord_rows)
else:
    coord_df = pd.DataFrame(columns=coord_columns)
coord_df = coord_df.reindex(columns=coord_columns)
if isinstance(st.session_state.get("batch_coord_editor_v2"), pd.DataFrame):
    coord_df = st.session_state.batch_coord_editor_v2
coord_edited = st.data_editor(
    coord_df,
    num_rows="dynamic",
    hide_index=True,
    width='stretch',
    column_config={
        "label": st.column_config.TextColumn("Label", help="Display name for result comparison"),
        "lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
        "lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
        "address": st.column_config.TextColumn("Address", required=False),
        "preset": st.column_config.SelectboxColumn("Preset", options=preset_names or ["Default"]),
        "incident_size": st.column_config.NumberColumn("incident_size", step=1, format="%d"),
        "amb_count": st.column_config.NumberColumn("amb_count", step=1, format="%d"),
        "uav_count": st.column_config.NumberColumn("uav_count", step=1, format="%d"),
        "amb_velocity": st.column_config.NumberColumn("AMB Speed", step=1, format="%d"),
        "uav_velocity": st.column_config.NumberColumn("UAV Speed", step=1, format="%d"),
        "amb_handover": st.column_config.NumberColumn("AMB Handover(min)", step=0.1, format="%.1f"),
        "uav_handover": st.column_config.NumberColumn("UAV Handover(min)", step=0.1, format="%.1f"),
        "total_samples": st.column_config.NumberColumn("totalSamples", step=1, format="%d"),
        "random_seed": st.column_config.NumberColumn("random_seed", step=1, format="%d"),
        "buffer_ratio": st.column_config.NumberColumn("buffer_ratio", step=0.1, format="%.2f"),
        "max_send_coeff": st.column_config.TextColumn("max_send_coeff"),
        "is_use_time": st.column_config.CheckboxColumn("Use API Duration"),
        "duration_coeff": st.column_config.NumberColumn("duration_coeff", step=0.1, format="%.1f"),
        "source": st.column_config.TextColumn("Source", disabled=True),
    },
    key="batch_coord_editor_v2"
)
if isinstance(st.session_state.get("batch_coord_editor_v2"), pd.DataFrame):
    coord_edited = st.session_state.batch_coord_editor_v2
st.session_state.batch_coord_rows = coord_edited.dropna(how="all").to_dict(orient="records")

# Execution settings
st.markdown("**Batch Execution Settings**")
col_b1, col_b2 = st.columns(2)
with col_b1:
    batch_prefix = st.text_input("exp_id prefix", value="batch")
    st.caption("Prefix after exp_; ASCII-only recommended")
    add_ts = st.checkbox("Auto Timestamp", value=True)
    if re.search(r"\s", batch_prefix or "") or any(ord(ch) > 127 for ch in batch_prefix):
        st.warning("exp_id prefix should use ASCII characters only.")
with col_b2:
    do_run = st.radio("Execution Mode", ["Generate Only", "Generate + Simulate"], horizontal=True)

if st.button("Batch Run", type="primary", key="btn_batch_run"):
    rows = [r for r in st.session_state.batch_coord_rows if pd.notna(r.get("lat")) and pd.notna(r.get("lon"))]
    if not rows:
        st.error("Enter lat/lon in the coordinate table or use 'Add to List'.")
    elif not kakao_api_key:
        st.error("Please enter the Kakao REST API key first.")
    else:
        prefix = batch_prefix.strip() or "batch"
        if add_ts:
            prefix = f"{prefix}_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}"
        env = parse_env_kv(st.session_state.env_txt)
        run_log = []
        label_records = []
        orc = Orchestrator(base_path=bp)
        preset_lookup = {p.get("name"): p for p in st.session_state.batch_presets}

        for ridx, row in enumerate(rows, start=1):
            preset_name = row.get("preset") or (preset_names[0] if preset_names else "default")
            preset = preset_lookup.get(preset_name) or _get_preset_by_name(preset_name)

            def pick(key, default):
                val = row.get(key)
                return preset.get(key, default) if pd.isna(val) else val

            lat_val = float(row.get("lat"))
            lon_val = float(row.get("lon"))
            label_val = row.get("label") or f"coord{ridx}"
            exp_id = f"{prefix}_c{ridx}"
            extra_args = {
                "buffer_ratio": float(pick("buffer_ratio", 1.5)),
                "hospital_max_send_coeff": str(pick("max_send_coeff", "1,1")).strip(),
                "kakao_api_key": kakao_api_key.strip(),
                "departure_time": departure_time_str,
                "is_use_time": str(bool(pick("is_use_time", True))).lower(),
                "amb_handover_time": float(pick("amb_handover", 10.0)),
                "uav_handover_time": float(pick("uav_handover", 15.0)),
                "duration_coeff": float(pick("duration_coeff", 1.0)),
            }
            rec = {
                "exp_id": exp_id,
                "label": label_val,
                "coord": (lat_val, lon_val),
                "preset": preset_name,
                "status": "pending",
                "config_path": None,
                "log_file": None,
                "error": "",
            }


            try:
                gen = orc.generate_scenario(
                    latitude=lat_val,
                    longitude=lon_val,
                    incident_size=int(pick("incident_size", 30)),
                    amb_count=int(pick("amb_count", 30)),
                    uav_count=int(pick("uav_count", 3)),
                    amb_velocity=int(pick("amb_velocity", 40)),
                    uav_velocity=int(pick("uav_velocity", 80)),
                    total_samples=int(pick("total_samples", 30)),
                    random_seed=int(pick("random_seed", 0)),
                    exp_id=exp_id,
                    extra_env=env,
                    extra_args=extra_args,
                )
                rec["status"] = "generated"
                rec["config_path"] = gen.get("config_path")
                rec["log_file"] = gen.get("log_file")
                exp_id_actual = gen.get("exp_id", exp_id)
                coord_str = str(gen.get("coord") or f"({lat_val},{lon_val})")
                label_records.append({
                    "exp_id": exp_id_actual,
                    "coord": coord_str,
                    "label": label_val,
                    "source": row.get("source") or ("keyword" if row.get("source") == "keyword" else "manual"),
                })

                if do_run == "Generate + Simulate" and gen.get("config_path"):
                    sim = orc.run_simulation(config_path=gen["config_path"], extra_env=env)
                    rec["status"] = "simulated" if sim.get("ok") else f"sim fail ({sim.get('returncode')})"
                    rec["log_file"] = sim.get("log_file") or rec["log_file"]
                run_log.append(rec)
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)
                run_log.append(rec)

        st.session_state.batch_run_log = run_log
        try:
            _write_label_map(bp, label_records)
        except Exception as lm_err:
            st.warning(f"Label save warning: {lm_err}")
        st.success(f"Batch execution complete ({len(run_log)} items)")

if st.session_state.batch_run_log:
    st.markdown("**Batch Execution Log**")
    st.dataframe(pd.DataFrame(st.session_state.batch_run_log), width='stretch', hide_index=True)

st.markdown("---")
st.caption("💡 Check generated scenarios in the main app, or modify and re-run existing ones.")
