"""
batch_runner.py
Batch experiment runner for 1000 Korean coordinates (scenario generation + simulation automated pipeline)

Pipeline (per coordinate):
    1. Orchestrator.generate_scenario()  <- Kakao API call -> scenario CSV + YAML generation
    2. Orchestrator.run_simulation()     <- simulation execution (no API)
    3. progress.json update

API rate-limit handling:
    - Before exhausting the daily budget (--daily-limit), pre-calculates the number of coordinates to process
    - If generate_scenario subprocess terminates abnormally due to API errors (429, timeout, etc.),
      detected via RuntimeError or returncode != 0 -> marks that coordinate as failed
    - On the next run, failed coordinates are automatically retried (within --max-retries limit)

Usage (same command every day, automatically resumes from progress.json):
    python experiment_1/batch_runner.py \
        --base-path C:/Users/User/MCI_ADV \
        --coords experiment_1/coords_korea_1000.csv \
        --progress experiment_1/progress.json \
        --kakao-api-key YOUR_KEY \
        --experiment-id exp_batch_research \
        --daily-limit 4900 --calls-per-coord 55 \
        --incident-size 30 --amb-count 30 --uav-count 3 \
        --amb-velocity 40 --uav-velocity 80 \
        --total-samples 30 --random-seed 42

Check progress only:
    python experiment_1/batch_runner.py --status \
        --progress experiment_1/progress.json
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup (Orchestrator import)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCE_SRC = _PROJECT_ROOT / "src" / "sce_src"
if str(_SCE_SRC) not in sys.path:
    sys.path.insert(0, str(_SCE_SRC))

KST = timezone(timedelta(hours=9))


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MCI batch experiment runner")

    # Paths
    p.add_argument("--base-path",   default=str(_PROJECT_ROOT),
                   help="Project root path")
    p.add_argument("--coords",      default="experiment_1/coords_korea.csv",
                   help="Coordinates CSV path (default: experiment_1/coords_korea.csv)")
    p.add_argument("--progress",    default="experiment_1/progress.json",
                   help="Progress state JSON path (default: experiment_1/progress.json)")

    # API
    p.add_argument("--kakao-api-key", default="", help="Kakao REST API key")
    p.add_argument("--departure-time", default="",
                   help="Kakao routing departure time (YYYYMMDDHHmm, optional)")

    # ── Experiment identification ─────────────────────────────────────────
    p.add_argument("--experiment-id",  default="exp_batch_research",
                   help="Experiment ID (subfolder name under scenarios/ and progress.json identifier)")

    # ── Common scenario/simulation parameters ────────────────────────────
    p.add_argument("--incident-size",  type=int,   default=30,
                   help="Number of patients at the incident scene (default: 30)")
    p.add_argument("--amb-count",      type=int,   default=30,
                   help="Number of dispatched ambulances (default: 30)")
    p.add_argument("--uav-count",      type=int,   default=3,
                   help="Number of dispatched UAVs (default: 3)")
    p.add_argument("--amb-velocity",   type=int,   default=40,
                   help="Average ambulance speed in km/h (default: 40)")
    p.add_argument("--uav-velocity",   type=int,   default=80,
                   help="Average UAV speed in km/h (default: 80)")
    p.add_argument("--total-samples",  type=int,   default=30,
                   help="Number of simulation iterations (default: 30)")
    p.add_argument("--random-seed",    type=int,   default=42,
                   help="Random seed (default: 42)")

    # ── Transport time parameters ─────────────────────────────────────────
    p.add_argument("--amb-handover-time", type=float, default=10.0,
                   help="Ambulance patient handover time in minutes (default: 10.0)")
    p.add_argument("--uav-handover-time", type=float, default=15.0,
                   help="UAV patient handover time in minutes (default: 15.0)")
    p.add_argument("--is-use-time",    type=str,   default="true",
                   help="true: use Kakao API duration / false: OSRM static distance-based (distance/velocity) (default: true)")
    p.add_argument("--duration-coeff", type=float, default=1.0,
                   help="API duration time weight coefficient (default: 1.0)")
    p.add_argument("--osrm-url",       type=str,   default=None,
                   help="OSRM HTTP API base URL (used when is_use_time=false. default: env MCI_OSRM_URL or https://router.project-osrm.org)")

    # ── Hospital assignment parameters ────────────────────────────────────
    p.add_argument("--hospital-max-send-coeff", type=str, default=None,
                   help="Hospital send coefficient 'a,b' format (e.g., 1.1,1.0). Uses default if not specified")
    p.add_argument("--buffer-ratio",   type=float, default=None,
                   help="Candidate hospital buffer multiplier (uses make_csv_yaml_dynamic internal default if not specified)")
    p.add_argument("--util-by-tier",   type=str,   default=None,
                   help="Hospital utilization rate by tier '1:0.90,11:0.75,etc:0.60' format")

    # Batch control
    p.add_argument("--daily-limit",    type=int, default=4900,
                   help="Maximum daily Kakao API calls (default: 4900)")
    p.add_argument("--calls-per-coord", type=int, default=55,
                   help="Estimated API calls per coordinate (default: 55)")
    p.add_argument("--max-retries",    type=int, default=2,
                   help="Maximum retries per coordinate (default: 2)")

    # Mode
    p.add_argument("--dry-run",  action="store_true",
                   help="Print planned processing list without making API calls")
    p.add_argument("--status",   action="store_true",
                   help="Print progress status and exit")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Load coordinates CSV
# ---------------------------------------------------------------------------

def load_coords(coords_path: str) -> dict:
    """Return coord_id -> {latitude, longitude} dictionary"""
    path = _resolve_path(coords_path)
    if not path.exists():
        print(f"[ERROR] Coordinates CSV not found: {path}")
        sys.exit(1)

    coords = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(int(row["coord_id"]))
            coords[cid] = {
                "latitude":  float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
    return coords


# ---------------------------------------------------------------------------
# progress.json management
# ---------------------------------------------------------------------------

def _resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def load_progress(progress_path: str, experiment_id: str, total: int) -> dict:
    path = _resolve_path(progress_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    # Initial creation
    return {
        "experiment_id": experiment_id,
        "total": total,
        "statuses": {str(i): {"status": "pending"} for i in range(1, total + 1)},
        "api_log": [],
    }


def save_progress(data: dict, progress_path: str):
    """Atomic write: write to .tmp file then os.replace()"""
    path = _resolve_path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Progress statistics
# ---------------------------------------------------------------------------

def calc_stats(progress: dict) -> dict:
    statuses = progress.get("statuses", {})
    total = progress.get("total", len(statuses))

    done        = sum(1 for v in statuses.values() if v.get("status") == "done")
    sim_ok      = sum(1 for v in statuses.values()
                      if v.get("status") == "done" and v.get("sim_ok", False))
    sim_fail    = done - sim_ok
    failed      = sum(1 for v in statuses.values() if v.get("status") == "failed")
    abandoned   = sum(1 for v in statuses.values() if v.get("status") == "abandoned")
    pending     = sum(1 for v in statuses.values()
                      if v.get("status") in ("pending", "running"))

    # Today's API usage
    today = today_kst()
    today_calls = 0
    today_processed = 0
    for entry in progress.get("api_log", []):
        if entry.get("date") == today:
            today_calls     += entry.get("calls_used", 0)
            today_processed += entry.get("coords_processed", 0)

    return dict(
        total=total, done=done, sim_ok=sim_ok, sim_fail=sim_fail,
        failed=failed, abandoned=abandoned, pending=pending,
        today_calls=today_calls, today_processed=today_processed,
    )


def print_status(progress: dict, daily_limit: int, calls_per_coord: int):
    s = calc_stats(progress)
    today = today_kst()
    print()
    print(f"=== Progress Status ({now_kst()}) ===")
    print(f"Done      : {s['done']} / {s['total']}  ({s['done']/s['total']*100:.1f}%)")
    print(f"  |- Sim OK   : {s['sim_ok']}")
    print(f"  +- Sim Fail : {s['sim_fail']}")
    print(f"Errors    : {s['failed']}")
    print(f"Abandoned : {s['abandoned']}")
    print(f"Pending   : {s['pending']}")
    print(f"Today API usage: {s['today_calls']} / {daily_limit}")
    remaining_budget = daily_limit - s['today_calls']
    can_process = max(0, remaining_budget // calls_per_coord)
    print(f"Today remaining budget: {remaining_budget} calls -> approx. {can_process} coords processable")

    # Most recently processed coordinate
    last_done = None
    last_time = None
    for cid, v in progress.get("statuses", {}).items():
        if v.get("status") == "done" and v.get("finished_at"):
            if last_time is None or v["finished_at"] > last_time:
                last_time = v["finished_at"]
                last_done = cid
    if last_done:
        print(f"Last processed: coord_id={last_done} @ {last_time}")
    print()


# ---------------------------------------------------------------------------
# Select processing targets
# ---------------------------------------------------------------------------

def select_pending(progress: dict, max_retries: int) -> list:
    """Return sorted list of coord_ids that are pending or failed but retryable"""
    result = []
    for cid, v in progress.get("statuses", {}).items():
        st = v.get("status", "pending")
        if st == "pending":
            result.append(int(cid))
        elif st == "failed":
            if v.get("attempts", 0) < max_retries:
                result.append(int(cid))
    return sorted(result)


# ---------------------------------------------------------------------------
# Process a single coordinate
# ---------------------------------------------------------------------------

def process_coord(
    orch,
    coord_id: str,
    lat: float,
    lon: float,
    args,
    progress: dict,
    session_calls: int,
    session_idx: int,
    session_total: int,
) -> tuple:
    """
    Execute the full pipeline for a single coordinate: scenario generation -> simulation.

    Returns: (updated_progress, api_calls_used, sim_ok: bool)

    API error handling:
        generate_scenario() raises RuntimeError when make_csv_yaml_dynamic.py subprocess
        fails (CONFIG_PATH not found in stdout).
        When Kakao API 429/timeout etc. causes abnormal subprocess termination, this path is taken.
        -> caught by try/except, marked as failed, automatically retried in next session.
    """
    statuses = progress["statuses"]
    cur = statuses.get(coord_id, {"status": "pending"})
    attempts = cur.get("attempts", 0) + 1

    # Mark as running
    statuses[coord_id] = {
        "status": "running",
        "attempts": attempts,
        "started_at": now_kst(),
    }

    prefix = f"[{session_idx:3d}/{session_total}] coord_id={coord_id:>4s} ({lat:.4f}, {lon:.4f})"

    # ── Step 1: Scenario generation (Kakao API call) ─────────────────────
    print(f"{prefix}  Generating scenario...")
    t0 = time.time()

    # Assemble extra arguments to pass to make_csv_yaml_dynamic.py
    # (those not in orchestrator.generate_scenario() direct parameters)
    extra_args: dict = {}
    if args.kakao_api_key:
        extra_args["kakao_api_key"] = args.kakao_api_key
    if args.departure_time:
        extra_args["departure_time"] = args.departure_time
    # Transport/time parameters
    extra_args["is_use_time"]       = args.is_use_time
    extra_args["amb_handover_time"] = args.amb_handover_time
    extra_args["uav_handover_time"] = args.uav_handover_time
    extra_args["duration_coeff"]    = args.duration_coeff
    if args.osrm_url:
        extra_args["osrm_url"]      = args.osrm_url
    # Hospital assignment parameters (not passed if None -> uses make_csv internal defaults)
    if args.hospital_max_send_coeff is not None:
        extra_args["hospital_max_send_coeff"] = args.hospital_max_send_coeff
    if args.buffer_ratio is not None:
        extra_args["buffer_ratio"] = args.buffer_ratio
    if args.util_by_tier is not None:
        extra_args["util_by_tier"] = args.util_by_tier

    try:
        gen_result = orch.generate_scenario(
            latitude=lat,
            longitude=lon,
            incident_size=args.incident_size,
            amb_count=args.amb_count,
            uav_count=args.uav_count,
            amb_velocity=args.amb_velocity,
            uav_velocity=args.uav_velocity,
            total_samples=args.total_samples,
            random_seed=args.random_seed,
            exp_id=args.experiment_id,
            extra_args=extra_args,
        )
    except Exception as exc:
        # Handle all exceptions: API quota exceeded, timeout, CONFIG_PATH not output, etc.
        gen_elapsed = time.time() - t0
        err = f"{type(exc).__name__}: {str(exc)}"[:300]
        print(f"{prefix}  Scenario generation exception ({gen_elapsed:.1f}s) X  {err}")
        statuses[coord_id] = {
            "status": "failed",
            "step": "generate",
            "attempts": attempts,
            "error": err,
            "finished_at": now_kst(),
        }
        return progress, 0, False

    gen_elapsed = time.time() - t0

    # Actual API call count (falls back to estimate if unavailable)
    actual_calls = gen_result.get("api_call_count") or args.calls_per_coord

    # Subprocess completed but returncode != 0
    if not gen_result.get("ok"):
        err = (gen_result.get("stderr", "") or "")[:200].strip()
        if not err:
            err = f"returncode={gen_result.get('returncode')}"
        print(f"{prefix}  Scenario generation failed ({gen_elapsed:.1f}s) X  {err}")
        statuses[coord_id] = {
            "status": "failed",
            "step": "generate",
            "attempts": attempts,
            "error": err,
            "finished_at": now_kst(),
        }
        return progress, actual_calls, False

    config_path = gen_result.get("config_path", "")
    actual_calls = gen_result.get("api_call_count") or args.calls_per_coord
    print(f"{prefix}  Scenario done ({gen_elapsed:.1f}s, API {actual_calls} calls) -> Running simulation...")

    # ── Step 2: Run simulation (no API calls) ────────────────────────────
    # Simulation parameters (patient count, ambulance count, UAV count, speed, iterations, etc.)
    # are all written to the YAML at generate_scenario() time, so no extra arguments needed.
    # main.py only takes --config_path and reads all settings from the YAML.
    t1 = time.time()
    sim_ok = False
    sim_error = ""
    try:
        sim_result = orch.run_simulation(config_path)
        sim_ok = sim_result.get("ok", False)
        if not sim_ok:
            sim_error = (sim_result.get("stderr", "") or "")[:200].strip()
    except Exception as exc:
        sim_error = f"{type(exc).__name__}: {str(exc)}"[:200]
        print(f"{prefix}  Simulation exception X  {sim_error}")

    sim_elapsed = time.time() - t1
    cumulative_calls = session_calls + actual_calls

    status_icon = "[OK]" if sim_ok else "[SIM FAIL]"
    print(f"{prefix}  Simulation done ({sim_elapsed:.1f}s) {status_icon}  "
          f"API calls: {actual_calls}  Cumulative: {cumulative_calls} / {args.daily_limit}")

    statuses[coord_id] = {
        "status": "done",          # done if scenario generation completed, even if sim failed
        "config_path": config_path,
        "sim_ok": sim_ok,
        "attempts": attempts,
        "gen_elapsed_sec": round(gen_elapsed, 1),
        "sim_elapsed_sec": round(sim_elapsed, 1),
        "finished_at": now_kst(),
    }
    if sim_error:
        statuses[coord_id]["sim_error"] = sim_error

    return progress, actual_calls, sim_ok


# ---------------------------------------------------------------------------
# dry-run output
# ---------------------------------------------------------------------------

def print_dry_run(pending_ids: list, coords: dict, daily_limit: int,
                  calls_per_coord: int, today_calls: int):
    remaining_budget = daily_limit - today_calls
    can_process = max(0, remaining_budget // calls_per_coord)
    to_run = pending_ids[:can_process]

    print()
    print("=== [DRY-RUN] Planned Processing List ===")
    print(f"Today remaining budget : {remaining_budget} calls")
    print(f"Processable coords    : {can_process} (total pending: {len(pending_ids)})")
    print()
    print(f"{'#':>4}  {'coord_id':>8}  {'latitude':>10}  {'longitude':>11}")
    print("-" * 40)
    for i, cid in enumerate(to_run, start=1):
        c = coords.get(str(cid), {})
        print(f"{i:4d}  {cid:8d}  {c.get('latitude', 0):10.6f}  {c.get('longitude', 0):11.6f}")
    if len(pending_ids) > can_process:
        print(f"  ... remaining {len(pending_ids) - can_process} coords will be processed in next run")
    print()


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Normalize paths
    coords_path   = _resolve_path(args.coords)
    progress_path = _resolve_path(args.progress)

    # Load coordinates
    coords = load_coords(str(coords_path))
    total  = len(coords)

    # Load progress.json (create new if not found)
    progress = load_progress(str(progress_path), args.experiment_id, total)

    # ── --status mode ───────────────────────────────────────────────────────
    if args.status:
        print_status(progress, args.daily_limit, args.calls_per_coord)
        return

    # ── Select processing targets ────────────────────────────────────────────
    pending_ids = select_pending(progress, args.max_retries)

    s = calc_stats(progress)
    today_calls = s["today_calls"]
    remaining_budget = args.daily_limit - today_calls
    can_process = max(0, remaining_budget // args.calls_per_coord)
    session_targets = pending_ids[:can_process]

    # ── --dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        print_dry_run(pending_ids, coords, args.daily_limit,
                      args.calls_per_coord, today_calls)
        return

    # ── Print header ─────────────────────────────────────────────────────────
    est_days = (len(pending_ids) - len(session_targets))
    est_days_str = f"~{est_days // max(len(session_targets), 1) + 1} days" \
                   if session_targets else "N/A"

    print()
    print("=" * 55)
    print("  MCI Batch Runner")
    print("=" * 55)
    print(f"  Total coords : {total}")
    print(f"  Done         : {s['done']}  (sim OK {s['sim_ok']} / fail {s['sim_fail']})")
    print(f"  Failed(aband): {s['failed'] + s['abandoned']}")
    print(f"  Pending      : {s['pending']}")
    print(f"  Today budget : {remaining_budget} calls -> approx. {can_process} coords processable "
          f"({args.calls_per_coord} calls/coord)")
    print("-" * 55)

    if not session_targets:
        if remaining_budget <= 0:
            print("  Today's API budget exhausted. Please run again tomorrow.")
        else:
            print("  No coordinates to process. All coords are done or retry limit exceeded.")
        print("=" * 55)
        return

    # ── Initialize Orchestrator ─────────────────────────────────────────────
    try:
        from orchestrator import Orchestrator
    except ImportError as e:
        print(f"[ERROR] Orchestrator import failed: {e}")
        print(f"        Check that {_SCE_SRC} is included in sys.path.")
        sys.exit(1)

    orch = Orchestrator(str(_resolve_path(args.base_path)))

    # ── Batch processing loop ────────────────────────────────────────────────
    session_calls      = today_calls
    session_done       = 0
    session_sim_ok     = 0
    session_sim_fail   = 0
    session_failed     = 0
    session_start_time = time.time()

    try:
        for idx, cid in enumerate(session_targets, start=1):
            coord_id = str(cid)
            c = coords.get(coord_id)
            if c is None:
                print(f"[WARN] coord_id={coord_id} coordinate info not found. Skipping.")
                continue

            progress, calls_used, ok = process_coord(
                orch=orch,
                coord_id=coord_id,
                lat=c["latitude"],
                lon=c["longitude"],
                args=args,
                progress=progress,
                session_calls=session_calls,
                session_idx=idx,
                session_total=len(session_targets),
            )

            session_calls += calls_used
            session_done  += 1
            if progress["statuses"][coord_id]["status"] == "done":
                if ok:
                    session_sim_ok += 1
                else:
                    session_sim_fail += 1
            else:
                session_failed += 1

            # Save progress.json immediately (crash-safe)
            save_progress(progress, str(progress_path))

            # Check budget exceeded
            if session_calls >= args.daily_limit:
                print()
                print("[WARN] Today's API budget exhausted. Stopping batch.")
                break

    except KeyboardInterrupt:
        print()
        print("[WARN] User interrupted (Ctrl+C). progress.json saved up to this point.")
        save_progress(progress, str(progress_path))

    # ── Update api_log ─────────────────────────────────────────────────────
    today = today_kst()
    api_log = progress.get("api_log", [])
    # Find today's entry
    today_entry = next((e for e in api_log if e.get("date") == today), None)
    calls_this_session = session_calls - today_calls
    if today_entry:
        today_entry["calls_used"]       += calls_this_session
        today_entry["coords_processed"] += session_done
    else:
        api_log.append({
            "date":              today,
            "calls_used":        calls_this_session,
            "coords_processed":  session_done,
        })
    progress["api_log"] = api_log
    save_progress(progress, str(progress_path))

    # ── Print session summary ───────────────────────────────────────────────
    session_elapsed = time.time() - session_start_time
    remaining_after = len(pending_ids) - session_done

    final_stats = calc_stats(progress)
    days_remaining = (remaining_after // max(can_process, 1)) + 1 \
                     if remaining_after > 0 else 0

    print()
    print("-" * 55)
    print(f"  Session done: {session_done} processed  "
          f"(OK {session_sim_ok} / fail {session_sim_fail + session_failed})")
    print(f"  Remaining coords: {remaining_after} | Est. remaining days: ~{days_remaining}")
    print(f"  Elapsed  : {session_elapsed:.0f}s")
    print(f"  Cumulative done: {final_stats['done']} / {total}")
    print("=" * 55)


if __name__ == "__main__":
    main()
