"""
generate_coords.py
Generate N random coordinates within the South Korean land boundary (shapefile) → save as CSV + folium map

Usage:
    python experiment_1/generate_coords.py [--n 1000] [--seed 42] \
        [--shp ctprvn.shp] [--out experiment_1/coords_korea_1000.csv]

The shapefile (ctprvn.shp/.shx/.dbf) must be located in the experiment_1/ folder.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Directory where this script is located (experiment_1/)
_SCRIPT_DIR = Path(__file__).resolve().parent
# Project root (MCI_ADV/)
_PROJECT_ROOT = _SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate random coordinates within the South Korean land boundary")
    parser.add_argument("--n",    type=int,   default=1000,
                        help="Number of coordinates to generate (default: 1000)")
    parser.add_argument("--seed", type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--shp",  type=str,   default="ctprvn.shp",
                        help="Shapefile path (default: ctprvn.shp, relative to experiment_1/)")
    parser.add_argument("--out",  type=str,   default="experiment_1/coords_korea.csv",
                        help="Output CSV path (relative to project root)")
    parser.add_argument("--map",  type=str,   default="",
                        help="Folium HTML output path (default: <out_dir>/coords_map.html)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Coordinate generation
# ---------------------------------------------------------------------------

def load_korea_boundary(shp_path: str):
    """Load shapefile and return a single WGS84 polygon."""
    try:
        import geopandas as gpd
    except ImportError:
        print("[ERROR] geopandas is not installed. Please run: pip install geopandas")
        sys.exit(1)

    shp_path = Path(shp_path)
    if not shp_path.exists():
        print(f"[ERROR] Shapefile not found: {shp_path}")
        sys.exit(1)

    print(f"[1/4] Loading shapefile: {shp_path}")
    korea = gpd.read_file(str(shp_path))
    # If CRS is missing (naive geometry), manually assign EPSG:5179
    if korea.crs is None:
        korea = korea.set_crs(epsg=5179)
    korea_wgs84 = korea.to_crs(epsg=4326)         # Convert to WGS84
    korea_union = korea_wgs84.geometry.union_all() # Merge into a single polygon for all of South Korea
    print(f"      CRS: {korea_wgs84.crs} -> EPSG:4326 conversion complete")
    print(f"      bounds: {korea_union.bounds}")
    return korea_union


def generate_points(korea_union, n: int, seed: int) -> list:
    """Generate n coordinates within the boundary using rejection sampling."""
    try:
        import numpy as np
        from shapely.geometry import Point
    except ImportError as e:
        print(f"[ERROR] Required package missing: {e}")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = korea_union.bounds
    pts = []
    batch = max(n * 4, 10_000)  # hit rate within bounding box ≈ 25~30%

    print(f"[2/4] Generating coordinates (n={n}, seed={seed}) ...")
    while len(pts) < n:
        lons = rng.uniform(minx, maxx, batch)
        lats = rng.uniform(miny, maxy, batch)
        for lat, lon in zip(lats, lons):
            if korea_union.contains(Point(lon, lat)):
                pts.append((lat, lon))
                if len(pts) % 100 == 0:
                    print(f"      {len(pts)}/{n}")
                if len(pts) >= n:
                    break

    print(f"      {n} coordinates generated")
    return pts[:n]


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(pts: list, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[3/4] Saving CSV: {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coord_id", "latitude", "longitude", "generated_at"])
        for i, (lat, lon) in enumerate(pts, start=1):
            writer.writerow([i, round(lat, 8), round(lon, 8), generated_at])
    print(f"      {len(pts)} rows saved")


# ---------------------------------------------------------------------------
# Folium visualization
# ---------------------------------------------------------------------------

def save_map(pts: list, map_path: str):
    try:
        import folium
    except ImportError:
        print("[WARN] folium is not installed; skipping map generation. pip install folium")
        return

    map_path = Path(map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] Saving folium map: {map_path}")
    center_lat = sum(p[0] for p in pts) / len(pts)
    center_lon = sum(p[1] for p in pts) / len(pts)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                   tiles="OpenStreetMap")

    for i, (lat, lon) in enumerate(pts, start=1):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.7,
            popup=f"coord_id={i}<br>({lat:.6f}, {lon:.6f})",
            tooltip=f"{i}",
        ).add_to(m)

    m.save(str(map_path))
    print(f"      {len(pts)} markers saved")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve shapefile path: search experiment_1/ first, then project root
    shp_path = Path(args.shp)
    if not shp_path.is_absolute() and not shp_path.exists():
        for base in (_SCRIPT_DIR, _PROJECT_ROOT):
            candidate = base / args.shp
            if candidate.exists():
                shp_path = candidate
                break

    # Output CSV path: relative paths are resolved from project root
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / args.out

    # Folium map path
    map_path: Path
    if not args.map:
        map_path = out_path.parent / "coords_map.html"
    else:
        mp = Path(args.map)
        map_path = mp if mp.is_absolute() else _PROJECT_ROOT / mp

    print("=" * 50)
    print("  South Korea Land Boundary Coordinate Generator")
    print("=" * 50)
    print(f"  n={args.n}, seed={args.seed}")
    print(f"  shp  : {shp_path}")
    print(f"  out  : {out_path}")
    print(f"  map  : {map_path}")
    print("=" * 50)

    korea_union = load_korea_boundary(str(shp_path))
    pts = generate_points(korea_union, args.n, args.seed)
    save_csv(pts, str(out_path))
    save_map(pts, str(map_path))

    print()
    print("=" * 50)
    print(f"  Done! {args.n} coordinates generated")
    print(f"  CSV : {out_path}")
    print(f"  Map : {map_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
