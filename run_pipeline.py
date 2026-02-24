"""
SONA - run_pipeline.py
Offline analysis pipeline: fetch → classify → recommend → save
Run this once per area. Results are stored in data/processed/
Usage: python run_pipeline.py --area jakarta_pusat --poi mosque --sample 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src/sona")
from fetcher import fetch_poi, fetch_urban_context, PILOT_AREAS, POI_PRESETS
from classifier import classify_all
from acoustics import recommend_all


def run(area: str, poi_type: str, sample: int, buffer_radius: int, coverage_radius: int):
    t0 = time.time()

    print("\n" + "═" * 55)
    print("  🔊 SONA — Offline Analysis Pipeline")
    print("═" * 55)
    print(f"  Area            : {area}")
    print(f"  POI type        : {poi_type}")
    print(f"  Sample size     : {sample}")
    print(f"  Buffer radius   : {buffer_radius}m")
    print(f"  Coverage radius : {coverage_radius}m")
    print("═" * 55 + "\n")

    # ── Step 1: Fetch POI ────────────────────────────────────────────────────
    print("📡 [1/3] Fetching POI data...")
    poi_cache = Path(f"data/raw/{poi_type}_{area}.geojson")

    if poi_cache.exists():
        import geopandas as gpd
        poi_gdf = gpd.read_file(poi_cache)
        print(f"   ✅ Loaded from cache: {len(poi_gdf)} {poi_type}(s)")
    else:
        poi_gdf = fetch_poi(area, poi_type=poi_type, save=True)

    sample_gdf = poi_gdf.head(sample)
    print(f"   Using sample: {len(sample_gdf)} POIs\n")

    # ── Step 2: Fetch urban context ──────────────────────────────────────────
    print("🏙️  [2/3] Fetching urban context...")

    buildings_cache = Path(f"data/raw/buildings_{area}.geojson")
    roads_cache     = Path(f"data/raw/roads_{area}.geojson")
    landuse_cache   = Path(f"data/raw/landuse_{area}.geojson")

    import geopandas as gpd

    if buildings_cache.exists() and roads_cache.exists() and landuse_cache.exists():
        print("   ✅ Loading context from cache...")
        context = {
            "buildings": gpd.read_file(buildings_cache),
            "roads":     gpd.read_file(roads_cache),
            "landuse":   gpd.read_file(landuse_cache),
        }
        print(f"   🏠 Buildings : {len(context['buildings'])}")
        print(f"   🛣️  Roads     : {len(context['roads'])}")
        print(f"   🌿 Land use  : {len(context['landuse'])}")
    else:
        context = fetch_urban_context(area, save=True)

    print()

    # ── Step 3: Classify + Recommend ────────────────────────────────────────
    print("🧠 [3/3] Classifying + computing recommendations...")

    classified = classify_all(
        sample_gdf,
        context.get("buildings"),
        context.get("roads"),
        context.get("landuse"),
        buffer_radius_m=buffer_radius,
        area_name=area,
        save=False,
    )

    result = recommend_all(
        classified,
        coverage_radius_m=coverage_radius,
        area_name=area,
        save=True,
    )

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    out_path = Path(f"data/processed/recommendations_{area}.geojson")

    print("\n" + "═" * 55)
    print("  ✅ Pipeline complete")
    print(f"  Output : {out_path}")
    print(f"  POIs   : {len(result)}")
    print(f"  Time   : {elapsed:.1f}s")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONA Offline Analysis Pipeline")
    parser.add_argument("--area",     default="jakarta_pusat", choices=list(PILOT_AREAS.keys()))
    parser.add_argument("--poi",      default="mosque",        choices=list(POI_PRESETS.keys()))
    parser.add_argument("--sample",   default=100,  type=int)
    parser.add_argument("--buffer",   default=200,  type=int)
    parser.add_argument("--coverage", default=150,  type=int)
    args = parser.parse_args()

    run(args.area, args.poi, args.sample, args.buffer, args.coverage)