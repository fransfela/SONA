"""
SONA - fetcher.py
Spatial Outdoor Noise & Acoustics
Fetch points of interest and urban morphology data from OpenStreetMap
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ── Pilot area definitions ───────────────────────────────────────────────────
PILOT_AREAS = {
    "jakarta_pusat":   (-6.2088, 106.8456, 5000),
    "jakarta_selatan": (-6.2615, 106.8106, 8000),
    "jakarta_utara":   (-6.1384, 106.8134, 6000),
    "depok":           (-6.4025, 106.7942, 6000),
    "bekasi":          (-6.2383, 106.9756, 6000),
    "tangerang":       (-6.1783, 106.6319, 6000),
}

# ── Preset POI tag bundles ───────────────────────────────────────────────────
POI_PRESETS = {
    "mosque": {
        "amenity": ["mosque"],
        "building": ["mosque"],
    },
    "school": {
        "amenity": ["school", "university", "college"],
    },
    "stadium": {
        "leisure": ["stadium", "sports_centre"],
    },
    "market": {
        "amenity": ["marketplace"],
        "shop": ["supermarket"],
    },
    "park": {
        "leisure": ["park", "recreation_ground"],
    },
    "hospital": {
        "amenity": ["hospital", "clinic"],
    },
}


def fetch_poi(
    area_name: str = "jakarta_pusat",
    poi_type: str = "mosque",
    custom_tags: dict = None,
    save: bool = True,
) -> gpd.GeoDataFrame:
    """
    Fetch points of interest within a pilot area from OSM.

    Parameters
    ----------
    area_name   : str   — key from PILOT_AREAS
    poi_type    : str   — key from POI_PRESETS (ignored if custom_tags given)
    custom_tags : dict  — custom OSM tags, e.g. {"amenity": ["cafe"]}
    save        : bool  — save result to data/raw/ as GeoJSON

    Returns
    -------
    GeoDataFrame with POI locations and metadata
    """
    if area_name not in PILOT_AREAS:
        raise ValueError(f"Area '{area_name}' not in PILOT_AREAS. Options: {list(PILOT_AREAS.keys())}")

    lat, lon, radius = PILOT_AREAS[area_name]
    tags = custom_tags if custom_tags else POI_PRESETS.get(poi_type, {})

    if not tags:
        raise ValueError(f"POI type '{poi_type}' not found. Options: {list(POI_PRESETS.keys())}")

    print(f"\n📡 SONA Fetcher")
    print(f"   Area    : {area_name}")
    print(f"   POI     : {poi_type}")
    print(f"   Center  : ({lat}, {lon})")
    print(f"   Radius  : {radius}m")

    gdf = ox.features_from_point(
        center_point=(lat, lon),
        tags=tags,
        dist=radius,
    )

    # Normalize columns
    cols_priority = [
        "geometry", "name", "amenity", "building", "leisure",
        "addr:street", "addr:city", "capacity",
    ]
    existing = [c for c in cols_priority if c in gdf.columns]
    gdf = gdf[existing].copy().reset_index()

    # Add derived fields
    gdf["lat"]       = gdf.geometry.centroid.y
    gdf["lon"]       = gdf.geometry.centroid.x
    gdf["area_name"] = area_name
    gdf["poi_type"]  = poi_type

    print(f"✅ Found: {len(gdf)} {poi_type}(s) in {area_name}")

    if save:
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{poi_type}_{area_name}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"💾 Saved : {out_path}")

    return gdf


def fetch_urban_context(
    area_name: str = "jakarta_pusat",
    save: bool = True,
) -> dict:
    """
    Fetch urban morphology context for acoustic analysis:
    - Building footprints (density, height proxy)
    - Road network (noise floor proxy)
    - Land use zones

    Returns dict of GeoDataFrames: {"buildings": gdf, "roads": gdf, "landuse": gdf}
    """
    if area_name not in PILOT_AREAS:
        raise ValueError(f"Area '{area_name}' not in PILOT_AREAS.")

    lat, lon, radius = PILOT_AREAS[area_name]
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏙️  Fetching urban context for: {area_name}")

    results = {}

    # Buildings
    try:
        buildings = ox.features_from_point(
            (lat, lon),
            tags={"building": True},
            dist=radius,
        )
        buildings = buildings[["geometry"]].copy().reset_index()
        buildings["area_name"] = area_name
        results["buildings"] = buildings
        print(f"   🏠 Buildings : {len(buildings)}")
        if save:
            buildings.to_file(out_dir / f"buildings_{area_name}.geojson", driver="GeoJSON")
    except Exception as e:
        print(f"   ⚠️  Buildings failed: {e}")

    # Roads
    try:
        G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
        _, roads = ox.graph_to_gdfs(G)
        roads = roads[["geometry", "name", "highway", "lanes", "maxspeed"]].copy().reset_index()
        roads["area_name"] = area_name
        results["roads"] = roads
        print(f"   🛣️  Road segments: {len(roads)}")
        if save:
            roads.to_file(out_dir / f"roads_{area_name}.geojson", driver="GeoJSON")
    except Exception as e:
        print(f"   ⚠️  Roads failed: {e}")

    # Land use
    try:
        landuse = ox.features_from_point(
            (lat, lon),
            tags={"landuse": True},
            dist=radius,
        )
        landuse = landuse[["geometry", "landuse"]].copy().reset_index()
        landuse["area_name"] = area_name
        results["landuse"] = landuse
        print(f"   🌿 Land use zones: {len(landuse)}")
        if save:
            landuse.to_file(out_dir / f"landuse_{area_name}.geojson", driver="GeoJSON")
    except Exception as e:
        print(f"   ⚠️  Land use failed: {e}")

    print(f"✅ Urban context fetch complete for {area_name}")
    return results


def summary(gdf: gpd.GeoDataFrame) -> None:
    """Print quick summary of a fetched GeoDataFrame."""
    print("\n── SONA Fetch Summary ────────────────────────")
    print(f"   Total entries : {len(gdf)}")
    if "area_name" in gdf.columns:
        print(f"   Areas         : {gdf['area_name'].unique().tolist()}")
    if "poi_type" in gdf.columns:
        print(f"   POI type      : {gdf['poi_type'].unique().tolist()}")
    if "name" in gdf.columns:
        named = gdf["name"].notna().sum()
        print(f"   Named entries : {named} / {len(gdf)}")
    print("──────────────────────────────────────────────\n")


if __name__ == "__main__":
    # Test 1: fetch mosques di Jakarta Pusat
    gdf = fetch_poi(area_name="jakarta_pusat", poi_type="mosque")
    summary(gdf)
    print(gdf[["name", "lat", "lon"]].head(10))

    # Test 2: fetch urban context
    context = fetch_urban_context(area_name="jakarta_pusat")
    print(f"\nContext layers fetched: {list(context.keys())}")