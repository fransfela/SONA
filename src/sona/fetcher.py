"""
SONA - fetcher.py
Spatial Outdoor Noise & Acoustics
Fetch place-of-interest data from OpenStreetMap via Overpass API
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import json
from pathlib import Path

# ── Bounding boxes pilot areas ──────────────────────────────────────────────
PILOT_AREAS = {
    "jakarta_pusat": (-6.2088, 106.8456, 5000),      # lat, lon, radius_meter
    "jakarta_selatan": (-6.2615, 106.8106, 8000),
    "depok": (-6.4025, 106.7942, 6000),
    "bekasi": (-6.2383, 106.9756, 6000),
    "tangerang": (-6.1783, 106.6319, 6000),
}

def fetch_mosques(area_name: str = "jakarta_pusat", save: bool = True) -> gpd.GeoDataFrame:
    """
    Fetch all mosques and musholas within a pilot area from OSM.
    
    Parameters
    ----------
    area_name : str
        Key from PILOT_AREAS dict
    save : bool
        Save result to data/raw/ as GeoJSON
    
    Returns
    -------
    GeoDataFrame with mosque locations and metadata
    """
    if area_name not in PILOT_AREAS:
        raise ValueError(f"Area '{area_name}' not found. Available: {list(PILOT_AREAS.keys())}")
    
    lat, lon, radius = PILOT_AREAS[area_name]
    
    print(f"📡 Fetching mosque data for: {area_name}")
    print(f"   Center: ({lat}, {lon}) | Radius: {radius}m")
    
    # Tags to query — mosque + mushola
    tags = {
        "amenity": ["mosque"],
        "building": ["mosque"],
    }
    
    # Fetch from OSM
    gdf = ox.features_from_point(
        center_point=(lat, lon),
        tags=tags,
        dist=radius
    )
    
    # Keep relevant columns only
    cols_to_keep = ["geometry", "name", "amenity", "building", "religion",
                    "addr:street", "addr:city", "capacity", "wheelchair"]
    existing_cols = [c for c in cols_to_keep if c in gdf.columns]
    gdf = gdf[existing_cols].copy()
    
    # Reset index — OSM returns multi-index (element_type, osmid)
    gdf = gdf.reset_index()
    
    # Add centroid coordinates for point-based analysis
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["area_name"] = area_name
    
    print(f"✅ Found {len(gdf)} mosques/musholas in {area_name}")
    
    if save:
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mosques_{area_name}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"💾 Saved to {out_path}")
    
    return gdf


def fetch_all_pilot_areas(save: bool = True) -> gpd.GeoDataFrame:
    """Fetch all pilot areas and combine into one GeoDataFrame."""
    all_gdfs = []
    
    for area_name in PILOT_AREAS:
        try:
            gdf = fetch_mosques(area_name, save=save)
            all_gdfs.append(gdf)
        except Exception as e:
            print(f"⚠️  Failed to fetch {area_name}: {e}")
    
    combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    
    if save:
        out_path = Path("data/raw/mosques_all_pilot.geojson")
        combined.to_file(out_path, driver="GeoJSON")
        print(f"\n💾 Combined dataset saved: {out_path}")
        print(f"📊 Total: {len(combined)} mosques across all pilot areas")
    
    return combined


def summary(gdf: gpd.GeoDataFrame) -> None:
    """Print quick summary of fetched data."""
    print("\n── SONA Fetch Summary ──────────────────────")
    print(f"Total entries  : {len(gdf)}")
    print(f"Areas covered  : {gdf['area_name'].unique().tolist()}")
    print(f"Columns        : {gdf.columns.tolist()}")
    if "name" in gdf.columns:
        named = gdf["name"].notna().sum()
        print(f"Named entries  : {named} / {len(gdf)}")
    print("──────────────────────────────────────────────\n")


if __name__ == "__main__":
    # Quick test — fetch jakarta pusat saja dulu
    gdf = fetch_mosques("jakarta_pusat")
    summary(gdf)
    print(gdf[["name", "lat", "lon"]].head(10))