"""
SONA - classifier.py
Spatial Outdoor Noise & Acoustics
Classify urban context around each POI based on OSM morphology data.

Urban Classes:
    0 - OPEN_FIELD      : parks, fields, low building density
    1 - RESIDENTIAL     : housing, kampung, low-mid rise
    2 - MIXED_COMMERCIAL: shops, markets, mixed use
    3 - DENSE_URBAN     : high-rise, CBD, dense buildings
    4 - INDUSTRIAL      : factories, warehouses, industrial zones

Rule-based MVP → ML clustering in next iteration.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point

# ── Class definitions ────────────────────────────────────────────────────────
URBAN_CLASSES = {
    0: "OPEN_FIELD",
    1: "RESIDENTIAL",
    2: "MIXED_COMMERCIAL",
    3: "DENSE_URBAN",
    4: "INDUSTRIAL",
}

# Acoustic properties per class (used later by acoustics.py)
CLASS_ACOUSTIC_PROFILE = {
    "OPEN_FIELD": {
        "noise_floor_dB": 45,
        "rt60_est_s": 0.3,
        "reflection_factor": 0.1,
        "description": "Low ambient noise, minimal reflections, sound disperses freely",
    },
    "RESIDENTIAL": {
        "noise_floor_dB": 55,
        "rt60_est_s": 0.6,
        "reflection_factor": 0.4,
        "description": "Moderate ambient noise, some building reflections",
    },
    "MIXED_COMMERCIAL": {
        "noise_floor_dB": 65,
        "rt60_est_s": 0.8,
        "reflection_factor": 0.6,
        "description": "High ambient noise, significant reflections from facades",
    },
    "DENSE_URBAN": {
        "noise_floor_dB": 70,
        "rt60_est_s": 1.2,
        "reflection_factor": 0.8,
        "description": "Very high ambient noise, strong urban canyon reflections",
    },
    "INDUSTRIAL": {
        "noise_floor_dB": 72,
        "rt60_est_s": 0.5,
        "reflection_factor": 0.3,
        "description": "Very high ambient noise, large open structures",
    },
}

# OSM landuse → urban class mapping
LANDUSE_CLASS_MAP = {
    # Open field
    "grass": "OPEN_FIELD",
    "meadow": "OPEN_FIELD",
    "recreation_ground": "OPEN_FIELD",
    "village_green": "OPEN_FIELD",
    "cemetery": "OPEN_FIELD",
    # Residential
    "residential": "RESIDENTIAL",
    "allotments": "RESIDENTIAL",
    # Commercial / mixed
    "commercial": "MIXED_COMMERCIAL",
    "retail": "MIXED_COMMERCIAL",
    "mixed": "MIXED_COMMERCIAL",
    "market": "MIXED_COMMERCIAL",
    # Industrial
    "industrial": "INDUSTRIAL",
    "warehouse": "INDUSTRIAL",
    "port": "INDUSTRIAL",
    # Dense urban (non-residential built-up)
    "civic": "DENSE_URBAN",
    "institutional": "DENSE_URBAN",
    "education": "DENSE_URBAN",
}

# OSM highway types → noise weight (for noise floor estimation)
ROAD_NOISE_WEIGHT = {
    "motorway": 1.0,
    "trunk": 0.9,
    "primary": 0.8,
    "secondary": 0.65,
    "tertiary": 0.5,
    "residential": 0.3,
    "service": 0.2,
    "footway": 0.0,
    "path": 0.0,
}


def _buffer_poi(poi_row: pd.Series, radius_m: float = 200) -> object:
    """Create a metric-CRS buffer around a POI centroid."""
    point = Point(poi_row["lon"], poi_row["lat"])
    gdf_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    gdf_proj = gdf_point.to_crs(epsg=32748)
    buffered = gdf_proj.buffer(radius_m)
    return buffered.to_crs(epsg=4326).iloc[0]


def _building_density(poi_buffer, buildings_gdf: gpd.GeoDataFrame) -> float:
    """
    Estimate building density within buffer.
    Returns: ratio of building footprint area to total buffer area (0–1)
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return 0.0

    try:
        # Clip buildings to buffer
        buf_gdf = gpd.GeoDataFrame(geometry=[poi_buffer], crs="EPSG:4326")
        clipped = gpd.clip(buildings_gdf, buf_gdf)

        if len(clipped) == 0:
            return 0.0

        # Project to UTM for area calculation
        clipped_proj = clipped.to_crs(epsg=32748)
        buf_proj = buf_gdf.to_crs(epsg=32748)

        building_area = clipped_proj.geometry.area.sum()
        buffer_area = buf_proj.geometry.area.iloc[0]

        return float(np.clip(building_area / buffer_area, 0, 1))

    except Exception:
        return 0.0


def _dominant_landuse(poi_buffer, landuse_gdf: gpd.GeoDataFrame) -> str | None:
    """Get dominant landuse type within buffer."""
    if landuse_gdf is None or len(landuse_gdf) == 0:
        return None

    try:
        buf_gdf = gpd.GeoDataFrame(geometry=[poi_buffer], crs="EPSG:4326")
        clipped = gpd.clip(landuse_gdf, buf_gdf)

        if len(clipped) == 0 or "landuse" not in clipped.columns:
            return None

        clipped_proj = clipped.to_crs(epsg=32748)
        clipped_proj["area"] = clipped_proj.geometry.area

        # Get landuse with largest area
        dominant = clipped_proj.groupby("landuse")["area"].sum().idxmax()
        return str(dominant)

    except Exception:
        return None


def _road_noise_score(poi_buffer, roads_gdf: gpd.GeoDataFrame) -> float:
    """
    Estimate road-based noise contribution within buffer.
    Returns: weighted score 0–1
    """
    if roads_gdf is None or len(roads_gdf) == 0:
        return 0.0

    try:
        buf_gdf = gpd.GeoDataFrame(geometry=[poi_buffer], crs="EPSG:4326")
        clipped = gpd.clip(roads_gdf, buf_gdf)

        if len(clipped) == 0 or "highway" not in clipped.columns:
            return 0.0

        weights = clipped["highway"].map(ROAD_NOISE_WEIGHT).fillna(0.2)
        return float(np.clip(weights.mean(), 0, 1))

    except Exception:
        return 0.0


def classify_single(
    poi_row: pd.Series,
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    landuse_gdf: gpd.GeoDataFrame,
    buffer_radius_m: float = 200,
) -> dict:
    """
    Classify urban context for a single POI.

    Returns dict with:
        urban_class, class_id, building_density,
        road_noise_score, dominant_landuse, acoustic_profile
    """
    poi_buffer = _buffer_poi(poi_row, buffer_radius_m)

    building_density = _building_density(poi_buffer, buildings_gdf)
    road_noise_score = _road_noise_score(poi_buffer, roads_gdf)
    dominant_landuse = _dominant_landuse(poi_buffer, landuse_gdf)

    # ── Rule-based classification ────────────────────────────────────────────

    # Priority 1: explicit landuse tag
    if dominant_landuse and dominant_landuse in LANDUSE_CLASS_MAP:
        urban_class = LANDUSE_CLASS_MAP[dominant_landuse]

    # Priority 2: industrial override — only if VERY dominant (>60% buffer area)
    # Prevents single nearby warehouse from tagging a whole neighborhood
    elif dominant_landuse == "industrial" and building_density < 0.15:
        urban_class = "INDUSTRIAL"

    # Priority 3: building density + road noise rules
    elif building_density >= 0.50:
        urban_class = "DENSE_URBAN"
    elif building_density >= 0.25:
        if road_noise_score >= 0.55:
            urban_class = "MIXED_COMMERCIAL"
        else:
            urban_class = "RESIDENTIAL"
    elif building_density >= 0.08:
        urban_class = "RESIDENTIAL"
    else:
        urban_class = "OPEN_FIELD"

    # Priority 4: road noise override
    # Heavy arterial road nearby → bump RESIDENTIAL → MIXED_COMMERCIAL
    if road_noise_score >= 0.80 and urban_class == "RESIDENTIAL":
        urban_class = "MIXED_COMMERCIAL"

    # Priority 5: zero road score sanity check
    # If no roads detected but high building density → likely enclosed compound
    # Keep class as-is but flag it
    no_road_detected = road_noise_score == 0.0

    class_id = {v: k for k, v in URBAN_CLASSES.items()}[urban_class]
    acoustic_profile = CLASS_ACOUSTIC_PROFILE[urban_class]

    return {
        "urban_class": urban_class,
        "class_id": class_id,
        "building_density": round(building_density, 3),
        "road_noise_score": round(road_noise_score, 3),
        "dominant_landuse": dominant_landuse,
        "noise_floor_dB": acoustic_profile["noise_floor_dB"],
        "rt60_est_s": acoustic_profile["rt60_est_s"],
        "reflection_factor": acoustic_profile["reflection_factor"],
        "enclosed_compound": no_road_detected and building_density > 0.15,
    }

def classify_all(
    poi_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    landuse_gdf: gpd.GeoDataFrame,
    buffer_radius_m: float = 200,
    save: bool = True,
    area_name: str = "unknown",
) -> gpd.GeoDataFrame:
    """
    Classify urban context for all POIs in a GeoDataFrame.
    Shows progress every 50 rows.
    """
    print(f"\n🏙️  SONA Classifier")
    print(f"   POIs to classify : {len(poi_gdf)}")
    print(f"   Buffer radius    : {buffer_radius_m}m")
    print(f"   Approach         : rule-based (v1)\n")

    results = []
    for i, (_, row) in enumerate(poi_gdf.iterrows()):
        if i % 50 == 0:
            print(f"   Processing {i+1}/{len(poi_gdf)}...")
        result = classify_single(row, buildings_gdf, roads_gdf, landuse_gdf, buffer_radius_m)
        results.append(result)

    results_df = pd.DataFrame(results)
    classified = poi_gdf.copy().reset_index(drop=True)
    classified = pd.concat([classified, results_df], axis=1)

    print(f"\n✅ Classification complete")
    _print_class_distribution(classified)

    if save:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"classified_{area_name}.geojson"
        classified.to_file(out_path, driver="GeoJSON")
        print(f"💾 Saved : {out_path}")

    return classified


def _print_class_distribution(gdf: gpd.GeoDataFrame) -> None:
    print("\n── Urban Class Distribution ──────────────────")
    dist = gdf["urban_class"].value_counts()
    total = len(gdf)
    for cls, count in dist.items():
        bar = "█" * int(count / total * 30)
        print(f"   {cls:<20} {count:>4}  {bar}")
    print("──────────────────────────────────────────────\n")


if __name__ == "__main__":
    from fetcher import fetch_poi, fetch_urban_context

    area = "jakarta_pusat"

    # Load or fetch data
    poi_gdf     = fetch_poi(area, poi_type="mosque", save=False)
    context     = fetch_urban_context(area, save=False)

    buildings   = context.get("buildings")
    roads       = context.get("roads")
    landuse     = context.get("landuse")

    # Classify — test dengan 20 POI dulu agar cepat
    sample      = poi_gdf.head(20)
    classified  = classify_all(sample, buildings, roads, landuse, area_name=area)

    print(classified[["name", "urban_class", "building_density",
                       "road_noise_score", "noise_floor_dB"]].to_string())