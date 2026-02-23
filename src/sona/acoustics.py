"""
SONA - acoustics.py
Spatial Outdoor Noise & Acoustics
Compute acoustic recommendations per POI based on urban classification.

Output per POI:
    - Required SPL at boundary
    - Recommended amplifier power (Watts)
    - Number of speakers
    - Coverage angle recommendation
    - EQ suggestion per band
    - Neighbor overlap risk flag
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point


# ── Constants ────────────────────────────────────────────────────────────────

# SNR target above noise floor (dB) — speech intelligibility standard
SNR_TARGET_DB = 10

# Standard speaker sensitivity (dB SPL @ 1W, 1m) — typical horn speaker
DEFAULT_SENSITIVITY_DB = 96

# Max recommended SPL for residential areas (WHO guideline)
MAX_RESIDENTIAL_SPL_DB = 70

# EQ suggestions per urban class — (band, action, amount_dB, reason)
EQ_PROFILES = {
    "OPEN_FIELD": [
        ("80-200 Hz",   "cut",   3, "Reduce low-end buildup on ground reflection"),
        ("1-4 kHz",     "boost", 2, "Enhance speech presence and clarity"),
        ("6-12 kHz",    "cut",   2, "Reduce wind noise sensitivity"),
    ],
    "RESIDENTIAL": [
        ("125-250 Hz",  "cut",   4, "Reduce low-mid buildup between buildings"),
        ("500-1k Hz",   "flat",  0, "Maintain natural midrange"),
        ("2-5 kHz",     "boost", 2, "Speech intelligibility in moderate reflection"),
    ],
    "MIXED_COMMERCIAL": [
        ("80-160 Hz",   "cut",   6, "Heavy low-end absorption needed in busy area"),
        ("300-600 Hz",  "cut",   3, "Reduce muddy reflections from facades"),
        ("2-4 kHz",     "boost", 4, "Cut through ambient noise for speech clarity"),
        ("8-16 kHz",    "cut",   3, "Tame harshness from hard surface reflections"),
    ],
    "DENSE_URBAN": [
        ("63-125 Hz",   "cut",   8, "Aggressive low cut — urban canyon resonance"),
        ("250-500 Hz",  "cut",   4, "Reduce comb filtering between tall buildings"),
        ("1-3 kHz",     "boost", 5, "Maximum speech presence needed"),
        ("6-10 kHz",    "cut",   4, "Control sibilance harshness"),
    ],
    "INDUSTRIAL": [
        ("63-200 Hz",   "cut",   5, "Reduce machinery low-frequency masking"),
        ("400-800 Hz",  "cut",   2, "Reduce metallic resonance"),
        ("1-4 kHz",     "boost", 4, "Speech clarity over broadband noise"),
    ],
}

# Speaker coverage angle recommendation per urban class
COVERAGE_ANGLE_MAP = {
    "OPEN_FIELD":       {"h_angle": 90,  "v_angle": 40, "reason": "Wide coverage, no nearby reflectors"},
    "RESIDENTIAL":      {"h_angle": 60,  "v_angle": 30, "reason": "Moderate focus, avoid neighbor spill"},
    "MIXED_COMMERCIAL": {"h_angle": 45,  "v_angle": 25, "reason": "Focused beam, high ambient noise"},
    "DENSE_URBAN":      {"h_angle": 40,  "v_angle": 20, "reason": "Narrow beam, strong urban canyon"},
    "INDUSTRIAL":       {"h_angle": 60,  "v_angle": 30, "reason": "Directional to overcome broadband noise"},
}


# ── Core calculations ────────────────────────────────────────────────────────

def required_spl(noise_floor_db: float, snr_target: float = SNR_TARGET_DB) -> float:
    """Minimum SPL needed at listener position to achieve target SNR."""
    return noise_floor_db + snr_target


def spl_at_distance(power_w: float, sensitivity_db: float, distance_m: float) -> float:
    """
    SPL at given distance using inverse square law.
    SPL(d) = sensitivity + 10*log10(power) - 20*log10(distance)
    """
    if distance_m <= 0 or power_w <= 0:
        return 0.0
    return sensitivity_db + 10 * np.log10(power_w) - 20 * np.log10(distance_m)


def required_power(
    target_spl_db: float,
    distance_m: float,
    sensitivity_db: float = DEFAULT_SENSITIVITY_DB,
    reflection_factor: float = 0.0,
) -> float:
    """
    Calculate required amplifier power (Watts) to achieve target SPL at distance.
    reflection_factor reduces effective distance (urban reflections help propagation).
    """
    effective_distance = distance_m * (1 - reflection_factor * 0.3)
    effective_distance = max(effective_distance, 1.0)

    power_raw = 10 ** ((target_spl_db - sensitivity_db + 20 * np.log10(effective_distance)) / 10)
    return round(float(power_raw), 1)


def recommend_speaker_count(power_w: float, urban_class: str) -> dict:
    """
    Recommend number and type of speakers based on total power and urban class.
    Returns: {count, unit_power_w, type, config}
    """
    if urban_class in ["DENSE_URBAN", "MIXED_COMMERCIAL"]:
        # Multiple smaller directional speakers better than one big one
        if power_w <= 30:
            return {"count": 1, "unit_power_w": 30,  "type": "horn",     "config": "single directional"}
        elif power_w <= 60:
            return {"count": 2, "unit_power_w": 30,  "type": "horn",     "config": "stereo directional"}
        elif power_w <= 120:
            return {"count": 2, "unit_power_w": 60,  "type": "horn",     "config": "stereo high-power"}
        else:
            count = int(np.ceil(power_w / 60))
            return {"count": count, "unit_power_w": 60, "type": "horn",  "config": "distributed array"}

    else:  # OPEN_FIELD, RESIDENTIAL, INDUSTRIAL
        if power_w <= 25:
            return {"count": 1, "unit_power_w": 25,  "type": "column",   "config": "single column"}
        elif power_w <= 50:
            return {"count": 1, "unit_power_w": 50,  "type": "column",   "config": "single column"}
        elif power_w <= 100:
            return {"count": 2, "unit_power_w": 50,  "type": "column",   "config": "dual column"}
        else:
            count = int(np.ceil(power_w / 50))
            return {"count": count, "unit_power_w": 50, "type": "column","config": "distributed column"}


def estimate_coverage_radius(
    power_w: float,
    sensitivity_db: float,
    target_spl_db: float,
) -> float:
    """Estimate maximum radius (meters) where SPL >= target."""
    if power_w <= 0:
        return 0.0
    # Rearranged inverse square law
    radius = 10 ** ((sensitivity_db + 10 * np.log10(power_w) - target_spl_db) / 20)
    return round(float(radius), 1)


# ── Main recommendation engine ───────────────────────────────────────────────

def recommend_single(
    poi_row: pd.Series,
    coverage_radius_m: float = 150,
    sensitivity_db: float = DEFAULT_SENSITIVITY_DB,
) -> dict:
    """
    Generate full acoustic recommendation for a single POI.

    Parameters
    ----------
    poi_row           : row from classified GeoDataFrame
    coverage_radius_m : target coverage distance in meters
    sensitivity_db    : speaker sensitivity spec

    Returns
    -------
    dict with full acoustic recommendation
    """
    urban_class      = poi_row.get("urban_class", "RESIDENTIAL")
    noise_floor_db   = float(poi_row.get("noise_floor_dB", 55))
    rt60             = float(poi_row.get("rt60_est_s", 0.6))
    reflection_f     = float(poi_row.get("reflection_factor", 0.4))
    enclosed         = bool(poi_row.get("enclosed_compound", False))

    # Adjust noise floor if enclosed compound (sheltered = lower effective noise)
    if enclosed:
        noise_floor_db = max(noise_floor_db - 5, 40)

    req_spl          = required_spl(noise_floor_db)
    power_w          = required_power(req_spl, coverage_radius_m, sensitivity_db, reflection_f)

    # Cap power for residential areas (noise ordinance)
    if urban_class == "RESIDENTIAL":
        power_w = min(power_w, 50.0)

    speaker_rec      = recommend_speaker_count(power_w, urban_class)
    coverage_rec     = COVERAGE_ANGLE_MAP.get(urban_class, COVERAGE_ANGLE_MAP["RESIDENTIAL"])
    eq_rec           = EQ_PROFILES.get(urban_class, EQ_PROFILES["RESIDENTIAL"])
    max_radius       = estimate_coverage_radius(power_w, sensitivity_db, req_spl)

    # Compliance check
    boundary_spl     = spl_at_distance(power_w, sensitivity_db, coverage_radius_m)
    who_compliant    = boundary_spl <= MAX_RESIDENTIAL_SPL_DB

    return {
        # Input summary
        "urban_class":          urban_class,
        "noise_floor_dB":       noise_floor_db,
        "rt60_s":               rt60,
        # SPL targets
        "required_spl_dB":      round(req_spl, 1),
        "boundary_spl_dB":      round(boundary_spl, 1),
        "who_compliant":        who_compliant,
        # Power & speaker
        "total_power_w":        power_w,
        "speaker_count":        speaker_rec["count"],
        "unit_power_w":         speaker_rec["unit_power_w"],
        "speaker_type":         speaker_rec["type"],
        "speaker_config":       speaker_rec["config"],
        # Coverage
        "coverage_radius_m":    max_radius,
        "h_angle_deg":          coverage_rec["h_angle"],
        "v_angle_deg":          coverage_rec["v_angle"],
        # EQ
        "eq_profile":           eq_rec,
        # Flags
        "enclosed_compound":    enclosed,
    }

def recommend_all(
    classified_gdf: gpd.GeoDataFrame,
    coverage_radius_m: float = 150,
    save: bool = True,
    area_name: str = "unknown",
) -> gpd.GeoDataFrame:
    """Run acoustic recommendations for all POIs in classified GeoDataFrame."""
    print(f"\n🔊 SONA Acoustics Engine")
    print(f"   POIs            : {len(classified_gdf)}")
    print(f"   Coverage target : {coverage_radius_m}m\n")

    records = []
    for _, row in classified_gdf.iterrows():
        rec = recommend_single(row, coverage_radius_m)
        records.append(rec)

    rec_df = pd.DataFrame(records)
    eq_profiles = rec_df.pop("eq_profile")

    # Drop columns from rec_df that already exist in classified_gdf to avoid duplicates
    overlap_cols = [c for c in rec_df.columns if c in classified_gdf.columns]
    rec_df = rec_df.drop(columns=overlap_cols)

    result = classified_gdf.copy().reset_index(drop=True)
    result = pd.concat([result, rec_df], axis=1)
    result["eq_profile"] = eq_profiles

    print(f"✅ Recommendations generated for {len(result)} POIs")
    _print_recommendation_summary(result)

    if save:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        export = result.drop(columns=["eq_profile"])
        out_path = out_dir / f"recommendations_{area_name}.geojson"
        export.to_file(out_path, driver="GeoJSON")
        print(f"💾 Saved : {out_path}")

    return result


def _print_recommendation_summary(gdf: gpd.GeoDataFrame) -> None:
    print("\n── Acoustic Recommendation Summary ──────────────────")
    print(f"   Avg total power  : {gdf['total_power_w'].mean():.1f} W")
    print(f"   Avg coverage     : {gdf['coverage_radius_m'].mean():.1f} m")
    print(f"   WHO compliant    : {gdf['who_compliant'].sum()} / {len(gdf)}")
    print(f"   Enclosed compound: {int(gdf['enclosed_compound'].sum())} flagged")
    print("\n   Power by urban class:")
    # Flatten urban_class in case of multi-dimensional issue
    temp = gdf.copy()
    temp["urban_class"] = temp["urban_class"].astype(str)
    summary = temp.groupby("urban_class")["total_power_w"].agg(["mean", "min", "max"])
    print(summary.round(1).to_string())
    print("─────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src/sona")
    from fetcher import fetch_poi, fetch_urban_context
    from classifier import classify_all

    area = "jakarta_pusat"

    poi_gdf  = fetch_poi(area, poi_type="mosque", save=False)
    context  = fetch_urban_context(area, save=False)
    sample   = poi_gdf.head(20)

    classified = classify_all(
        sample,
        context.get("buildings"),
        context.get("roads"),
        context.get("landuse"),
        area_name=area,
        save=False,
    )

    result = recommend_all(classified, coverage_radius_m=150, area_name=area)

    cols = ["name", "urban_class", "total_power_w", "speaker_count",
            "speaker_type", "coverage_radius_m", "who_compliant"]
    print(result[cols].to_string())