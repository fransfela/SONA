"""
SONA - app.py
Spatial Outdoor Noise & Acoustics
Lightweight display dashboard — loads pre-processed GeoJSON only.
Run run_pipeline.py first to generate data.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SONA — Spatial Outdoor Noise & Acoustics",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "OPEN_FIELD":       "#2ecc71",
    "RESIDENTIAL":      "#3498db",
    "MIXED_COMMERCIAL": "#f39c12",
    "DENSE_URBAN":      "#e74c3c",
    "INDUSTRIAL":       "#9b59b6",
}

CLASS_EMOJI = {
    "OPEN_FIELD":       "🌿",
    "RESIDENTIAL":      "🏘️",
    "MIXED_COMMERCIAL": "🏪",
    "DENSE_URBAN":      "🏙️",
    "INDUSTRIAL":       "🏭",
}

AREA_CENTERS = {
    "jakarta_pusat":   (-6.2088, 106.8456),
    "jakarta_selatan": (-6.2615, 106.8106),
    "jakarta_utara":   (-6.1384, 106.8134),
    "depok":           (-6.4025, 106.7942),
    "bekasi":          (-6.2383, 106.9756),
    "tangerang":       (-6.1783, 106.6319),
}

# ── Load available results ───────────────────────────────────────────────────
processed_dir = Path("data/processed")
available = sorted([f.stem.replace("recommendations_", "")
                    for f in processed_dir.glob("recommendations_*.geojson")])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔊 SONA")
    st.markdown("*Spatial Outdoor Noise & Acoustics*")
    st.divider()

    if not available:
        st.error("No processed data found.\nRun `run_pipeline.py` first.")
        st.stop()

    st.markdown("### 📍 Dataset")
    selected = st.selectbox(
        "Area",
        options=available,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.divider()
    st.markdown("### 🎛️ Map Style")
    map_tile = st.selectbox("Basemap", ["CartoDB dark_matter", "CartoDB positron", "OpenStreetMap"])

    show_coverage = st.checkbox("Show coverage circles", value=False,
                                help="Draw estimated coverage radius per POI")

    st.divider()
    st.caption("Data generated offline via `run_pipeline.py`")
    st.caption("Built by [@fransfela](https://github.com/fransfela)")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_result(area: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    path = processed_dir / f"recommendations_{area}.geojson"
    gdf  = gpd.read_file(path)
    df   = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    return gdf, df

gdf, df = load_result(selected)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔊 SONA — Spatial Outdoor Noise & Acoustics</p>',
            unsafe_allow_html=True)
st.caption(f"Showing: **{selected.replace('_', ' ').title()}** · {len(df)} POIs · Rule-based classifier v1")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📊 Analytics", "🔊 Recommendations"])

# ── TAB 1: Map ───────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total POIs",    len(df))
    c2.metric("Avg Power",     f"{df['total_power_w'].mean():.0f} W")
    c3.metric("WHO Compliant", f"{int(df['who_compliant'].sum())} / {len(df)}")
    c4.metric("Avg Coverage",  f"{df['coverage_radius_m'].mean():.0f} m")
    st.divider()

    center = AREA_CENTERS.get(selected, (-6.2088, 106.8456))
    m = folium.Map(location=center, zoom_start=14, tiles=map_tile)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:rgba(0,0,0,0.8); padding:12px 16px;
                border-radius:8px; color:white; font-size:12px;">
    <b>Urban Class</b><br>"""
    for cls, color in CLASS_COLORS.items():
        legend_html += f'<span style="color:{color}">●</span> {CLASS_EMOJI[cls]} {cls}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in gdf.iterrows():
        urban_class = str(row.get("urban_class", "RESIDENTIAL"))
        color       = CLASS_COLORS.get(urban_class, "#3498db")
        name        = row.get("name", "Unknown") or "Unknown"
        emoji       = CLASS_EMOJI.get(urban_class, "📍")
        compliant   = "✅" if row.get("who_compliant") else "⚠️"
        lat, lon    = float(row["lat"]), float(row["lon"])

        popup_html = f"""
        <div style="font-family:sans-serif; min-width:260px;">
            <h4 style="margin:0 0 8px 0; color:{color};">{emoji} {name}</h4>
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <tr><td><b>Urban Class</b></td><td>{urban_class}</td></tr>
                <tr><td><b>Noise Floor</b></td><td>{row.get('noise_floor_dB','–')} dB</td></tr>
                <tr><td><b>Building Density</b></td><td>{float(row.get('building_density',0)):.1%}</td></tr>
                <tr><td><b>RT60 estimate</b></td><td>{row.get('rt60_s','–')} s</td></tr>
                <tr style="background:#f0f0f0"><td colspan="2"><b>🔊 Recommendation</b></td></tr>
                <tr><td><b>Total Power</b></td><td>{row.get('total_power_w','–')} W</td></tr>
                <tr><td><b>Speakers</b></td><td>{row.get('speaker_count','–')}× {row.get('unit_power_w','–')}W {row.get('speaker_type','–')}</td></tr>
                <tr><td><b>Config</b></td><td>{row.get('speaker_config','–')}</td></tr>
                <tr><td><b>Coverage radius</b></td><td>{row.get('coverage_radius_m','–')} m</td></tr>
                <tr><td><b>H / V Angle</b></td><td>{row.get('h_angle_deg','–')}° / {row.get('v_angle_deg','–')}°</td></tr>
                <tr><td><b>WHO Compliant</b></td><td>{compliant}</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{name} | {urban_class} | {row.get('total_power_w','?')}W",
        ).add_to(m)

        if show_coverage:
            folium.Circle(
                location=[lat, lon],
                radius=float(row.get("coverage_radius_m", 100)),
                color=color,
                fill=True,
                fill_opacity=0.05,
                weight=1,
            ).add_to(m)

    st_folium(m, width="100%", height=560, returned_objects=[])

# ── TAB 2: Analytics ─────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        class_counts = df["urban_class"].value_counts().reset_index()
        class_counts.columns = ["Urban Class", "Count"]
        fig1 = px.bar(class_counts, x="Urban Class", y="Count",
                      color="Urban Class", color_discrete_map=CLASS_COLORS,
                      title="Urban Class Distribution", text="Count")
        fig1.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.box(df, x="urban_class", y="total_power_w",
                      color="urban_class", color_discrete_map=CLASS_COLORS,
                      title="Recommended Power by Urban Class (W)",
                      labels={"urban_class": "Urban Class", "total_power_w": "Power (W)"})
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.scatter(df, x="building_density", y="noise_floor_dB",
                          color="urban_class", color_discrete_map=CLASS_COLORS,
                          size="total_power_w", size_max=15,
                          title="Building Density vs Noise Floor",
                          hover_data=["name", "total_power_w"],
                          labels={"building_density": "Building Density",
                                  "noise_floor_dB": "Noise Floor (dB)"})
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        compliant_n = int(df["who_compliant"].sum())
        fig4 = px.pie(
            values=[compliant_n, len(df) - compliant_n],
            names=["WHO Compliant ✅", "Exceeds Limit ⚠️"],
            title="WHO Noise Compliance",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: Recommendations table ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🔊 Full Acoustic Recommendations")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_class = st.multiselect(
            "Filter by Urban Class",
            options=df["urban_class"].unique().tolist(),
            default=df["urban_class"].unique().tolist(),
        )
    with col_f2:
        filter_compliant = st.selectbox(
            "WHO Compliance", ["All", "Compliant only", "Non-compliant only"]
        )
    with col_f3:
        sort_by = st.selectbox(
            "Sort by",
            ["total_power_w", "coverage_radius_m", "noise_floor_dB", "building_density"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    filtered = df[df["urban_class"].isin(filter_class)]
    if filter_compliant == "Compliant only":
        filtered = filtered[filtered["who_compliant"] == True]
    elif filter_compliant == "Non-compliant only":
        filtered = filtered[filtered["who_compliant"] == False]
    filtered = filtered.sort_values(sort_by, ascending=False)

    display_cols = ["name", "urban_class", "noise_floor_dB", "total_power_w",
                    "speaker_count", "unit_power_w", "speaker_type", "speaker_config",
                    "coverage_radius_m", "h_angle_deg", "v_angle_deg",
                    "who_compliant", "enclosed_compound"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def row_color(row):
        color = CLASS_COLORS.get(row["urban_class"], "#ffffff")
        return [f"background-color: {color}18"] * len(row)

    st.dataframe(
        filtered[display_cols].style.apply(row_color, axis=1),
        use_container_width=True,
        height=480,
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} POIs")

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv,
                       f"sona_{selected}.csv", "text/csv")