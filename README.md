# 🔊 SONA
### Spatial Outdoor Noise & Acoustics

SONA is an open-source GIS-based urban acoustic planning tool.
Given a geographic area, SONA fetches points of interest from OpenStreetMap,
classifies the urban environment, estimates outdoor acoustic conditions,
and recommends appropriate sound system specifications.

## Use Cases
- Public address systems in dense urban areas
- Outdoor venue acoustic planning
- Urban noise impact assessment
- Sound system specification for community facilities

## Features (Roadmap)
- [ ] OSM data fetcher (points of interest + urban morphology)
- [ ] Urban area classifier (residential, commercial, open field, etc.)
- [ ] Noise floor estimator from road network
- [ ] Acoustic propagation estimator
- [ ] Speaker spec recommender
- [ ] Interactive map dashboard (Streamlit)
- [ ] Sound energy distribution visualization

## Pilot Area
Jakarta / Jabodetabek, Indonesia

## Tech Stack
Python 3.12 · osmnx · geopandas · folium · streamlit · scipy

## Setup
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```