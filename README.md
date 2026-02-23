# 🕌 MIMBAR
### Mosque Integrated Mapping for Building Acoustic Recommendations

MIMBAR is an open-source GIS-based tool for analyzing and recommending 
sound system specifications for mosques and musholas in Indonesia.

Given an area, MIMBAR pulls mosque locations from OpenStreetMap, 
classifies the urban environment, estimates acoustic conditions, 
and recommends appropriate speaker specifications — so every adzan 
reaches its intended area clearly, without noise pollution.

## Features (Roadmap)
- [ ] OSM mosque/mushola data fetcher
- [ ] Urban area classifier (residential, commercial, open field, etc.)
- [ ] Noise floor estimator from road network
- [ ] Acoustic propagation estimator
- [ ] Speaker spec recommender
- [ ] Interactive map dashboard (Streamlit)
- [ ] Sound energy distribution visualization

## Pilot Area
Jakarta / Jabodetabek

## Tech Stack
Python 3.12 · osmnx · geopandas · folium · streamlit · scipy

## Setup
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```