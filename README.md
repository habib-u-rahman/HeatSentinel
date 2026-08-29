# HeatSentinel

Hyperlocal urban-heat intelligence platform — FortyGuard Hackathon '26.

Turns a per-block FortyGuard heat grid into decisions: which route to walk, which
vulnerable sites need warning first, and which street intervention would cool a
specific block the most.

**Live demo:** https://heatsentinel-frontend.vercel.app
**API:** https://heatsentinel-backend.onrender.com/api/health

## What it does

- **Heat grid** — per-block WBGT (heat-stress index, not just raw temperature),
  computed from a FortyGuard temperature grid joined with Open-Meteo weather
  (humidity, wind, solar radiation).
- **Cool routing** — Pareto-optimal walking routes trading off distance against
  heat exposure.
- **Vulnerability alerts** — proximity flagging for at-risk sites (schools,
  clinics, elder-care) sitting in high-risk heat cells.
- **Intervention prediction** — a Random Forest model trained on street-level
  surface composition (from a YOLOv8 + SegFormer vision pipeline over Mapillary
  imagery) predicts the °C cooling impact of specific interventions per block.
- **On-demand new-city builds** — point `/api/aoi/build` at any place name or
  bounding box and it runs the full pipeline (OSM street network, Mapillary
  imagery, CV analysis, POI lookup) live to stand up a new AOI.

Piloted on a ~4 km² area of Rawalpindi, Pakistan. FortyGuard's current coverage
doesn't include Rawalpindi, so the temperature layer falls back to clearly-labelled
synthetic data there — the integration itself is verified against real FortyGuard
responses (see `backend/app/fortyguard/client.py` and
`backend/scripts/fetch_fortyguard_grid.py`).

## Stack

FastAPI · osmnx/networkx · geopandas/shapely · scikit-learn · PyTorch/Transformers/Ultralytics
on the backend; React · Leaflet · Tailwind on the frontend.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in FORTYGUARD_API_KEY, GROQ_API_KEY, CITY_NAME, AOI_BBOX
```

```bash
cd frontend
npm install
copy .env.example .env
```

## Run tests

```bash
cd backend
pytest
```

## Run the API + frontend locally

```bash
cd backend
uvicorn app.main:app --reload
```

```bash
cd frontend
npm run dev
```

## Fetch a live FortyGuard grid

```bash
cd backend
python -m scripts.fetch_fortyguard_grid --bbox=<min_lon,min_lat,max_lon,max_lat>
```

Saves a live temperature grid snapshot that `app.api.deps.get_grid_for_timestamp`
picks up automatically (within a 3-hour freshness window) instead of the synthetic
fallback. Fails cleanly with a "no coverage" message for AOIs FortyGuard doesn't
currently cover.
