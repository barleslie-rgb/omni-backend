# Omni TouristOS — Destination Engine v2

This package introduces a local-catalog-first destination architecture while preserving the existing Flutter `/api/v1/explore-city` and `/api/v1/place-details` contracts.

## What changed

- `main.py` mounts `services.destination_engine.router` before the legacy destination routes, so the new endpoints own:
  - `POST /api/v1/explore-city`
  - `POST /api/v1/place-details`
  - `GET /api/v1/place-photo`
  - `GET /api/v1/destination-health`
  - `GET /api/v1/destination-catalog`
- The destination path does **not** use Overpass or Wikipedia GeoSearch.
- Curated destinations are served from `data/destinations_arch/destination_catalog.json`.
- Unknown destinations fall back to OpenStreetMap Nominatim with serialized requests and caching.
- Images are non-critical and use Openverse as a free image source when available.
- Nearby hotels are a free geographic directory from Nominatim. They are **not** live room rates.
- Groq guidance is optional. The destination screen still returns useful deterministic details when Groq is unavailable.
- Google Places is optional enrichment only. A Google 403 cannot block the destination screen.

## Files to copy into the repository

Copy all files while preserving these paths:

- `main.py` → repository root, replacing the current `main.py`
- `services/__init__.py`
- `services/destination_engine.py`
- `data/destinations_arch/destination_catalog.json`
- `destinatin_catalogue_view.dart` → replace the current destination screen file if this is the version used by the Flutter project

## Render

No new paid service is required and no new environment secret is required for the core destination flow.

Keep the existing Render build/start commands. Deploy the files and then test:

`GET /api/v1/destination-health`

`GET /api/v1/destination-catalog`

Then test the existing Flutter request:

`POST /api/v1/explore-city`

The response should contain `destination_engine.overpass_used: false` and, for a curated city such as Mumbai or Dubai, `places_state: VERIFIED` with real catalog place records.

## Important

The local catalog is intentionally conservative about live facts. It does not hard-code volatile opening hours or ticket prices. Use official venue sources for current hours, fees, closures and access conditions.
