from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None


router = APIRouter(tags=["Destination Engine"])

BASE_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = BASE_DIR / "data" / "destinations_arch" / "destination_catalog.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPENVERSE_URL = "https://api.openverse.org/v1/images/"

USER_AGENT = os.environ.get(
    "DESTINATION_USER_AGENT",
    "OmniTouristOS/2.0 (destination explorer; contact via app)",
).strip()

# Public-service protection: one Nominatim request at a time with >=1.05s spacing.
_nominatim_lock = asyncio.Lock()
_nominatim_last_request = 0.0
_openverse_sem = asyncio.Semaphore(3)

_destination_cache: Dict[str, Dict[str, Any]] = {}
_places_cache: Dict[str, List[Dict[str, Any]]] = {}
_hotels_cache: Dict[str, List[Dict[str, Any]]] = {}
_image_cache: Dict[str, Dict[str, Any]] = {}


def _load_catalog() -> Dict[str, Any]:
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[Destination Catalog Notice] Could not load catalog: {exc}")
        return {}


CATALOG = _load_catalog()


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _destination_key(city: str, state: str, country: str) -> str:
    return "|".join((_norm(city), _norm(state), _norm(country)))


def _country_currency(country: str) -> Tuple[str, str]:
    c = _norm(country)
    mapping = {
        "india": ("INR", "₹"), "in": ("INR", "₹"),
        "united arab emirates": ("AED", "د.إ"), "uae": ("AED", "د.إ"), "ae": ("AED", "د.إ"),
        "singapore": ("SGD", "S$"), "sg": ("SGD", "S$"),
        "united states": ("USD", "$"), "usa": ("USD", "$"), "us": ("USD", "$"),
        "united kingdom": ("GBP", "£"), "uk": ("GBP", "£"), "gb": ("GBP", "£"),
        "thailand": ("THB", "฿"), "th": ("THB", "฿"),
        "japan": ("JPY", "¥"), "jp": ("JPY", "¥"),
        "malaysia": ("MYR", "RM"), "my": ("MYR", "RM"),
        "indonesia": ("IDR", "Rp"), "id": ("IDR", "Rp"),
        "australia": ("AUD", "A$"), "au": ("AUD", "A$"),
        "canada": ("CAD", "C$"), "ca": ("CAD", "C$"),
        "germany": ("EUR", "€"), "france": ("EUR", "€"), "italy": ("EUR", "€"),
        "spain": ("EUR", "€"), "portugal": ("EUR", "€"), "netherlands": ("EUR", "€"),
    }
    return mapping.get(c, ("USD", "$"))


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return r * 2 * math.asin(min(1.0, math.sqrt(a)))


def _category_for(text: str) -> str:
    t = _norm(text)
    if any(k in t for k in ("temple", "church", "mosque", "shrine", "spiritual", "gurdwara")):
        return "Sacred & Spiritual"
    if any(k in t for k in ("beach", "marine", "waterfall", "national park", "park", "garden", "nature", "zoo", "wildlife", "island")):
        return "Nature & Wildlife"
    if any(k in t for k in ("museum", "gallery", "arts", "theatre", "theater")):
        return "Arts & Culture"
    if any(k in t for k in ("mall", "bazaar", "market", "food", "culinary", "restaurant", "street food")):
        return "Culinary & Bazaars"
    if any(k in t for k in ("fort", "palace", "monument", "heritage", "historic", "tomb", "gate")):
        return "Heritage & Forts"
    if any(k in t for k in ("aquarium", "theme park", "amusement", "universal studios", "family")):
        return "Family & Attractions"
    return "Sights & Landmarks"


def _map_url(name: str, lat: float, lng: float) -> str:
    return (
        "https://www.google.com/maps/search/?api=1&query="
        + urllib.parse.quote(f"{name} {lat},{lng}")
    )


def _catalog_destination(city: str, state: str, country: str) -> Optional[Dict[str, Any]]:
    candidates = [
        _destination_key(city, state, country),
        _destination_key(city, "", country),
    ]
    for key in candidates:
        item = CATALOG.get("destinations", {}).get(key)
        if isinstance(item, dict):
            destination = dict(item.get("destination") or {})
            destination.setdefault("city", city)
            destination.setdefault("state", state)
            destination.setdefault("country", country)
            destination.setdefault("display_name", city)
            destination.setdefault("source", "Omni TouristOS Local Destination Catalog")
            destination.setdefault("data_state", "VERIFIED_LOCAL_CATALOG")
            destination.setdefault("open_data", False)
            return destination
    return None


def _catalog_places(city: str, state: str, country: str) -> List[Dict[str, Any]]:
    candidates = [_destination_key(city, state, country), _destination_key(city, "", country)]
    record = None
    for key in candidates:
        maybe = CATALOG.get("destinations", {}).get(key)
        if isinstance(maybe, dict):
            record = maybe
            break
    if not record:
        return []

    dest = record.get("destination") or {}
    origin_lat = float(dest.get("latitude"))
    origin_lng = float(dest.get("longitude"))
    output: List[Dict[str, Any]] = []
    for raw in record.get("places") or []:
        if not isinstance(raw, dict):
            continue
        lat = raw.get("lat")
        lng = raw.get("lng")
        name = str(raw.get("name") or "").strip()
        if not name or lat is None or lng is None:
            continue
        lat = float(lat)
        lng = float(lng)
        distance = _distance_km(origin_lat, origin_lng, lat, lng)
        item = {
            "name": name,
            "category": raw.get("category") or _category_for(name),
            "distance": f"{distance:.1f} km from Center",
            "distance_km": round(distance, 2),
            "distance_type": "straight_line",
            "timing": raw.get("timing") or "See official source",
            "entry": raw.get("entry") or "See official source",
            "lat": lat,
            "lng": lng,
            "address": raw.get("address") or "",
            "history": raw.get("history") or "Verified local destination-catalog record.",
            "overview": raw.get("overview") or "A notable place in this destination.",
            "best_food": raw.get("best_food") or [],
            "things_to_do": raw.get("things_to_do") or [],
            "best_time": raw.get("best_time") or "Check local conditions and official opening information before visiting.",
            "warnings": raw.get("warnings") or "Check official access, opening hours and local conditions before visiting.",
            "rating": raw.get("rating"),
            "reviews": raw.get("reviews"),
            "images": list(raw.get("images") or []),
            "google_photo_names": [],
            "image_provider": raw.get("image_provider") or ("Local catalog" if raw.get("images") else "NONE"),
            "image_credits": list(raw.get("image_credits") or []),
            "maps_url": raw.get("maps_url") or _map_url(name, lat, lng),
            "website_url": raw.get("website_url"),
            "place_id": raw.get("place_id") or f"catalog:{_norm(country)}:{_norm(city)}:{_norm(name)}",
            "source": "Omni TouristOS Local Destination Catalog",
            "data_state": "VERIFIED_LOCAL_CATALOG",
            "provider_verified": True,
            "attribution_required": raw.get("attribution_required") or "Omni TouristOS Local Destination Catalog",
        }
        output.append(item)
    output.sort(key=lambda x: (x.get("distance_km", 999999), x.get("name", "")))
    return output


async def _nominatim_get(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    global _nominatim_last_request
    async with _nominatim_lock:
        wait = 1.05 - (time.monotonic() - _nominatim_last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0)) as client:
            response = await client.get(
                NOMINATIM_URL,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
        _nominatim_last_request = time.monotonic()

    if response.status_code != 200:
        print(f"[Nominatim Notice] {response.status_code}: {response.text[:250]}")
        return []
    try:
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except Exception as exc:
        print(f"[Nominatim Notice] Invalid JSON: {exc}")
        return []


async def _resolve_with_nominatim(city: str, state: str, country: str) -> Optional[Dict[str, Any]]:
    key = _destination_key(city, state, country)
    cached = _destination_cache.get(key)
    if cached:
        return cached

    query = ", ".join(x for x in (city, state, country) if str(x).strip())
    results = await _nominatim_get({
        "q": query,
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": "5",
        "accept-language": "en",
    })
    if not results:
        return None

    city_norm = _norm(city)
    country_norm = _norm(country)

    def score(item: Dict[str, Any]) -> int:
        display = _norm(item.get("display_name"))
        item_name = _norm(item.get("name"))
        item_type = _norm(item.get("type"))
        address = item.get("address") or {}
        score_value = 0
        if item_name == city_norm:
            score_value += 120
        if city_norm and city_norm in display:
            score_value += 35
        if item_type in {"city", "town", "municipality", "village", "locality"}:
            score_value += 60
        if country_norm and country_norm in display:
            score_value += 20
        if _norm(address.get("city")) == city_norm:
            score_value += 20
        return score_value

    chosen = max(results, key=score)
    lat = chosen.get("lat")
    lon = chosen.get("lon")
    if lat is None or lon is None:
        return None

    resolved = {
        "city": city,
        "state": state,
        "country": country,
        "display_name": chosen.get("name") or city,
        "place_id": f"osm:{chosen.get('osm_type','')}/{chosen.get('osm_id','')}",
        "latitude": float(lat),
        "longitude": float(lon),
        "formatted_address": chosen.get("display_name"),
        "google_maps_url": _map_url(city, float(lat), float(lon)),
        "website_url": None,
        "types": [str(chosen.get("type") or "")],
        "address_components": chosen.get("address") or {},
        "source": "OpenStreetMap Nominatim",
        "data_state": "VERIFIED_OPEN_DATA",
        "open_data": True,
    }
    _destination_cache[key] = resolved
    return resolved


async def _discover_places_with_nominatim(city: str, country: str, state: str, destination: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = _destination_key(city, state, country)
    if key in _places_cache:
        return _places_cache[key]

    lat = float(destination["latitude"])
    lng = float(destination["longitude"])
    queries = [
        "tourist attraction",
        "historic landmark",
        "museum",
        "park",
        "beach",
        "temple",
        "church",
        "mosque",
    ]

    raw_results: List[Dict[str, Any]] = []
    for term in queries:
        results = await _nominatim_get({
            "q": f"{term} in {city}, {country}",
            "format": "jsonv2",
            "addressdetails": "1",
            "limit": "8",
            "accept-language": "en",
        })
        raw_results.extend(results)

    seen: set[str] = set()
    places: List[Dict[str, Any]] = []
    city_norm = _norm(city)
    for item in raw_results:
        name = str(item.get("name") or "").strip()
        item_type = str(item.get("type") or "").strip()
        if not name or _norm(name) == city_norm:
            continue
        lat_i = item.get("lat")
        lon_i = item.get("lon")
        if lat_i is None or lon_i is None:
            continue
        lat_i = float(lat_i)
        lon_i = float(lon_i)
        distance = _distance_km(lat, lng, lat_i, lon_i)
        if distance > 50:
            continue
        dedup_key = f"{_norm(name)}|{round(lat_i,4)}|{round(lon_i,4)}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        address = item.get("display_name") or ""
        category = _category_for(f"{name} {item_type} {address}")
        website = None
        extratags = item.get("extratags") or {}
        if isinstance(extratags, dict):
            website = extratags.get("website") or extratags.get("contact:website")
        places.append({
            "name": name,
            "category": category,
            "distance": f"{distance:.1f} km from Center",
            "distance_km": round(distance, 2),
            "distance_type": "straight_line",
            "timing": "See official source",
            "entry": "See official source",
            "lat": lat_i,
            "lng": lon_i,
            "address": address,
            "history": "Verified OpenStreetMap place record.",
            "overview": f"{name} is a mapped point of interest in {city}.",
            "best_food": [],
            "things_to_do": [],
            "best_time": "Check official information and local conditions before visiting.",
            "warnings": "Use official sources for opening hours, access, fees and local safety conditions.",
            "rating": None,
            "reviews": None,
            "images": [],
            "google_photo_names": [],
            "image_provider": "NONE",
            "image_credits": [],
            "maps_url": _map_url(name, lat_i, lon_i),
            "website_url": website if isinstance(website, str) and website.startswith(("https://", "http://")) else None,
            "place_id": f"osm:{item.get('osm_type','')}/{item.get('osm_id','')}",
            "source": "OpenStreetMap Nominatim",
            "data_state": "VERIFIED_OPEN_DATA",
            "provider_verified": True,
            "attribution_required": "OpenStreetMap",
        })

    places.sort(key=lambda x: (x.get("distance_km", 999999), x.get("name", "")))
    places = places[:30]
    _places_cache[key] = places
    return places


async def _openverse_images(query: str, limit: int = 3) -> Dict[str, Any]:
    cache_key = _norm(query)
    cached = _image_cache.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Any] = {"images": [], "credits": []}
    try:
        async with _openverse_sem:
            async with httpx.AsyncClient(timeout=httpx.Timeout(7.0, connect=3.0)) as client:
                response = await client.get(
                    OPENVERSE_URL,
                    params={"q": query[:200], "page_size": max(1, min(limit, 5)), "page": 1},
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Openverse Image Notice] {response.status_code}: {response.text[:250]}")
            _image_cache[cache_key] = result
            return result
        data = response.json() or {}
        for item in data.get("results") or []:
            thumb = str(item.get("thumbnail") or item.get("url") or "").strip()
            if not thumb.startswith(("https://", "http://")):
                continue
            result["images"].append(thumb)
            result["credits"].append({
                "title": str(item.get("title") or query),
                "source_url": str(item.get("foreign_landing_url") or item.get("detail_url") or "https://openverse.org/"),
                "artist": str(item.get("creator") or "").strip(),
                "license": str(item.get("license") or "").strip(),
                "provider": "Openverse",
                "attribution": str(item.get("attribution") or "").strip(),
            })
            if len(result["images"]) >= limit:
                break
    except Exception as exc:
        print(f"[Openverse Image Notice] {exc}")

    _image_cache[cache_key] = result
    return result


async def _enrich_images(places: List[Dict[str, Any]], city: str, country: str, limit_places: int = 8) -> None:
    targets = [p for p in places if not p.get("images")][: max(0, int(limit_places))]
    tasks = [_openverse_images(f"{p.get('name','')} {city} {country}", 2) for p in targets]
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for place, result in zip(targets, results):
        if isinstance(result, Exception) or not isinstance(result, dict):
            continue
        images = result.get("images") or []
        if images:
            place["images"] = images
            place["image_provider"] = "Openverse"
            place["image_credits"] = result.get("credits") or []
            place["attribution_required"] = "Openverse + original image provider license"


async def _discover_hotels(city: str, country: str, state: str, destination: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = _destination_key(city, state, country)
    if key in _hotels_cache:
        return _hotels_cache[key]

    results = await _nominatim_get({
        "q": f"hotel in {city}, {country}",
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": "12",
        "accept-language": "en",
    })
    center_lat = float(destination["latitude"])
    center_lng = float(destination["longitude"])
    hotels: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        name = str(item.get("name") or "").strip()
        lat = item.get("lat")
        lon = item.get("lon")
        if not name or lat is None or lon is None:
            continue

        # Nominatim's free text search can return non-hotel POIs whose names
        # happen to match the query (for example an island or district named
        # like a country). Accept only records that look like accommodation.
        item_class = _norm(item.get("class"))
        item_type = _norm(item.get("type"))
        address_text = _norm(item.get("display_name"))
        hotel_name_text = _norm(name)
        hotel_keywords = (
            "hotel", "resort", "inn", "motel", "hostel", "guest house",
            "guesthouse", "apartments", "apartment hotel", "lodge", "suites",
            "villa", "palace hotel"
        )
        accommodation_types = {"hotel", "motel", "hostel", "guest_house", "resort", "apartments"}
        looks_like_accommodation = (
            (item_class == "tourism" and item_type in accommodation_types)
            or any(k in hotel_name_text for k in hotel_keywords)
            or any(k in address_text for k in hotel_keywords)
        )
        if not looks_like_accommodation:
            continue

        lat_f, lon_f = float(lat), float(lon)
        dedup = f"{_norm(name)}|{round(lat_f,4)}|{round(lon_f,4)}"
        if dedup in seen:
            continue
        seen.add(dedup)
        d = _distance_km(center_lat, center_lng, lat_f, lon_f)
        # The destination screen asks for nearby hotels; do not surface remote
        # results simply because Nominatim returned them for the city query.
        if d > 25.0:
            continue
        address = str(item.get("display_name") or "")
        hotels.append({
            "name": name,
            "city": city,
            "distance": f"{d:.1f} km from Center",
            "distance_km": round(d, 2),
            "lat": lat_f,
            "lng": lon_f,
            "rating": None,
            "reviews": None,
            "price": None,
            "currency": None,
            "website_url": None,
            "booking_url": _map_url(name, lat_f, lon_f),
            "maps_url": _map_url(name, lat_f, lon_f),
            "address": address,
            "provider": "OpenStreetMap Nominatim",
            "data_state": "VERIFIED_OPEN_DATA",
            "live_rates": False,
            "source": "OpenStreetMap Nominatim",
        })
    hotels.sort(key=lambda x: x["distance_km"])
    _hotels_cache[key] = hotels[:12]
    return _hotels_cache[key]


def _groq_guidance(place: Dict[str, Any], city: str, country: str) -> Optional[Dict[str, Any]]:
    if Groq is None:
        return None
    key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        return None
    prompt = {
        "place": place.get("name"),
        "category": place.get("category"),
        "city": city,
        "country": country,
    }
    system = (
        "Return JSON with overview, best_time, things_to_do, best_food, family, group, solo, couple, "
        "safety_alerts, local_tips. This is travel guidance, not a claim of live facts. "
        "Never invent exact opening hours, ticket prices, closures, crime incidents or current alerts."
    )
    try:
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            model=os.environ.get("DESTINATION_TEXT_MODEL", "openai/gpt-oss-20b"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(prompt)}],
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
            timeout=20,
        )
        raw = completion.choices[0].message.content or ""
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[Destination AI Notice] {exc}")
        return None


def _as_list(value: Any, limit: int = 8) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(x).strip() for x in value[:limit] if str(x).strip()]
    return []


@router.get("/api/v1/destination-health")
async def destination_health():
    return {
        "status": "ok",
        "architecture": "local-catalog-first",
        "providers": {
            "local_catalog": "available" if CATALOG else "empty",
            "nominatim": "fallback",
            "openverse": "image-fallback",
            "groq": "optional-guidance",
            "google_places": "optional-enrichment",
            "overpass": "not-used",
            "wikipedia_geosearch": "not-used",
        },
        "catalog_destinations": len(CATALOG.get("destinations", {})),
    }


@router.post("/api/v1/explore-city")
async def explore_city_new(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON request: {exc}")

    city = str(body.get("city") or "").strip()
    state = str(body.get("state") or "").strip()
    country = str(body.get("country") or "").strip()
    traveler_country = str(body.get("traveler_country") or "IN").strip()
    traveler_currency = str(body.get("traveler_currency") or _country_currency(traveler_country)[0]).upper()
    start_date = str(body.get("start_date") or "").strip()
    return_date = str(body.get("return_date") or "").strip()

    if not city or not country:
        raise HTTPException(status_code=400, detail="city and country are required.")

    destination = _catalog_destination(city, state, country)
    source_mode = "LOCAL_CATALOG"
    if destination is None:
        destination = await _resolve_with_nominatim(city, state, country)
        source_mode = "NOMINATIM_FALLBACK"
    if destination is None:
        return {
            "status": "unavailable",
            "message": "This destination could not be resolved from the local catalog or the free geocoder.",
            "destination": None,
            "places": [],
            "landmarks": [],
            "hotels": [],
            "places_provider": source_mode,
            "places_state": "UNAVAILABLE",
        }

    if source_mode == "LOCAL_CATALOG":
        places = _catalog_places(city, state, country)
    else:
        places = await _discover_places_with_nominatim(city, country, state, destination)

    # Images are non-critical: failure never prevents attractions from loading.
    await _enrich_images(places, city, country, limit_places=8)

    hotels = await _discover_hotels(city, country, state, destination)
    destination_currency, destination_symbol = _country_currency(country)

    return {
        "status": "success",
        "destination": {
            **destination,
            "currency": destination_currency,
            "currency_symbol": destination_symbol,
            "display_currency": traveler_currency,
            "display_currency_symbol": _country_currency(traveler_country)[1],
        },
        "traveler_currency": traveler_currency,
        "traveler_currency_symbol": _country_currency(traveler_country)[1],
        "destination_currency": destination_currency,
        "destination_currency_symbol": destination_symbol,
        "start_date": start_date,
        "return_date": return_date,
        "adults": int(body.get("adults", 2)),
        "kids": int(body.get("kids", 0)),
        "child_ages": list(body.get("child_ages") or []),
        "places_provider": "Omni Local Catalog" if source_mode == "LOCAL_CATALOG" else "OpenStreetMap Nominatim",
        "places_state": "VERIFIED" if places else "EMPTY",
        "landmarks": places,
        "places": places,
        "hotels": hotels,
        "hotel_provider": "OpenStreetMap Nominatim" if hotels else None,
        "hotel_state": "directory" if hotels else "empty",
        "hotel_reason": "Free geographic hotel directory; live room rates/availability are not claimed.",
        "hotel_request_id": None,
        "attribution_required": ["OpenStreetMap", "Openverse"],
        "destination_engine": {
            "architecture": "local-catalog-first",
            "source_mode": source_mode,
            "overpass_used": False,
            "wikipedia_geosearch_used": False,
            "google_places_required": False,
        },
    }


@router.post("/api/v1/place-details")
async def place_details_new(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON request: {exc}")

    name = str(body.get("name") or "Verified place").strip()
    city = str(body.get("city") or "").strip()
    country = str(body.get("country") or "").strip()
    state = str(body.get("state") or "").strip()
    lat = body.get("lat")
    lng = body.get("lng")
    place_id = str(body.get("place_id") or "").strip()

    requested = {
        "name": name,
        "category": body.get("category"),
        "address": body.get("address"),
        "lat": float(lat) if lat is not None else None,
        "lng": float(lng) if lng is not None else None,
        "rating": body.get("rating"),
        "reviews": body.get("reviews"),
        "timing": body.get("timing"),
        "maps_url": body.get("maps_url"),
        "website_url": body.get("website_url"),
        "history": body.get("history"),
        "overview": body.get("overview"),
        "best_food": _as_list(body.get("best_food")),
        "things_to_do": _as_list(body.get("things_to_do")),
        "best_time": body.get("best_time"),
        "warnings": body.get("warnings"),
        "images": list(body.get("images") or []),
        "google_photo_names": [],
        "image_provider": body.get("image_provider"),
        "image_credits": list(body.get("image_credits") or []),
        "place_id": place_id,
    }

    # Prefer the catalog's richer record when this place belongs to a curated destination.
    # Only explicit request values override catalog fields; placeholder defaults must not
    # erase the curated overview, guidance, images, or attribution.
    catalog_places = _catalog_places(city, state, country) if city and country else []
    match = next((p for p in catalog_places if str(p.get("place_id")) == place_id), None)
    if match is None:
        match = next((p for p in catalog_places if _norm(p.get("name")) == _norm(name)), None)
    if match:
        place = dict(match)
        for key, value in requested.items():
            if key == "place_id":
                continue
            if value not in (None, "", [], {}):
                place[key] = value
        place.setdefault("name", name)
        place.setdefault("category", _category_for(name))
    else:
        place = {
            "name": name,
            "category": str(requested.get("category") or _category_for(name)),
            "address": str(requested.get("address") or ""),
            "lat": requested.get("lat"),
            "lng": requested.get("lng"),
            "rating": requested.get("rating"),
            "reviews": requested.get("reviews"),
            "timing": str(requested.get("timing") or "See official source"),
            "maps_url": requested.get("maps_url"),
            "website_url": requested.get("website_url"),
            "history": str(requested.get("history") or "Verified place record."),
            "overview": str(requested.get("overview") or "A notable place in the destination."),
            "best_food": _as_list(requested.get("best_food")),
            "things_to_do": _as_list(requested.get("things_to_do")),
            "best_time": str(requested.get("best_time") or "Check official information and local conditions before visiting."),
            "warnings": str(requested.get("warnings") or "Use official sources for current access, opening hours and local conditions."),
            "images": list(requested.get("images") or []),
            "google_photo_names": [],
            "image_provider": str(requested.get("image_provider") or "NONE"),
            "image_credits": list(requested.get("image_credits") or []),
            "place_id": place_id,
        }

    # Optional AI enrichment; never required for the endpoint to succeed.
    ai = _groq_guidance(place, city, country)
    if ai:
        place["overview"] = str(ai.get("overview") or place.get("overview") or "Verified place information.")
        place["best_time"] = str(ai.get("best_time") or place.get("best_time") or "Check local information before visiting.")
        place["things_to_do"] = _as_list(ai.get("things_to_do"), 8) or _as_list(place.get("things_to_do"), 8)
        place["best_food"] = _as_list(ai.get("best_food"), 8) or _as_list(place.get("best_food"), 8)
        place["family"] = str(ai.get("family") or "Choose activities appropriate for the group and venue rules.")
        place["group"] = str(ai.get("group") or "Allow extra time for groups, meeting points and queues.")
        place["solo"] = str(ai.get("solo") or "Keep valuables secure and use well-lit routes after dark.")
        place["couple"] = str(ai.get("couple") or "Consider quieter hours for a more relaxed visit.")
        place["safety_alerts"] = _as_list(ai.get("safety_alerts"), 8)
        place["local_tips"] = _as_list(ai.get("local_tips"), 8)
        intelligence_state = "AI-GUIDED"
    else:
        place.setdefault("family", "Suitable activities depend on mobility, venue rules and current conditions.")
        place.setdefault("group", "Groups should allow extra time for queues and meeting points.")
        place.setdefault("solo", "Keep valuables secure and use well-lit routes after dark.")
        place.setdefault("couple", "Consider quieter visiting hours for a more relaxed experience.")
        place.setdefault("safety_alerts", [])
        place.setdefault("local_tips", [])
        intelligence_state = "DETERMINISTIC_GUIDANCE"

    if not place.get("images"):
        image_data = await _openverse_images(f"{name} {city} {country}", 3)
        if image_data.get("images"):
            place["images"] = image_data["images"]
            place["image_provider"] = "Openverse"
            place["image_credits"] = image_data.get("credits") or []

    hotels = await _discover_hotels(city, country, state, {"latitude": place.get("lat") or 0, "longitude": place.get("lng") or 0}) if city and country and place.get("lat") is not None else []

    return {
        "status": "success",
        "place": {
            **place,
            "intelligence_state": intelligence_state,
            "data_state": place.get("data_state") or "VERIFIED_PROVIDER_FACTS_PLUS_GUIDANCE",
            "details_state": "VERIFIED_FACTS_PLUS_TRAVEL_GUIDANCE",
            "source": place.get("source") or "Omni TouristOS Destination Engine",
            "attribution_required": place.get("attribution_required") or "OpenStreetMap / Openverse where applicable",
            "visit_plans": [
                {"title": "2-hour visit", "steps": _as_list(place.get("things_to_do"), 3)},
                {"title": "Half-day visit", "steps": _as_list(place.get("things_to_do"), 5)},
                {"title": "Full-day visit", "steps": _as_list(place.get("things_to_do"), 8)},
            ],
        },
        "hotels": hotels,
        "hotel_state": "directory" if hotels else "empty",
        "hotel_provider": "OpenStreetMap Nominatim" if hotels else None,
        "hotel_reason": "Free geographic hotel directory; live availability requires a booking provider.",
        "attribution_required": ["OpenStreetMap", "Openverse"],
    }


@router.get("/api/v1/destination-catalog")
async def destination_catalog():
    destinations: List[Dict[str, Any]] = []
    for key, value in (CATALOG.get("destinations") or {}).items():
        if not isinstance(value, dict):
            continue
        d = value.get("destination") or {}
        destinations.append({
            "key": key,
            "city": d.get("city"),
            "state": d.get("state"),
            "country": d.get("country"),
            "place_count": len(value.get("places") or []),
        })
    return {"status": "success", "count": len(destinations), "destinations": destinations}


@router.get("/api/v1/place-photo")
async def destination_place_photo(
    url: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    max_width: int = Query(1200, ge=320, le=1600),
):
    """Compatibility image proxy.

    Supports the existing Google-photo query parameter (name) when a Google key
    is configured, while also supporting the new architecture's direct image URL.
    Direct URLs are allow-listed to reduce SSRF risk.
    """
    google_key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip().strip('"').strip("'")
    if name:
        clean_name = str(name).strip()
        if not google_key or not clean_name.startswith("places/") or "/photos/" not in clean_name:
            raise HTTPException(status_code=400, detail="Google photo resource is unavailable.")
        endpoint = "https://places.googleapis.com/v1/" + urllib.parse.quote(clean_name, safe="/") + "/media"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=True) as client:
                response = await client.get(
                    endpoint,
                    params={"maxWidthPx": int(max_width), "key": google_key},
                    headers={"User-Agent": USER_AGENT},
                )
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="Google photo could not be retrieved.")
            media_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
            if not media_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="Google did not return an image.")
            return Response(content=response.content, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Google photo service unavailable: {exc}")

    if not url:
        raise HTTPException(status_code=400, detail="url or name is required.")
    parsed = urllib.parse.urlparse(str(url))
    allowed_hosts = {
        "api.openverse.org",
        "openverse.org",
        "commons.wikimedia.org",
        "upload.wikimedia.org",
    }
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in allowed_hosts:
        raise HTTPException(status_code=400, detail="Image URL host is not allowed.")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0), follow_redirects=True) as client:
            response = await client.get(str(url), headers={"User-Agent": USER_AGENT})
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Image provider returned an error.")
        media_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
        if not media_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Provider URL did not return an image.")
        return Response(content=response.content, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image service unavailable: {exc}")
