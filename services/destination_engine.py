from __future__ import annotations

import asyncio
import html
from email.utils import parsedate_to_datetime
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
from .shared_places_catalog import is_generic_place_name, search_shared_places

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None


router = APIRouter(tags=["Destination Engine"])
_shared_supabase_client = None


def set_supabase_client(client: Any) -> None:
    """Use the existing application Supabase client for shared places."""
    global _shared_supabase_client
    _shared_supabase_client = client

BASE_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = BASE_DIR / "data" / "destinations_arch" / "destination_catalog.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = (
    # This instance is listed as a global public Overpass provider; use the
    # official FOSSGIS instance once as a fallback if it is busy/unavailable.
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
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
_overpass_sem = asyncio.Semaphore(1)
_overpass_places_cache: Dict[str, Dict[str, Any]] = {}
_OVERPASS_CACHE_TTL = 6 * 60 * 60

_DEST_WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
_DEST_WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
_DEST_IMAGE_USER_AGENT = os.environ.get(
    "DESTINATION_IMAGE_USER_AGENT",
    "OmniTouristOS/2.0 (https://github.com/barleslie-rgb/omni-backend; destination photo search)",
).strip()
_dest_wikimedia_sem = asyncio.Semaphore(3)
_dest_wikipedia_sem = asyncio.Semaphore(3)
_dest_wikimedia_blocked = False
_dest_wikimedia_cooldown_until = 0.0
# Cache successful results only. Empty/error responses are deliberately not cached,
# so a provider outage cannot leave every place photo-less for the entire process.
_destination_fallback_image_cache: Dict[str, Dict[str, Any]] = {}
_DESTINATION_IMAGE_CACHE_TTL = 6 * 60 * 60


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


# Destination Hub is a sightseeing/attractions feed, not a directory of every
# nearby business. Keep these filters local to this router so Community Gems can
# continue to show shops, pharmacies, cafés and restaurants.
_DESTINATION_BUSINESS_NAME_RE = re.compile(
    r"\b(?:pharmacy|pharmacies|chemist|chemists|drugstore|drugstores|"
    r"salon|salons|barber|barbers|hair\s*cutting|hairdresser|hairdressers|beauty\s+salon|"
    r"supermarket|hypermarket|grocery|grocer|grocers|butcher|butchers|fishmonger|fishmongers|"
    r"restaurant|restaurants|cafe|cafes|coffee\s+day|coffee\s+shop|coffeehouse|"
    r"diner|seafood|pub|cinema|cinemas|multiplex|movie\s+theater|movie\s+theatre|"
    r"ticket|tickets|ticketing|travel\s+agency|tour\s+operator|"
    r"store|stores|shop|shops|kirana|convenience\s+store|retail|"
    r"clinic|hospital|dentist|doctor|bank|atm|fuel\s+station|car\s+wash)\b",
    re.IGNORECASE,
)
_DESTINATION_GENERIC_NAMES = {
    "temple", "mosque", "church", "shrine", "gurdwara", "mandir", "dargah",
    "park", "garden", "beach", "waterfall", "museum", "market", "informal market",
    "monument", "attraction", "place", "building", "landmark", "historic place",
    "restaurant", "cafe", "shop", "store", "supermarket", "pharmacy", "salon",
}
_DESTINATION_WEEKDAY_MARKET_RE = re.compile(
    r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+market$",
    re.IGNORECASE,
)
_DESTINATION_BLOCKED_SHARED_CATEGORIES = {
    "pharmacy chemist", "barber salon", "kirana essentials", "ice cream dairy",
    "cold storage meat", "indo chinese snacks", "chai quick bites", "bar restaurant",
    "diner seafood", "market bazaar mall", "movie cinema theater",
}
_DESTINATION_OSM_BUSINESS_CLASSES = {
    "shop", "office", "craft", "healthcare", "emergency", "financial",
    "industrial", "commercial", "power", "telecom",
}
_DESTINATION_OSM_BUSINESS_TYPES = {
    "cafe", "restaurant", "fast_food", "food_court", "bar", "pub", "nightclub",
    "pharmacy", "chemist", "clinic", "doctors", "dentist", "hospital", "bank", "atm",
    "fuel", "car_wash", "car_repair", "marketplace", "cinema", "theatre", "hairdresser",
    "beauty", "ice_cream", "butcher", "supermarket", "convenience", "department_store",
    "mall", "ticket", "ticket_office", "travel_agency", "travel_agent", "money_transfer",
    "mobile_phone", "electronics", "clothes", "variety_store", "general_store", "laundry",
}
_DESTINATION_ALLOWED_OSM_TYPES = {
    "tourism": {
        "attraction", "museum", "gallery", "artwork", "viewpoint", "theme_park",
        "zoo", "aquarium", "miniature_park", "planetarium", "wildlife_park",
    },
    "historic": {
        "archaeological_site", "castle", "city_gate", "fort", "manor", "memorial",
        "monument", "ruins", "tomb", "wayside_shrine", "battlefield", "fortification", "monastery", "church", "chapel", "wayside_cross", "citywalls", "fortified_manor",
    },
    "natural": {
        "beach", "waterfall", "peak", "cliff", "cave_entrance", "spring", "hot_spring",
        "water", "wood", "wetland", "bay", "cape", "volcano", "glacier", "island",
        "islet", "reef", "archipelago", "valley", "peninsula",
    },
    "leisure": {"park", "garden", "nature_reserve", "water_park", "bird_hide", "marina"},
    "waterway": {"river", "waterfall", "canal", "lake"},
    "man_made": {"lighthouse", "tower", "obelisk", "pier", "watermill", "windmill"},
    "boundary": {"protected_area"},
    "amenity": {"place_of_worship"},
}
_DESTINATION_OSM_KNOWN_CLASSES = set(_DESTINATION_ALLOWED_OSM_TYPES) | _DESTINATION_OSM_BUSINESS_CLASSES | {
    "amenity", "building", "railway", "aeroway", "public_transport", "landuse", "place",
    "boundary", "geological", "route", "waterway", "man_made", "military",
}


def _destination_category_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _is_destination_attraction_record(record: Dict[str, Any]) -> bool:
    """Reject general commercial/service listings from Destination Hub.

    This filter is deliberately used only by Destination Engine; Community Gems
    remains free to display local businesses and services.
    """
    if not isinstance(record, dict):
        return False
    name = re.sub(r"\s+", " ", str(record.get("name") or "")).strip()
    if not name:
        return False

    name_key = _destination_category_key(name)
    if name_key in _DESTINATION_GENERIC_NAMES or _DESTINATION_WEEKDAY_MARKET_RE.fullmatch(name_key):
        return False
    # Transport infrastructure is not a sightseeing attraction.
    if re.search(
        r"\b(?:metro|subway|railway(?:\s+station)?|railroad|train\s+station|rail\s+station|bus\s+stop|bus\s+station|bus\s+terminal|transit\s+station|tram\s+stop|tram\s+station|airport|terminal|taxi\s+stand|parking\s+lot|parking\s+garage|car\s+park)\b",
        name,
        flags=re.IGNORECASE,
    ):
        return False
    try:
        if is_generic_place_name(name):
            return False
    except Exception:
        pass

    # Community catalogue categories identify service/business records even
    # when their names do not contain the service type.
    category_value = record.get("category") or record.get("destination_type") or ""
    category_key = _destination_category_key(category_value)
    if category_key in _DESTINATION_BLOCKED_SHARED_CATEGORIES:
        return False

    # Reject business words in names. A named historic/visitor attraction is
    # still allowed through its explicitly recognized OSM attraction tags below
    # only when it is not itself a business listing (for example a local cinema).
    if _DESTINATION_BUSINESS_NAME_RE.search(name):
        return False
    if name_key in {"wednesday market", "informal market", "weekly market", "local market"}:
        return False

    tags = record.get("extratags") or {}
    if not isinstance(tags, dict):
        tags = {}
    for key in ("shop", "office", "craft", "healthcare"):
        if str(tags.get(key) or "").strip():
            return False

    osm_class = _norm(record.get("class") or (record.get("category") if _norm(record.get("category")) in _DESTINATION_OSM_KNOWN_CLASSES else ""))
    osm_type = _norm(record.get("type") or "").replace(" ", "_")
    amenity_tag = _norm(tags.get("amenity") or "").replace(" ", "_")
    if osm_class in _DESTINATION_OSM_BUSINESS_CLASSES:
        return False
    if amenity_tag in _DESTINATION_OSM_BUSINESS_TYPES or osm_type in _DESTINATION_OSM_BUSINESS_TYPES:
        return False

    # For Nominatim results with an OSM class/category, use an explicit
    # attraction-type allowlist. This prevents fuzzy text searches such as
    # "museum in city" from returning shops and cafés.
    if osm_class in _DESTINATION_OSM_KNOWN_CLASSES:
        allowed_types = _DESTINATION_ALLOWED_OSM_TYPES.get(osm_class, set())
        if osm_type not in allowed_types:
            return False
        # A worship place must be specifically named; generic "Temple" is not useful.
        return name_key not in _DESTINATION_GENERIC_NAMES

    # For normalized catalogue rows without raw OSM tags, accept named places
    # unless they look like commercial/service listings above.
    return True



def _deduplicate_destination_places(
    places: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one card per normalized name and merge duplicate image details."""
    output: List[Dict[str, Any]] = []
    index_by_name: Dict[str, int] = {}
    generic_categories = {"", "general", "sights and landmarks"}

    for candidate in places:
        if not isinstance(candidate, dict):
            continue

        name_key = _destination_category_key(candidate.get("name"))
        if not name_key:
            continue

        existing_index = index_by_name.get(name_key)
        if existing_index is None:
            index_by_name[name_key] = len(output)
            output.append(candidate)
            continue

        kept = output[existing_index]

        old_category = _destination_category_key(kept.get("category"))
        new_category = _destination_category_key(candidate.get("category"))
        if old_category in generic_categories and new_category not in generic_categories:
            kept["category"] = candidate.get("category")

        # Fill missing details, but do not overwrite populated fields.
        for field in (
            "address", "history", "overview", "maps_url", "website_url",
            "lat", "lng", "distance", "distance_km", "distance_type",
            "source", "data_state", "provider_verified", "attribution_required",
        ):
            if kept.get(field) in (None, "", [], {}):
                value = candidate.get(field)
                if value not in (None, "", [], {}):
                    kept[field] = value

        images = list(kept.get("images") or [])
        credits = list(kept.get("image_credits") or [])
        incoming_images = list(candidate.get("images") or [])
        incoming_credits = list(candidate.get("image_credits") or [])

        for image_index, image in enumerate(incoming_images):
            url = str(image or "").strip()
            if not url:
                continue

            credit = (
                incoming_credits[image_index]
                if image_index < len(incoming_credits)
                else {}
            )

            if url in images:
                existing_image_index = images.index(url)
                while len(credits) <= existing_image_index:
                    credits.append({})
                if not credits[existing_image_index] and credit:
                    credits[existing_image_index] = credit
                continue

            images.append(url)
            credits.append(credit or {})

        if images:
            kept["images"] = images
            kept["image_credits"] = credits
            if str(kept.get("image_provider") or "").upper() in {"", "NONE"}:
                kept["image_provider"] = candidate.get("image_provider") or "NONE"

    return output

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


def _destination_image_urls(value: Any) -> List[str]:
    """Normalize catalogue image fields without splitting a single URL into characters."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, dict):
        values = [value]
    else:
        return []

    output: List[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(
                item.get("url") or item.get("image_url") or item.get("thumbnail_url")
                or item.get("thumbnail") or item.get("src") or ""
            ).strip()
        else:
            continue
        if url.startswith(("https://", "http://")) and url not in seen:
            seen.add(url)
            output.append(url)
    return output


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
            "images": _destination_image_urls(raw.get("images") or raw.get("image_urls") or raw.get("image_url") or raw.get("photo_url") or raw.get("thumbnail_url")),
            "google_photo_names": [],
            "image_provider": raw.get("image_provider") or ("Local catalog" if _destination_image_urls(raw.get("images") or raw.get("image_urls") or raw.get("image_url") or raw.get("photo_url") or raw.get("thumbnail_url")) else "NONE"),
            "image_credits": raw.get("image_credits") if isinstance(raw.get("image_credits"), list) else [],
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

def _shared_catalog_places(
    city: str,
    state: str,
    country: str,
    destination: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert published shared-catalogue records to Destination Engine format."""
    client = _shared_supabase_client
    if not client:
        return []

    origin_lat = destination.get("latitude", destination.get("lat"))
    origin_lng = destination.get("longitude", destination.get("lng"))

    try:
        # Prefer exact city matches. If there are none, broaden the lookup and
        # use coordinates to include only nearby records (for example, Pune
        # results recorded under the neighbouring city Pimpri-Chinchwad).
        records = search_shared_places(
            client,
            city=city,
            query="",
            category="All",
            limit=100,
        )
        if not records:
            records = search_shared_places(
                client,
                city="",
                query="",
                category="All",
                limit=100,
            )
            nearby_fallback = True
        else:
            nearby_fallback = False
    except Exception as exc:
        print(f"[Destination Shared Catalogue Notice] {exc}")
        return []

    output: List[Dict[str, Any]] = []

    for record in records:
        record_country = str(record.get("country") or "").strip()
        record_state = str(record.get("state_region") or "").strip()
        record_city = str(record.get("city") or "").strip()

        if country and record_country and _norm(record_country) != _norm(country):
            continue
        if state and record_state and _norm(record_state) != _norm(state):
            continue

        name = str(record.get("name") or "").strip()
        if not name or not _is_destination_attraction_record(record):
            continue

        # Skip records that contain only a generic feature type rather than a
        # useful place name (for example, five OpenStreetMap entries named "Park").
        # Keep the published database rows untouched; filter them only from this feed.
        generic_names = {
            "park", "garden", "temple", "church", "mosque", "museum",
            "market", "mall", "hotel", "restaurant", "beach", "school",
            "hospital", "station", "monument", "attraction", "place", "building",
        }
        if _norm(name) in generic_names:
            continue

        lat = record.get("latitude")
        lng = record.get("longitude")
        distance = None

        try:
            if lat is not None and lng is not None:
                lat = float(lat)
                lng = float(lng)
                if origin_lat is not None and origin_lng is not None:
                    distance = _distance_km(
                        float(origin_lat),
                        float(origin_lng),
                        lat,
                        lng,
                    )
        except (TypeError, ValueError):
            lat = None
            lng = None
            distance = None

        # Broad fallback results must be geographically local. Do not let an
        # unrelated catalogue entry appear in a destination simply because
        # its city/state metadata is missing or uses a neighbouring city name.
        if nearby_fallback and _norm(record_city) != _norm(city):
            if distance is None or distance > 50:
                continue

        provider = str(record.get("source_provider") or "unknown")
        provider_id = str(record.get("source_place_id") or record.get("id") or "")
        image_url = record.get("image_url")
        image_attribution = record.get("image_attribution")
        image_license = record.get("image_license")
        image_source_url = record.get("image_source_url")

        item = {
            "name": name,
            "category": record.get("category") or _category_for(name),
            "city": record.get("city") or city,
            "state": record.get("state_region") or state,
            "country": record.get("country") or country,
            "country_code": record.get("country_code"),
            "description": record.get("description") or "",
            "distance": (
                f"{distance:.1f} km from Center"
                if distance is not None
                else "Distance unavailable"
            ),
            "distance_km": round(distance, 2) if distance is not None else None,
            "distance_type": "straight_line" if distance is not None else None,
            "timing": "See official source",
            "entry": "See official source",
            "lat": lat,
            "lng": lng,
            "address": record.get("address") or "",
            "history": record.get("description") or "Published shared-catalogue place.",
            "overview": record.get("description") or "Place listed in the shared catalogue.",
            "best_food": [],
            "things_to_do": [],
            "best_time": "Check official information before visiting.",
            "warnings": "Confirm current access and opening information before visiting.",
            "rating": None,
            "reviews": None,
            "images": [image_url] if image_url else [],
            "google_photo_names": [],
            "image_provider": provider if image_url else "NONE",
            "image_credits": [image_attribution] if image_attribution else [],
            "image_license": image_license,
            "image_source_url": image_source_url,
            "maps_url": _map_url(name, lat, lng) if lat is not None and lng is not None else None,
            "website_url": record.get("external_url"),
            "place_id": f"shared:{provider}:{provider_id}",
            "source_provider": provider,
            "source_place_id": provider_id,
            "source": "SHARED_CATALOG",
            "data_state": "PUBLISHED_SHARED_CATALOG_RECORD",
            "provider_verified": False,
            "attribution_required": image_attribution or f"Place source: {provider}",
        }
        output.append(item)

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


async def _discover_places_with_overpass(
    city: str,
    country: str,
    state: str,
    destination: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Discover named attractions from structured OSM tags, cached for six hours."""
    cache_key = _destination_key(city, state, country)
    now = time.monotonic()
    cached = _overpass_places_cache.get(cache_key)
    if isinstance(cached, dict) and now - float(cached.get("created_at") or 0.0) < _OVERPASS_CACHE_TTL:
        return [
            dict(place,
                 images=list(place.get("images") or []),
                 image_credits=list(place.get("image_credits") or []))
            for place in cached.get("places", [])
        ]

    try:
        lat = float(destination.get("latitude", destination.get("lat")))
        lng = float(destination.get("longitude", destination.get("lng")))
    except (TypeError, ValueError):
        return []

    radius_m = 35000
    # Use a bounding box for the database lookup (faster than repeated `around`
    # filters); the precise 35 km circular cut-off is still applied below.
    radius_km = radius_m / 1000.0
    lat_delta = radius_km / 110.574
    cos_lat = max(abs(math.cos(math.radians(lat))), 0.01)
    lng_delta = radius_km / (111.320 * cos_lat)
    south = max(-90.0, lat - lat_delta)
    north = min(90.0, lat + lat_delta)
    west = max(-180.0, lng - lng_delta)
    east = min(180.0, lng + lng_delta)
    query = f"""[out:json][timeout:20];
(
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["tourism"~"^(attraction|museum|gallery|artwork|viewpoint|theme_park|zoo|aquarium|picnic_site)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["historic"~"^(archaeological_site|castle|city_gate|fort|manor|memorial|monument|ruins|tomb|wayside_shrine|battlefield|fortification|church|chapel|monastery|wayside_cross|citywalls|fortified_manor)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["natural"~"^(beach|waterfall|peak|cliff|cave_entrance|spring|hot_spring|water|wood|wetland|bay|cape|island|islet|archipelago|valley|peninsula|reef)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["leisure"~"^(park|garden|nature_reserve|water_park|bird_hide|picnic_site)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["waterway"~"^(river|waterfall|canal|lake)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["man_made"~"^(lighthouse|obelisk|pier|watermill|windmill)$"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["boundary"="protected_area"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["amenity"="place_of_worship"];
  nwr({south:.6f},{west:.6f},{north:.6f},{east:.6f})["name"]["building"~"^(church|cathedral|chapel|temple|mosque|synagogue|religious|gurdwara)$"];
);
out center tags;
"""

    # Retry once against a second public instance if the first is busy or times out.
    # Failed lookups are not cached, so a transient outage will not poison future requests.
    elements: Optional[List[Dict[str, Any]]] = None
    last_error = ""
    for endpoint in OVERPASS_URLS:
        hostname = urllib.parse.urlparse(endpoint).hostname or endpoint
        try:
            async with _overpass_sem:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(24.0, connect=5.0),
                    follow_redirects=True,
                ) as client:
                    response = await client.post(
                        endpoint,
                        data={"data": query},
                        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                    )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code} from {hostname}"
                print(f"[Destination Overpass Notice] {last_error}; trying fallback if available.")
                continue
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
                last_error = f"invalid JSON payload from {hostname}"
                print(f"[Destination Overpass Notice] {last_error}; trying fallback if available.")
                continue
            elements = payload["elements"]
            print(f"[Destination Overpass] {hostname} returned {len(elements)} raw map features.")
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__} from {hostname}"
            print(f"[Destination Overpass Notice] {last_error}; trying fallback if available.")

    if elements is None:
        print(f"[Destination Overpass Notice] All configured instances failed ({last_error}); keeping existing results.")
        return []

    # Prefer the most informative records when the final list is capped.
    type_priority = (
        ("tourism", _DESTINATION_ALLOWED_OSM_TYPES.get("tourism", set())),
        ("historic", _DESTINATION_ALLOWED_OSM_TYPES.get("historic", set())),
        ("natural", _DESTINATION_ALLOWED_OSM_TYPES.get("natural", set())),
        ("leisure", _DESTINATION_ALLOWED_OSM_TYPES.get("leisure", set())),
        ("waterway", _DESTINATION_ALLOWED_OSM_TYPES.get("waterway", set())),
        ("man_made", _DESTINATION_ALLOWED_OSM_TYPES.get("man_made", set())),
        ("boundary", _DESTINATION_ALLOWED_OSM_TYPES.get("boundary", set())),
        ("amenity", _DESTINATION_ALLOWED_OSM_TYPES.get("amenity", set())),
    )
    worship_buildings = {
        "church", "cathedral", "chapel", "temple", "mosque",
        "synagogue", "religious", "gurdwara",
    }
    worship_category = {
        "christian": "church", "muslim": "mosque", "islam": "mosque",
        "hindu": "temple", "buddhist": "temple", "jain": "temple",
        "shinto": "temple", "sikh": "gurdwara", "jewish": "synagogue",
    }

    places: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for element in elements:
        if not isinstance(element, dict):
            continue
        tags = element.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        name = str(tags.get("name") or tags.get("name:en") or "").strip()
        if not name:
            continue

        primary_class = ""
        primary_type = ""
        for tag_key, allowed in type_priority:
            tag_value = str(tags.get(tag_key) or "").strip().lower()
            if tag_value and tag_value in allowed:
                primary_class, primary_type = tag_key, tag_value
                break

        # Some named churches/temples are mapped by building=* but lack
        # amenity=place_of_worship. Normalize only these explicit religious buildings.
        if not primary_class and str(tags.get("building") or "").strip().lower() in worship_buildings:
            primary_class, primary_type = "amenity", "place_of_worship"
        if not primary_class:
            continue

        filter_record = {
            "name": name,
            "class": primary_class,
            "type": primary_type,
            "category": primary_class,
            "extratags": tags,
        }
        if not _is_destination_attraction_record(filter_record):
            continue

        center = element.get("center") or {}
        lat_value = element.get("lat", center.get("lat"))
        lng_value = element.get("lon", center.get("lon"))
        if lat_value is None or lng_value is None:
            continue
        try:
            place_lat, place_lng = float(lat_value), float(lng_value)
        except (TypeError, ValueError):
            continue
        distance = _distance_km(lat, lng, place_lat, place_lng)
        if distance > 35.0:
            continue

        name_key = _destination_category_key(name)
        if not name_key or name_key in seen_names:
            continue
        seen_names.add(name_key)

        religion = _norm(tags.get("religion"))
        worship_hint = worship_category.get(religion, "")
        if primary_type == "place_of_worship":
            if not worship_hint:
                building_hint = _norm(tags.get("building"))
                worship_hint = "church" if building_hint in {"church", "cathedral", "chapel"} else building_hint
        category_text = " ".join(str(value or "") for value in (
            name, primary_class, primary_type, worship_hint,
            tags.get("historic"), tags.get("tourism"), tags.get("natural"),
            tags.get("leisure"), tags.get("waterway"), tags.get("building"),
        ))
        category = _category_for(category_text)

        address = str(tags.get("addr:full") or "").strip()
        if not address:
            address_parts = [
                tags.get("addr:housenumber"), tags.get("addr:street"),
                tags.get("addr:suburb"), tags.get("addr:city"),
            ]
            address = ", ".join(str(value).strip() for value in address_parts if str(value or "").strip())
        if not address:
            address = ", ".join(part for part in (city, state, country) if part)

        website = str(tags.get("website") or tags.get("contact:website") or "").strip()
        if not website.startswith(("https://", "http://")):
            website = None

        # This is a ranking aid only; it is removed before the API response.
        priority = 0
        if tags.get("wikipedia") or tags.get("wikidata"):
            priority += 4
        if tags.get("tourism") in {"attraction", "museum", "gallery", "viewpoint", "zoo", "aquarium", "theme_park"}:
            priority += 3
        if tags.get("historic"):
            priority += 3
        if any(tags.get(key) for key in ("natural", "leisure", "waterway", "boundary")):
            priority += 2
        if primary_type == "place_of_worship":
            priority += 2
        if website:
            priority += 1

        osm_kind = str(element.get("type") or "element")
        osm_id = str(element.get("id") or "")
        places.append({
            "name": name,
            "category": category,
            "distance": f"{distance:.1f} km from Center",
            "distance_km": round(distance, 2),
            "distance_type": "straight_line",
            "timing": str(tags.get("opening_hours") or "See official source"),
            "entry": "Free / see official source" if _norm(tags.get("fee")) in {"no", "0", "false"} else "See official source",
            "lat": place_lat,
            "lng": place_lng,
            "address": address,
            "history": f"Named OpenStreetMap feature ({primary_class}={primary_type}).",
            "overview": f"{name} is a mapped point of interest near {city}.",
            "best_food": [],
            "things_to_do": [],
            "best_time": "Check official information and local conditions before visiting.",
            "warnings": "Confirm current access, opening hours, fees and local safety conditions before visiting.",
            "rating": None,
            "reviews": None,
            "images": [],
            "google_photo_names": [],
            "image_provider": "NONE",
            "image_credits": [],
            "maps_url": _map_url(name, place_lat, place_lng),
            "website_url": website,
            "place_id": f"osm:{osm_kind}/{osm_id}",
            "source": "OpenStreetMap Overpass",
            "data_state": "VERIFIED_OPEN_DATA",
            "provider_verified": True,
            "attribution_required": "OpenStreetMap",
            "_destination_priority": priority,
        })

    places.sort(key=lambda place: (
        -int(place.get("_destination_priority") or 0),
        float(place.get("distance_km") or 999999),
        _norm(place.get("name")),
    ))
    # Store only successful HTTP/JSON responses, including a valid empty result,
    # so repeat searches do not hammer the public Overpass service.
    _overpass_places_cache[cache_key] = {
        "created_at": time.monotonic(),
        "places": [
            dict(place,
                 images=list(place.get("images") or []),
                 image_credits=list(place.get("image_credits") or []))
            for place in places
        ],
    }
    return places


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
            "extratags": "1",
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
        if not _is_destination_attraction_record(item):
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



# Canonical type labels make spelling/plural variants comparable in photo titles.
_DEST_IMAGE_TOKEN_ALIASES = {
    "temples": "temple", "mandir": "temple", "mandirs": "temple", "devasthan": "temple",
    "mosques": "mosque", "masjid": "mosque", "masjids": "mosque",
    "dargahs": "dargah", "shrines": "shrine", "tombs": "tomb",
    "churches": "church", "cathedrals": "cathedral", "gurudwara": "gurdwara",
    "forts": "fort", "fortress": "fort", "citadels": "citadel",
    "museums": "museum", "palaces": "palace", "gates": "gate", "gateways": "gateway",
    "beaches": "beach", "shore": "beach", "coast": "beach", "coastal": "beach",
    "waterfalls": "waterfall", "falls": "waterfall", "cascade": "waterfall", "cascades": "waterfall",
    "ghats": "ghat", "lakes": "lake", "rivers": "river", "dams": "dam",
    "parks": "park", "gardens": "garden", "sanctuaries": "sanctuary", "reserves": "reserve",
    "caves": "cave", "caverns": "cave", "monuments": "monument", "memorials": "memorial",
    "markets": "market", "bazaars": "bazaar", "malls": "mall",
    "supermarkets": "supermarket", "hypermarkets": "hypermarket",
    "pharmacies": "pharmacy", "chemists": "chemist", "drugstore": "pharmacy", "drugstores": "pharmacy",
    "salons": "salon", "barbers": "barber", "hairdresser": "salon", "hairdressers": "salon",
    "cafes": "cafe", "coffeehouse": "cafe", "coffeehouses": "cafe", "coffeeshop": "cafe",
    "restaurants": "restaurant", "diners": "restaurant", "eatery": "restaurant", "eateries": "restaurant",
    "hotels": "hotel", "resorts": "resort", "farmhouses": "farmhouse",
    "cinemas": "cinema", "theater": "theatre", "theaters": "theatre", "theatres": "theatre",
    "multiplex": "cinema", "multiplexes": "cinema",
    "stations": "station", "airports": "airport", "terminals": "terminal",
    "butchers": "butcher", "fishmongers": "fishmonger", "grocers": "grocer",
    "grocery": "grocer", "groceries": "grocer", "shops": "shop", "stores": "store",
}

_DEST_IMAGE_TYPE_GROUPS = (
    {"temple"}, {"mosque"}, {"dargah", "shrine", "tomb", "mausoleum"},
    {"church", "cathedral", "gurdwara", "synagogue"},
    {"fort", "citadel"}, {"museum"}, {"palace"}, {"gate", "gateway"},
    {"beach"}, {"waterfall"}, {"ghat"}, {"lake"}, {"river"}, {"dam"},
    {"park", "garden", "sanctuary", "reserve"}, {"cave"},
    {"monument", "memorial"}, {"market", "bazaar", "mall", "supermarket", "hypermarket", "grocer", "shop", "store"},
    {"pharmacy", "chemist"}, {"salon", "barber"}, {"cafe"},
    {"restaurant", "diner", "seafood"}, {"bar"},
    {"hotel", "resort", "farmhouse", "hostel", "motel"},
    {"cinema", "theatre"}, {"station"}, {"airport", "terminal"},
    {"butcher", "fishmonger"}, {"bank"}, {"clinic", "hospital"},
)

_DEST_IMAGE_BUSINESS_WORDS = {
    "market", "bazaar", "mall", "supermarket", "hypermarket", "grocer", "shop", "store",
    "pharmacy", "chemist", "salon", "barber", "cafe", "restaurant", "diner", "seafood",
    "bar", "hotel", "resort", "farmhouse", "hostel", "motel", "cinema", "theatre",
    "station", "airport", "terminal", "butcher", "fishmonger", "bank", "clinic", "hospital",
}
_DEST_IMAGE_GENERIC_WORDS = {
    "the", "and", "of", "in", "at", "near", "state", "district", "place", "places",
    "tourist", "tourism", "travel", "photo", "photos", "image", "images", "view", "views",
    "commons", "wikipedia", "official", "historic", "historical", "heritage", "india",
    "indian", "day", "local", "best", "famous",
}
# Single-name religious places often share a name across several cities. Require
# the requested city in the photo title for these, unless a better source exists.
_DEST_IMAGE_AMBIGUOUS_NAME_WORDS = {
    "sai", "ram", "hanuman", "shiva", "shiv", "ganesh", "ganesha", "krishna",
    "krishan", "durga", "kali", "lakshmi", "vishnu", "mahadev", "buddha",
}


def _destination_image_tokens(value: Any) -> List[str]:
    """Normalize a Commons/Wikipedia title without discarding relevant name words."""
    text = re.sub(r"\.[a-z0-9]{2,5}$", "", str(value or "").casefold())
    tokens = re.sub(r"[^a-z0-9]+", " ", text).strip().split()
    return [_DEST_IMAGE_TOKEN_ALIASES.get(token, token) for token in tokens]


def _destination_is_business_like(place_name: str, category: str = "") -> bool:
    name_words = set(_destination_image_tokens(place_name))
    category_text = " ".join(_destination_image_tokens(category))
    if name_words.intersection(_DEST_IMAGE_BUSINESS_WORDS):
        return True
    # Place categories produced from OSM sometimes carry type information that
    # is not present in the local display name.
    return any(word in category_text.split() for word in {
        "culinary", "bazaar", "shopping", "retail", "service", "medical",
        "food", "accommodation", "business",
    })


def _destination_title_has_city_context(candidate_title: str, city: str) -> bool:
    title_words = set(_destination_image_tokens(candidate_title))
    city_text = re.sub(r"[^a-z0-9]+", " ", str(city or "").casefold()).strip()
    if not city_text:
        return False
    # For hyphenated locality names, either locality component is accepted.
    original_parts = [
        part.strip() for part in re.split(r"[-/]", str(city or ""))
        if len(re.sub(r"[^a-z0-9]+", "", part.casefold())) >= 3
    ]
    if len(original_parts) > 1:
        return any(
            set(_destination_image_tokens(part)).issubset(title_words)
            for part in original_parts
        )
    city_tokens = _destination_image_tokens(city_text)
    return bool(city_tokens) and all(token in title_words for token in city_tokens)


def _destination_image_search_query(
    place_name: str, city: str = "", state: str = "", country: str = "", category: str = ""
) -> str:
    """Add locality terms when searching business or common-name places."""
    name_words = set(_destination_image_tokens(place_name))
    distinctive = name_words - _DEST_IMAGE_GENERIC_WORDS - set().union(*_DEST_IMAGE_TYPE_GROUPS)
    needs_locality = (
        _destination_is_business_like(place_name, category)
        or bool(name_words.intersection(_DEST_IMAGE_AMBIGUOUS_NAME_WORDS))
        or not distinctive
    )
    name = re.sub(r"\s+", " ", str(place_name or "")).strip()
    if not needs_locality:
        return name
    parts = [name, city, state, country]
    return " ".join(
        f'"{part.strip()}"' if index == 0 and part.strip() else part.strip()
        for index, part in enumerate(parts)
        if str(part or "").strip()
    )


def _destination_image_relevance(
    place_name: str,
    candidate_title: str,
    city: str = "",
    state: str = "",
    country: str = "",
    category: str = "",
) -> float:
    """Score an image title conservatively, including branch/location safeguards."""
    name_tokens = _destination_image_tokens(place_name)
    title_tokens_list = _destination_image_tokens(candidate_title.removeprefix("File:"))
    if not name_tokens or not title_tokens_list:
        return 0.0
    title_tokens = set(title_tokens_list)
    name_token_set = set(name_tokens)
    title_norm = " ".join(title_tokens_list)
    name_norm = " ".join(name_tokens)

    type_words = set().union(*_DEST_IMAGE_TYPE_GROUPS)
    # Generic-only place labels are not specific enough to justify attaching a photo.
    distinctive = {
        token for token in name_tokens
        if len(token) >= 3 and token not in _DEST_IMAGE_GENERIC_WORDS
        and token not in type_words
    }
    if not distinctive:
        return 0.0

    required_types = [
        group for group in _DEST_IMAGE_TYPE_GROUPS
        if name_token_set.intersection(group)
    ]
    for group in required_types:
        if not title_tokens.intersection(group):
            return 0.0

    matched = distinctive.intersection(title_tokens)
    if not matched:
        return 0.0
    if len(distinctive) == 1:
        required_matches = 1
    else:
        required_matches = min(len(distinctive), max(2, math.ceil(len(distinctive) * 0.60)))
    if len(matched) < required_matches:
        return 0.0

    # Local businesses/branches and frequently repeated religious names must
    # mention the requested city. A photo of the same chain in Prague or a
    # similarly named temple in another city must not be used.
    business_like = _destination_is_business_like(place_name, category)
    ambiguous_common_name = bool(name_token_set.intersection(_DEST_IMAGE_AMBIGUOUS_NAME_WORDS))
    city_match = _destination_title_has_city_context(candidate_title, city)
    if business_like and city.strip() and not city_match:
        return 0.0
    if ambiguous_common_name and city.strip() and not city_match:
        return 0.0

    # A single distinctive token with a generic feature type is fragile unless
    # the full normalized name appears in the title or the requested city does.
    full_name_in_title = name_norm in title_norm
    if len(distinctive) == 1 and not full_name_in_title and not city_match:
        return 0.0

    score = len(matched) / len(distinctive)
    if full_name_in_title:
        score = min(1.0, score + 0.05)
    if city_match:
        score = min(1.0, score + 0.05)
    return score



def _destination_image_cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _destination_fallback_image_cache.get(key)
    if not isinstance(entry, dict):
        return None
    created = float(entry.get("created_at") or 0.0)
    if time.monotonic() - created > _DESTINATION_IMAGE_CACHE_TTL:
        _destination_fallback_image_cache.pop(key, None)
        return None
    result = entry.get("result")
    return result if isinstance(result, dict) and result.get("images") else None


def _destination_image_cache_put(key: str, result: Dict[str, Any]) -> None:
    if result.get("images"):
        _destination_fallback_image_cache[key] = {
            "created_at": time.monotonic(),
            "result": result,
        }


async def _destination_commons_images(
    place_name: str,
    city: str = "",
    country: str = "",
    limit: int = 2,
    state: str = "",
    category: str = "",
) -> Dict[str, Any]:
    """Find likely place-specific files on Wikimedia Commons, with attribution."""
    global _dest_wikimedia_blocked, _dest_wikimedia_cooldown_until
    name = re.sub(r"\s+", " ", str(place_name or "")).strip()
    if not name:
        return {"images": [], "credits": []}
    search_query = _destination_image_search_query(name, city, state, country, category)
    cache_key = "commons|" + "|".join((
        name.casefold(), str(city or "").casefold(), str(state or "").casefold(),
        str(country or "").casefold(), str(category or "").casefold(),
    ))
    cached = _destination_image_cache_get(cache_key)
    if cached:
        return cached
    empty: Dict[str, Any] = {"images": [], "credits": []}
    if _dest_wikimedia_blocked or time.monotonic() < _dest_wikimedia_cooldown_until:
        return empty

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrnamespace": "6",
        "gsrlimit": "8",
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "1100",
    }
    try:
        async with _dest_wikimedia_sem:
            if _dest_wikimedia_blocked or time.monotonic() < _dest_wikimedia_cooldown_until:
                return empty
            async with httpx.AsyncClient(timeout=httpx.Timeout(4.5, connect=2.0), follow_redirects=True) as client:
                response = await client.get(
                    _DEST_WIKIMEDIA_API_URL,
                    params=params,
                    headers={"User-Agent": _DEST_IMAGE_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code == 403:
            _dest_wikimedia_blocked = True
            print("[Destination Images] Wikimedia Commons returned HTTP 403; pausing further Commons requests for this process.")
            return empty
        if response.status_code in (429, 503):
            retry_after = str(response.headers.get("Retry-After") or "").strip()
            delay = 60.0
            if retry_after:
                try:
                    delay = max(1.0, float(retry_after))
                except ValueError:
                    try:
                        from email.utils import parsedate_to_datetime
                        delay = max(1.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        delay = 60.0
            _dest_wikimedia_cooldown_until = time.monotonic() + delay
            print(f"[Destination Images] Wikimedia Commons returned HTTP {response.status_code}; cooling down for {int(delay)} seconds.")
            return empty
        if response.status_code != 200:
            print(f"[Destination Images] Wikimedia Commons HTTP {response.status_code} for {name!r}.")
            return empty

        payload = response.json() or {}
        pages = ((payload.get("query") or {}).get("pages") or {}).values()
        candidates = []
        seen = set()
        for page in pages:
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0] or {}
            mime = str(info.get("mime") or "").casefold()
            url = str(info.get("thumburl") or info.get("url") or "").strip()
            title = str(page.get("title") or "").strip()
            if not url.startswith("https://") or (mime and not mime.startswith("image/")) or url in seen:
                continue
            score = _destination_image_relevance(
                name, title.removeprefix("File:"), city, state, country, category
            )
            if score <= 0:
                continue
            seen.add(url)
            meta = info.get("extmetadata") or {}
            artist = html.unescape(re.sub(r"<[^>]*>", "", str((meta.get("Artist") or {}).get("value") or "")).strip())
            license_name = html.unescape(re.sub(r"<[^>]*>", "", str((meta.get("LicenseShortName") or {}).get("value") or "")).strip())
            file_page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="_:()")
            candidates.append((score, url, {
                "title": title.removeprefix("File:"),
                "source_url": file_page,
                "artist": artist,
                "license": license_name,
                "provider": "Wikimedia Commons",
            }))
        candidates.sort(key=lambda item: item[0], reverse=True)
        result = {"images": [], "credits": []}
        for _, url, credit in candidates[:max(1, min(int(limit), 3))]:
            result["images"].append(url)
            result["credits"].append(credit)
        _destination_image_cache_put(cache_key, result)
        return result
    except Exception as exc:
        print(f"[Destination Images] Commons lookup failed for {name!r}: {type(exc).__name__}.")
        return empty


async def _destination_wikipedia_images(
    place_name: str,
    city: str = "",
    country: str = "",
    limit: int = 1,
    state: str = "",
    category: str = "",
) -> Dict[str, Any]:
    """Use an English Wikipedia page thumbnail when it names the requested place."""
    name = re.sub(r"\s+", " ", str(place_name or "")).strip()
    if not name:
        return {"images": [], "credits": []}
    search_query = _destination_image_search_query(name, city, state, country, category)
    cache_key = "wikipedia|" + "|".join((
        name.casefold(), str(city or "").casefold(), str(state or "").casefold(),
        str(country or "").casefold(), str(category or "").casefold(),
    ))
    cached = _destination_image_cache_get(cache_key)
    if cached:
        return cached
    empty: Dict[str, Any] = {"images": [], "credits": []}
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrnamespace": "0",
        "gsrlimit": "5",
        "prop": "pageimages|info",
        "piprop": "thumbnail",
        "pithumbsize": "1100",
        "inprop": "url",
    }
    try:
        async with _dest_wikipedia_sem:
            async with httpx.AsyncClient(timeout=httpx.Timeout(4.5, connect=2.0), follow_redirects=True) as client:
                response = await client.get(
                    _DEST_WIKIPEDIA_API_URL,
                    params=params,
                    headers={"User-Agent": _DEST_IMAGE_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            return empty
        payload = response.json() or {}
        pages = ((payload.get("query") or {}).get("pages") or {}).values()
        candidates = []
        for page in pages:
            title = str(page.get("title") or "").strip()
            thumb = ((page.get("thumbnail") or {}).get("source") or "").strip()
            if not title or not thumb.startswith("https://"):
                continue
            score = _destination_image_relevance(
                name, title, city, state, country, category
            )
            if score <= 0:
                continue
            page_url = str(page.get("fullurl") or "").strip()
            if not page_url:
                page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="_:()")
            candidates.append((score, thumb, {
                "title": title,
                "source_url": page_url,
                "artist": "",
                "license": "See the linked Wikipedia page and original image file for attribution and license terms.",
                "provider": "Wikipedia page thumbnail",
            }))
        candidates.sort(key=lambda item: item[0], reverse=True)
        result = {"images": [], "credits": []}
        seen = set()
        for _, url, credit in candidates:
            if url in seen:
                continue
            seen.add(url)
            result["images"].append(url)
            result["credits"].append(credit)
            if len(result["images"]) >= max(1, min(int(limit), 2)):
                break
        _destination_image_cache_put(cache_key, result)
        return result
    except Exception as exc:
        print(f"[Destination Images] Wikipedia lookup failed for {name!r}: {type(exc).__name__}.")
        return empty


async def _openverse_images(query: str, limit: int = 3) -> Dict[str, Any]:
    """Search Openverse, caching only successful image results."""
    cache_key = _norm(query)
    cached = _image_cache.get(cache_key)
    if cached is not None and cached.get("images"):
        return cached

    result: Dict[str, Any] = {"images": [], "credits": []}
    try:
        async with _openverse_sem:
            async with httpx.AsyncClient(timeout=httpx.Timeout(4.5, connect=2.0), follow_redirects=True) as client:
                response = await client.get(
                    OPENVERSE_URL,
                    params={"q": query[:200], "page_size": max(1, min(limit, 5)), "page": 1},
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Openverse Image Notice] HTTP {response.status_code}; continuing with other destination photo providers.")
            return result
        data = response.json() or {}
        seen = set()
        for item in data.get("results") or []:
            thumb = str(item.get("thumbnail") or item.get("url") or "").strip()
            if not thumb.startswith("https://") or thumb in seen:
                continue
            seen.add(thumb)
            result["images"].append(thumb)
            result["credits"].append({
                "title": str(item.get("title") or "").strip(),
                "source_url": str(item.get("foreign_landing_url") or item.get("detail_url") or "https://openverse.org/"),
                "artist": str(item.get("creator") or "").strip(),
                "license": str(item.get("license") or "").strip(),
                "provider": "Openverse",
                "attribution": str(item.get("attribution") or "").strip(),
            })
            if len(result["images"]) >= max(1, min(limit, 3)):
                break
    except Exception as exc:
        print(f"[Openverse Image Notice] {type(exc).__name__}; continuing with other destination photo providers.")

    if result["images"]:
        _image_cache[cache_key] = result
    return result


def _destination_filter_images_by_title(
    place_name: str,
    result: Dict[str, Any],
    limit: int = 2,
    city: str = "",
    state: str = "",
    country: str = "",
    category: str = "",
) -> Dict[str, Any]:
    """Keep Openverse photos only when titles and locality support the requested place."""
    if not isinstance(result, dict):
        return {"images": [], "credits": []}
    raw_images = result.get("images") or []
    raw_credits = result.get("credits") or []
    images: List[str] = []
    credits: List[Dict[str, Any]] = []
    seen = set()
    for index, raw_url in enumerate(raw_images):
        url = str(raw_url or "").strip()
        if not url.startswith("https://") or url in seen:
            continue
        credit = raw_credits[index] if index < len(raw_credits) else {}
        if not isinstance(credit, dict):
            continue
        title = str(credit.get("title") or "").strip()
        # Missing titles are not evidence that a photo depicts the requested place.
        if not title or _destination_image_relevance(
            place_name, title, city, state, country, category
        ) <= 0:
            continue
        seen.add(url)
        images.append(url)
        credits.append(credit)
        if len(images) >= max(1, min(int(limit), 3)):
            break
    return {"images": images, "credits": credits}


async def _enrich_images(
    places: List[Dict[str, Any]],
    city: str,
    country: str,
    limit_places: int = 30,
    state: str = "",
) -> None:
    """Enrich blank destination cards without attaching cross-city business photos."""
    targets = [
        place for place in places if not place.get("images")
    ][:max(0, min(int(limit_places), 30))]
    if not targets:
        return

    async def enrich_one(place: Dict[str, Any]) -> None:
        name = str(place.get("name") or "").strip()
        if not name or place.get("images"):
            return
        category = str(place.get("category") or "")
        query = " ".join(
            part for part in (name, city, state, country) if str(part or "").strip()
        )

        # Commons often has the strongest file-level attribution. Searches for
        # local businesses/common names include the city, and titles are checked
        # again before any result is attached to a destination card.
        result = await _destination_commons_images(
            name, city, country, 2, state=state, category=category
        )
        images = result.get("images") or []
        if images:
            place["images"] = images
            place["image_provider"] = "Wikimedia Commons"
            place["image_credits"] = result.get("credits") or []
            place["attribution_required"] = "Wikimedia Commons; see image_credits for author and license"
            return

        # Wikipedia thumbnails are useful for well-known landmarks, while the
        # title/location check blocks unrelated branches and same-name shrines.
        result = await _destination_wikipedia_images(
            name, city, country, 1, state=state, category=category
        )
        images = result.get("images") or []
        if images:
            place["images"] = images
            place["image_provider"] = "Wikipedia page thumbnail"
            place["image_credits"] = result.get("credits") or []
            place["attribution_required"] = "Wikipedia page thumbnail; verify original file attribution/license"
            return

        # Openverse is a final fallback; accept only titled results that pass the
        # same place-type and locality relevance checks. Empty results aren't cached.
        result = await _openverse_images(query, 2)
        filtered_result = _destination_filter_images_by_title(
            name, result, 2, city=city, state=state, country=country, category=category
        )
        images = filtered_result.get("images") or []
        if images:
            place["images"] = images
            place["image_provider"] = "Openverse"
            place["image_credits"] = filtered_result.get("credits") or []
            place["attribution_required"] = "Openverse; see image_credits for the original provider license"

    try:
        # Keep the API response bounded; enrichment must never prevent place details loading.
        await asyncio.wait_for(
            asyncio.gather(*(enrich_one(place) for place in targets), return_exceptions=True),
            timeout=24.0,
        )
    except asyncio.TimeoutError:
        print("[Destination Images] Image enrichment reached the 24-second response budget; returning all images found so far.")


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
            "wikimedia_commons": "primary-image-fallback",
            "wikipedia": "last-resort-thumbnail",
            "groq": "optional-guidance",
            "google_places": "optional-enrichment",
            "overpass": "structured attraction discovery with six-hour cache",
            "wikipedia_geosearch": "not-used",
        },
        "catalog_destinations": len(CATALOG.get("destinations", {})),
        "image_fallback_version": "location-aware-v2-single-url-normalization-30-places",
        "destination_filter_version": "attractions-only-v2",
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
        places = await _discover_places_with_nominatim(
            city, country, state, destination
        )

    # Add structured attraction discovery even when a local catalogue entry exists.
    # The helper returns only named OSM attraction features and applies the same
    # business/transport exclusion filter used by the Nominatim results.
    overpass_places = await _discover_places_with_overpass(city, country, state, destination)
    if overpass_places:
        places.extend(overpass_places)
        print(f"[Destination Overpass] Added {len(overpass_places)} candidate attractions for {city}.")

    # Add published shared-catalogue records without replacing existing results.
    shared_places = _shared_catalog_places(
        city, state, country, destination
    )
    shared_added = 0

    for shared in shared_places:
        duplicate = False
        shared_name = _norm(shared.get("name"))

        if not shared_name:
            continue

        for existing in places:
            if _norm(existing.get("name")) != shared_name:
                continue

            try:
                existing_lat = existing.get("lat", existing.get("latitude"))
                existing_lng = existing.get("lng", existing.get("longitude"))
                shared_lat = shared.get("lat")
                shared_lng = shared.get("lng")

                if all(
                    value is not None
                    for value in (
                        existing_lat,
                        existing_lng,
                        shared_lat,
                        shared_lng,
                    )
                ):
                    duplicate = (
                        _distance_km(
                            float(existing_lat),
                            float(existing_lng),
                            float(shared_lat),
                            float(shared_lng),
                        )
                        <= 0.2
                    )
                else:
                    existing_address = _norm(existing.get("address"))
                    shared_address = _norm(shared.get("address"))
                    duplicate = bool(
                        existing_address
                        and existing_address == shared_address
                    )
            except (TypeError, ValueError):
                duplicate = False

            if duplicate:
                break

        if not duplicate:
            places.append(shared)
            shared_added += 1


    # Final destination-only filter covers local catalogue and shared-catalogue
    # additions too; community-service listings remain in Community Gems only.
    before_filter_count = len(places)
    places = [place for place in places if _is_destination_attraction_record(place)]
    removed_count = before_filter_count - len(places)
    if removed_count:
        print(f"[Destination Places] Filtered {removed_count} non-attraction business/service listing(s) for {city}.")

    before_dedup_count = len(places)
    places = _deduplicate_destination_places(places)
    duplicate_count = before_dedup_count - len(places)
    if duplicate_count:
        print(
            f"[Destination Places] Removed {duplicate_count} duplicate "
            f"place name(s) for {city}."
        )

    # Keep curated local-catalogue entries first; then prefer well-described OSM
    # attractions (Wikipedia/Wikidata, heritage/nature tags, and worship sites).
    places.sort(key=lambda place: (
        0 if str(place.get("source") or "") == "Omni TouristOS Local Destination Catalog" else 1,
        -int(place.get("_destination_priority") or 0),
        float(place.get("distance_km") or 999999),
        _norm(place.get("name")),
    ))
    places = places[:30]
    for place in places:
        place.pop("_destination_priority", None)

    # Images are non-critical: failure never prevents attractions from loading.
    await _enrich_images(places, city, country, limit_places=30, state=state)

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
        "places_provider": (
            "Omni Local Catalog" if source_mode == "LOCAL_CATALOG"
            else "OpenStreetMap Nominatim"
        ) + (" + OpenStreetMap Overpass" if overpass_places else "") + (" + Shared Places Catalog" if shared_added else ""),
        "places_state": "VERIFIED" if places else "EMPTY",
        "landmarks": places,
        "places": places,
        "hotels": hotels,
        "hotel_provider": "OpenStreetMap Nominatim" if hotels else None,
        "hotel_state": "directory" if hotels else "empty",
        "hotel_reason": "Free geographic hotel directory; live room rates/availability are not claimed.",
        "hotel_request_id": None,
        "attribution_required": ["OpenStreetMap", "Wikimedia Commons", "Wikipedia", "Openverse"],
        "destination_engine": {
            "architecture": "local-catalog-first",
            "source_mode": source_mode,
            "shared_catalog_places_added": shared_added,
            "overpass_used": bool(overpass_places),
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
        await _enrich_images([place], city, country, limit_places=1, state=state)

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
        "attribution_required": ["OpenStreetMap", "Wikimedia Commons", "Wikipedia", "Openverse"],
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
