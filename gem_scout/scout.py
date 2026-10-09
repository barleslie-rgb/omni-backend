from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import httpx

GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY", "").strip()
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
GEOAPIFY_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
GEOAPIFY_DETAILS_URL = "https://api.geoapify.com/v2/place-details"
OPEN_DATA_USER_AGENT = os.environ.get(
    "OPEN_DATA_USER_AGENT",
    "OmniTouristOS/1.0 (community-gem-scout; contact=admin@touristos.app)",
).strip()
NOMINATIM_API_URL = os.environ.get(
    "NOMINATIM_API_URL", "https://nominatim.openstreetmap.org/search"
).strip()
OVERPASS_API_URL = os.environ.get(
    "OVERPASS_API_URL", "https://overpass-api.de/api/interpreter"
).strip()

# Geoapify uses the OpenStreetMap-derived places dataset. These category keys
# are documented in its Places API taxonomy. Multiple categories can be sent
# in a single query.
GEOAPIFY_CATEGORIES: Dict[str, List[str]] = {
    "Pharmacy / Chemist": [
        "healthcare.pharmacy",
        "commercial.health_and_beauty.pharmacy",
        "commercial.chemist",
    ],
    "Barber & Salon": ["service.beauty", "service.beauty.hairdresser"],
    "Bar & Restaurant": ["catering.restaurant", "catering.bar", "catering.pub"],
    "Chai & Quick Bites": [
        "catering.cafe",
        "catering.fast_food",
        "catering.ice_cream",
        "commercial.food_and_drink.coffee_and_tea",
    ],
    "Food": [
        "catering.restaurant",
        "catering.cafe",
        "catering.fast_food",
        "catering.food_court",
        "catering.ice_cream",
    ],
    "Hotel & Stay": [
        "accommodation.hotel",
        "accommodation.guest_house",
        "accommodation.hostel",
        "accommodation.motel",
    ],
    "Market, Bazaar & Mall": ["commercial"],
    "Clothing & Fashion": ["commercial.clothing"],
    "Electronics & Mobile": ["commercial"],
    "Heritage & Sight": ["tourism.attraction", "tourism.sights", "heritage"],
    "Picnic Spot & Landscape": ["leisure.park", "leisure.garden"],
    "Clinic / Doctor": ["healthcare.clinic_or_praxis", "healthcare.hospital", "healthcare.pharmacy"],
    "Dental / Optical": ["healthcare.dentist", "commercial.health_and_beauty.optician"],
    "General Store / Supermarket": ["commercial.supermarket", "commercial.convenience", "commercial"],
    "Beauty & Spa": ["service.beauty"],
    "All": ["commercial", "catering", "accommodation", "healthcare", "tourism", "leisure", "entertainment"],
}

# The OSM fallback is intentionally secondary. Render previously could not
# reach Nominatim/Overpass reliably, so the main discovery path must not depend
# on it.
OSM_CATEGORY_QUERIES: Dict[str, str] = {
    "Pharmacy / Chemist": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["healthcare"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    "Barber & Salon": 'nwr(around:{radius},{lat},{lng})["shop"~"hairdresser|beauty",i];',
    "Bar & Restaurant": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|bar|pub|fast_food",i];',
    "Chai & Quick Bites": 'nwr(around:{radius},{lat},{lng})["amenity"~"cafe|fast_food|ice_cream",i];',
    "Food": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|cafe|fast_food|food_court|ice_cream",i];',
    "Hotel & Stay": 'nwr(around:{radius},{lat},{lng})["tourism"~"hotel|guest_house|hostel|motel",i];',
    "Market, Bazaar & Mall": 'nwr(around:{radius},{lat},{lng})["shop"];',
    "Clothing & Fashion": 'nwr(around:{radius},{lat},{lng})["shop"~"clothes|fashion|shoes",i];',
    "Electronics & Mobile": 'nwr(around:{radius},{lat},{lng})["shop"~"electronics|mobile_phone|computer",i];',
    "Heritage & Sight": 'nwr(around:{radius},{lat},{lng})["tourism"~"attraction|museum|viewpoint|zoo|theme_park",i];',
    "Picnic Spot & Landscape": 'nwr(around:{radius},{lat},{lng})["leisure"~"park|garden",i];',
}

CATEGORY_PROFILES: Dict[str, str] = {
    "pharmacy": "Pharmacy / Chemist",
    "chemist": "Pharmacy / Chemist",
    "medical store": "Pharmacy / Chemist",
    "pharmacy chemist": "Pharmacy / Chemist",
    "barber": "Barber & Salon",
    "salon": "Barber & Salon",
    "restaurant": "Bar & Restaurant",
    "bar": "Bar & Restaurant",
    "pub": "Bar & Restaurant",
    "cafe": "Chai & Quick Bites",
    "chai": "Chai & Quick Bites",
    "food": "Food",
    "hotel": "Hotel & Stay",
    "hotels": "Hotel & Stay",
    "accommodation": "Hotel & Stay",
    "mall": "Market, Bazaar & Mall",
    "shopping": "Market, Bazaar & Mall",
    "market": "Market, Bazaar & Mall",
    "clothing": "Clothing & Fashion",
    "fashion": "Clothing & Fashion",
    "electronics": "Electronics & Mobile",
    "attractions": "Heritage & Sight",
    "park": "Picnic Spot & Landscape",
    "parks": "Picnic Spot & Landscape",
    "doctor": "Clinic / Doctor",
    "clinic": "Clinic / Doctor",
    "dentist": "Dental / Optical",
    "optical": "Dental / Optical",
    "all": "All",
}

CANONICAL_TO_OSM_PROFILE: Dict[str, str] = {
    "Pharmacy / Chemist": "Pharmacy / Chemist",
    "Barber & Salon": "Barber & Salon",
    "Bar & Restaurant": "Bar & Restaurant",
    "Chai & Quick Bites": "Chai & Quick Bites",
    "Food": "Food",
    "Hotel & Stay": "Hotel & Stay",
    "Market, Bazaar & Mall": "Market, Bazaar & Mall",
    "Clothing & Fashion": "Clothing & Fashion",
    "Electronics & Mobile": "Electronics & Mobile",
    "Heritage & Sight": "Heritage & Sight",
    "Picnic Spot & Landscape": "Picnic Spot & Landscape",
}


def _normalize(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_category(category: str) -> str:
    raw = _normalize(category)
    for canonical in GEOAPIFY_CATEGORIES:
        if _normalize(canonical) == raw:
            return canonical
    if raw in CATEGORY_PROFILES:
        return CATEGORY_PROFILES[raw]
    for alias, canonical in CATEGORY_PROFILES.items():
        if alias and (alias in raw or raw in alias):
            return canonical
    if any(token in raw for token in ("pharmacy", "chemist", "medical store")):
        return "Pharmacy / Chemist"
    if any(token in raw for token in ("hotel", "stay", "resort", "hostel", "guest house")):
        return "Hotel & Stay"
    if any(token in raw for token in ("clothing", "dress", "fashion", "apparel")):
        return "Clothing & Fashion"
    if any(token in raw for token in ("mall", "shopping", "market", "supermarket")):
        return "Market, Bazaar & Mall"
    if any(token in raw for token in ("bar", "pub", "restaurant")):
        return "Bar & Restaurant"
    if any(token in raw for token in ("cafe", "tea", "chai", "snack")):
        return "Chai & Quick Bites"
    if any(token in raw for token in ("park", "garden", "picnic")):
        return "Picnic Spot & Landscape"
    if any(token in raw for token in ("attraction", "heritage", "museum", "tourist")):
        return "Heritage & Sight"
    if any(token in raw for token in ("doctor", "clinic", "hospital")):
        return "Clinic / Doctor"
    return "Food"


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _meaningful_address(props: Dict[str, Any], address: str, city: str) -> bool:
    if str(props.get("address_line1") or "").strip():
        return True
    if str(props.get("street") or "").strip():
        return True
    normalized_address = _normalize(address)
    normalized_city = _normalize(city)
    if not normalized_address or normalized_address == normalized_city:
        return False
    # A formatted address containing at least one additional address component
    # is useful for manual verification; a city name alone is not.
    return "," in str(address or "") or bool(props.get("housenumber") or props.get("postcode"))


def _geoapify_categories_match(categories: Any, canonical_category: str) -> bool:
    """Return true only when Geoapify's own category metadata matches the request."""
    if isinstance(categories, str):
        actual_categories = [categories.strip().lower()]
    elif isinstance(categories, list):
        actual_categories = [str(item).strip().lower() for item in categories if str(item).strip()]
    else:
        actual_categories = []
    requested_categories = [
        str(item).strip().lower()
        for item in GEOAPIFY_CATEGORIES.get(canonical_category, [])
        if str(item).strip()
    ]
    if not actual_categories or not requested_categories:
        return False
    # Category keys are hierarchical. A requested parent includes its children,
    # but a broader/unrelated returned category is not accepted as a match.
    return any(
        actual == requested or actual.startswith(requested + ".")
        for actual in actual_categories
        for requested in requested_categories
    )


def _is_in_requested_city(props: Dict[str, Any], address: str, city: str) -> bool:
    """Reject obvious cross-city results, especially Mira-Bhayandar in a Vasai-Virar run."""
    norm_city = _normalize(city)
    local_fields = (
        "city", "town", "village", "municipality", "suburb", "neighbourhood",
        "neighborhood", "quarter", "district_name", "locality",
    )
    local_values = [str(props.get(key) or "").strip() for key in local_fields]
    address_norm = _normalize(address)
    local_blob = _normalize(" ".join(value for value in local_values if value))
    combined = f" {local_blob} {address_norm} "

    if norm_city in {"vasai virar", "vasai-virar"}:
        # These nearby place names are not Vasai-Virar; reject them even when a
        # large circular search radius overlaps their area.
        blocked = ("mira bhayandar", "mira bhayander", "bhayandar", "bhayander")
        if any(term in combined for term in blocked):
            return False
        accepted_localities = (
            "vasai virar", "vasai", "naigaon", "nallasopara", "nalasopara", "virar",
        )
        return any(term in combined for term in accepted_localities)

    # For other cities, prefer explicit locality/address evidence rather than
    # trusting coordinates alone. Keep this generic and conservative.
    if norm_city and (norm_city in local_blob or norm_city in address_norm):
        return True
    explicit_places = [value for value in local_values if value]
    if explicit_places:
        return any(norm_city in _normalize(value) for value in explicit_places)
    return False


def _osm_source_url(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}"


@dataclass
class ScoutCandidate:
    dedupe_key: str
    city: str
    category: str
    name: str
    address: str
    latitude: float
    longitude: float
    website_url: str
    contact_phone: str
    source: str
    source_url: str
    google_place_id: str
    fsq_place_id: str
    geoapify_place_id: str
    confidence: float
    verification_status: str
    status: str = "candidate"
    discovered_at: str = ""


@dataclass
class ScoutRunResult:
    city: str
    requested_category: str
    canonical_category: str
    requested_quantity: int
    scanned: int
    inserted: int
    updated: int
    skipped_duplicates: int
    candidates: List[ScoutCandidate]
    source_counts: Dict[str, int]


class GemScout:
    """V10 Geoapify-first discovery using open-data evidence.

    This module never calls Google or Foursquare. Geoapify/OSM data can find
    named places, categories, coordinates and address details; contact data and
    image metadata are included only when the source actually provides them.
    Ratings and written reviews are not claimed because this provider does not
    supply them. Named, addressable and geolocated records are queued even when phone/site/image
    evidence is absent. They remain pending manual evidence review and are never
    published directly into Community Gems.
    """

    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client
        # Cache Geoapify's geocoded place ID so searches can be constrained to
        # the returned administrative/locality boundary instead of only a circle.
        self._city_place_ids: Dict[str, str] = {}
        self._last_geoapify_rejections: Dict[str, int] = {}

    async def _geocode_city(self, city: str) -> Optional[Tuple[float, float, str]]:
        if GEOAPIFY_API_KEY:
            params = {
                "text": f"{city.strip()}, India",
                "format": "json",
                "limit": "1",
                "lang": "en",
                "apiKey": GEOAPIFY_API_KEY,
            }
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
                    response = await client.get(GEOAPIFY_GEOCODE_URL, params=params)
                if response.status_code == 200:
                    rows = (response.json() or {}).get("results") or []
                    if rows:
                        row = rows[0]
                        place_id = str(row.get("place_id") or "").strip()
                        if place_id:
                            self._city_place_ids[_normalize(city)] = place_id
                        return float(row["lat"]), float(row["lon"]), str(row.get("formatted") or city)
                else:
                    body = response.text[:300].replace(GEOAPIFY_API_KEY, "[redacted]")
                    print(f"[Gem Scout Geoapify Geocode] HTTP {response.status_code}: {body}")
            except Exception as exc:
                safe_exc = str(exc).replace(GEOAPIFY_API_KEY, "[redacted]")
                print(f"[Gem Scout Geoapify Geocode] {type(exc).__name__}: {safe_exc[:180]}")

        # Best-effort geocoding fallback; this is not required for Geoapify place
        # search if its city geocode succeeded.
        params = {"q": f"{city.strip()}, India", "format": "jsonv2", "limit": "1", "accept-language": "en"}
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=4)) as client:
                response = await client.get(NOMINATIM_API_URL, params=params, headers=headers)
            if response.status_code == 200:
                rows = response.json() or []
                if rows:
                    row = rows[0]
                    return float(row["lat"]), float(row["lon"]), str(row.get("display_name") or city)
            else:
                print(f"[Gem Scout Geocode Fallback] Nominatim HTTP {response.status_code}: {response.text[:180]}")
        except Exception as exc:
            print(f"[Gem Scout Geocode Fallback] {type(exc).__name__}: {str(exc)[:180]}")
        return None

    async def _geoapify_place_details(self, place_id: str) -> Dict[str, Any]:
        if not GEOAPIFY_API_KEY or not place_id:
            return {}
        params = {"id": place_id, "features": "details", "apiKey": GEOAPIFY_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(12, connect=5)) as client:
                response = await client.get(GEOAPIFY_DETAILS_URL, params=params)
            if response.status_code != 200:
                body = response.text[:250].replace(GEOAPIFY_API_KEY, "[redacted]")
                print(f"[Gem Scout Geoapify Details] HTTP {response.status_code}: {body}")
                return {}
            features = (response.json() or {}).get("features") or []
            merged: Dict[str, Any] = {}
            for feature in features:
                props = feature.get("properties") or {}
                if isinstance(props, dict):
                    merged.update(props)
            return merged
        except Exception as exc:
            safe_exc = str(exc).replace(GEOAPIFY_API_KEY, "[redacted]")
            print(f"[Gem Scout Geoapify Details] {type(exc).__name__}: {safe_exc[:180]}")
            return {}

    @staticmethod
    def _geoapify_row(feature: Dict[str, Any], city: str) -> Dict[str, Any]:
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        lon = props.get("lon")
        lat = props.get("lat")
        if (lon is None or lat is None) and len(coordinates) >= 2:
            lon, lat = coordinates[0], coordinates[1]
        place_id = str(props.get("place_id") or "").strip()
        categories = props.get("categories") or []
        if isinstance(categories, str):
            categories = [categories]
        address = str(props.get("formatted") or "").strip()
        if not address:
            line1 = str(props.get("address_line1") or props.get("name") or "").strip()
            line2 = str(props.get("address_line2") or "").strip()
            address = ", ".join(part for part in (line1, line2) if part)
        contact_obj = props.get("contact") or {}
        if isinstance(contact_obj, dict):
            search_phone = contact_obj.get("phone") or ""
            search_website = contact_obj.get("website") or ""
        else:
            search_phone = props.get("phone") or ""
            search_website = ""
        media_obj = props.get("wiki_and_media") or {}
        if not isinstance(media_obj, dict):
            media_obj = {}
        return {
            "geoapify_place_id": place_id,
            # Do not promote address_line1 to a business name: for roads and
            # unnamed map features, that field may simply be a street name.
            "name": str(props.get("name") or "").strip(),
            "address": address,
            "latitude": float(lat) if lat is not None else None,
            "longitude": float(lon) if lon is not None else None,
            "categories": [str(item) for item in categories],
            "website_url": str(props.get("website") or search_website or "").strip(),
            "contact_phone": str(search_phone or props.get("phone") or "").strip(),
            "opening_hours": str(props.get("opening_hours") or "").strip(),
            "image_url": str(media_obj.get("image") or "").strip(),
            "source_url": _osm_source_url(float(lat), float(lon)) if lat is not None and lon is not None else "",
            "raw_properties": props,
            "city": city.strip(),
        }

    async def _geoapify_candidates(self, city: str, canonical_category: str, limit: int) -> List[Dict[str, Any]]:
        if not GEOAPIFY_API_KEY:
            print("[Gem Scout Geoapify] GEOAPIFY_API_KEY is not configured")
            return []
        center = await self._geocode_city(city)
        if not center:
            print(f"[Gem Scout Geoapify] Could not resolve city center for {city!r}")
            return []
        lat, lon, _formatted_city = center
        categories = GEOAPIFY_CATEGORIES.get(canonical_category, GEOAPIFY_CATEGORIES["Food"])
        city_place_id = self._city_place_ids.get(_normalize(city), "")
        spatial_filter = f"place:{city_place_id}" if city_place_id else f"circle:{lon},{lat},20000"
        params = {
            "categories": ",".join(categories),
            "filter": spatial_filter,
            "bias": f"proximity:{lon},{lat}",
            "limit": str(max(1, min(50, max(limit * 3, limit)))),
            "lang": "en",
            "apiKey": GEOAPIFY_API_KEY,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=5)) as client:
                response = await client.get(GEOAPIFY_PLACES_URL, params=params)
                if response.status_code != 200 and city_place_id:
                    body = response.text[:250].replace(GEOAPIFY_API_KEY, "[redacted]")
                    print(f"[Gem Scout Geoapify Boundary Filter] HTTP {response.status_code}: {body}; retrying with bounded radius")
                    params["filter"] = f"circle:{lon},{lat},20000"
                    response = await client.get(GEOAPIFY_PLACES_URL, params=params)
            if response.status_code != 200:
                body = response.text[:350].replace(GEOAPIFY_API_KEY, "[redacted]")
                print(f"[Gem Scout Geoapify Discovery] HTTP {response.status_code}: {body}")
                return []
            features = (response.json() or {}).get("features") or []
        except Exception as exc:
            safe_exc = str(exc).replace(GEOAPIFY_API_KEY, "[redacted]")
            print(f"[Gem Scout Geoapify Discovery] {type(exc).__name__}: {safe_exc[:180]}")
            return []

        rows: List[Dict[str, Any]] = []
        seen = set()
        details_calls = 0
        rejected = {"missing_real_name": 0, "wrong_category": 0, "outside_city": 0, "missing_place_id_or_coordinates": 0}
        max_details = min(max(limit * 3, limit), 40)
        for feature in features:
            row = self._geoapify_row(feature, city)
            pid, name = row.get("geoapify_place_id"), row.get("name")
            lat1, lon1 = row.get("latitude"), row.get("longitude")
            props = row.get("raw_properties") or {}
            if not name:
                rejected["missing_real_name"] += 1
                continue
            if not _geoapify_categories_match(row.get("categories"), canonical_category):
                rejected["wrong_category"] += 1
                continue
            if not _is_in_requested_city(props, str(row.get("address") or ""), city):
                rejected["outside_city"] += 1
                continue
            if not pid or lat1 is None or lon1 is None:
                rejected["missing_place_id_or_coordinates"] += 1
                continue
            dedupe = _normalize(pid) or _normalize(name)
            if dedupe in seen:
                continue
            if not _meaningful_address(props, str(row.get("address") or ""), city):
                # Do not discard it silently: it could still be useful after a
                # manual review, but this queue is specifically for addressable leads.
                continue
            if details_calls < max_details:
                details = await self._geoapify_place_details(str(pid))
                details_calls += 1
                contact = details.get("contact") or {}
                if isinstance(contact, dict):
                    row["contact_phone"] = str(
                        contact.get("phone")
                        or next(iter(contact.get("phone_other") or []), "")
                        or row.get("contact_phone")
                        or ""
                    ).strip()
                website_other = details.get("website_other") or []
                if not isinstance(website_other, list):
                    website_other = []
                row["website_url"] = str(
                    details.get("website") or (website_other[0] if website_other else "") or row.get("website_url") or ""
                ).strip()
                detail_address = str(details.get("formatted") or details.get("address_line1") or "").strip()
                if detail_address and len(detail_address) > len(str(row.get("address") or "")):
                    row["address"] = detail_address
                media = details.get("wiki_and_media") or {}
                if isinstance(media, dict):
                    row["image_url"] = str(media.get("image") or row.get("image_url") or "").strip()
                row["opening_hours"] = str(details.get("opening_hours") or row.get("opening_hours") or "").strip()
                row["source_url"] = _osm_source_url(float(lat1), float(lon1))
                row["details_requested"] = True
            else:
                row["details_requested"] = False
            row.pop("raw_properties", None)
            rows.append(row)
            seen.add(dedupe)
            if len(rows) >= max(1, min(limit * 3, 60)):
                break

        self._last_geoapify_rejections = rejected
        print(
            "[Gem Scout Geoapify Discovery] "
            f"features={len(features)} accepted={len(rows)} details_requested={details_calls} "
            f"with_phone={sum(bool(r.get('contact_phone')) for r in rows)} "
            f"with_website={sum(bool(r.get('website_url')) for r in rows)} "
            f"with_image_reference={sum(bool(r.get('image_url')) for r in rows)} "
            f"rejected={rejected} category={canonical_category!r} city={city!r} "
            f"spatial_filter={'place' if city_place_id else 'circle'}"
        )
        return rows

    async def _osm_candidates(self, city: str, canonical_category: str, limit: int, radius_m: int = 15000) -> List[Dict[str, Any]]:
        profile_query = OSM_CATEGORY_QUERIES.get(canonical_category)
        if not profile_query:
            return []
        center = await self._geocode_city(city)
        if not center:
            return []
        lat, lon, _ = center
        query = f"[out:json][timeout:15];({profile_query.format(radius=radius_m, lat=lat, lng=lon)});out center tags;"
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(18, connect=5)) as client:
                response = await client.post(OVERPASS_API_URL, data={"data": query}, headers=headers)
            if response.status_code != 200:
                print(f"[Gem Scout OSM] Overpass HTTP {response.status_code}: {response.text[:250]}")
                return []
            elements = (response.json() or {}).get("elements") or []
            rows: List[Dict[str, Any]] = []
            seen = set()
            for element in elements:
                tags = element.get("tags") or {}
                name = str(tags.get("name") or tags.get("name:en") or "").strip()
                if not name:
                    continue
                key = _normalize(name)
                if key in seen:
                    continue
                if element.get("lat") is not None and element.get("lon") is not None:
                    item_lat, item_lon = float(element["lat"]), float(element["lon"])
                else:
                    center_pos = element.get("center") or {}
                    if center_pos.get("lat") is None or center_pos.get("lon") is None:
                        continue
                    item_lat, item_lon = float(center_pos["lat"]), float(center_pos["lon"])
                parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:suburb"), tags.get("addr:city"), tags.get("addr:postcode")]
                address = ", ".join(str(item).strip() for item in parts if str(item or "").strip())
                osm_id = f"osm:{element.get('type', '')}/{element.get('id', '')}"
                rows.append({
                    "geoapify_place_id": "",
                    "osm_id": osm_id,
                    "name": name,
                    "address": address,
                    "latitude": item_lat,
                    "longitude": item_lon,
                    "website_url": str(tags.get("website") or tags.get("contact:website") or "").strip(),
                    "contact_phone": str(tags.get("phone") or tags.get("contact:phone") or "").strip(),
                    "opening_hours": str(tags.get("opening_hours") or "").strip(),
                    "image_url": str(tags.get("image") or "").strip(),
                    "source_url": _osm_source_url(item_lat, item_lon),
                    "city": city.strip(),
                })
                seen.add(key)
                if len(rows) >= min(max(limit * 3, limit), 60):
                    break
            print(f"[Gem Scout OSM] named_candidates={len(rows)} category={canonical_category!r} city={city!r}")
            return rows
        except Exception as exc:
            print(f"[Gem Scout OSM] {type(exc).__name__}: {str(exc)[:180]}")
            return []

    @staticmethod
    def _evidence_score(row: Dict[str, Any]) -> float:
        # This is discovery confidence, NOT a business rating. Ratings/reviews
        # are unavailable from this source and therefore are not fabricated.
        score = 0.35
        if str(row.get("address") or "").strip():
            score += 0.18
        if str(row.get("contact_phone") or "").strip():
            score += 0.14
        if str(row.get("website_url") or "").strip():
            score += 0.10
        if str(row.get("image_url") or "").strip():
            score += 0.08
        if str(row.get("opening_hours") or "").strip():
            score += 0.05
        if str(row.get("geoapify_place_id") or row.get("osm_id") or "").strip():
            score += 0.10
        return round(min(0.79, score), 3)

    async def run_community_scout(
        self,
        city: str,
        category: str,
        quantity: int = 10,
        verify_google: bool = True,
    ) -> ScoutRunResult:
        # Keep this argument to preserve the caller contract in main.py. It is
        # intentionally ignored; this module never calls Google Places.
        city = city.strip()
        requested_category = category.strip() or "Food"
        quantity = max(1, min(int(quantity), 50))
        canonical_category = _resolve_category(requested_category)
        if verify_google:
            print("[Gem Scout] Geoapify-first open-data discovery; Google verification is disabled in this module.")

        primary_rows = await self._geoapify_candidates(city, canonical_category, quantity)
        rows = list(primary_rows)
        primary_ids = set()
        for item in rows:
            primary_ids.add(_normalize(item.get("geoapify_place_id")))
            primary_ids.add(_normalize(item.get("name")))
        usable_primary_count = sum(
            bool(str(item.get("contact_phone") or "").strip() or str(item.get("website_url") or "").strip())
            for item in primary_rows
        )

        # OSM is only a secondary fallback. If the primary source produces fewer
        # contactable leads than requested, try OSM for additional local records.
        # Records remain pending and are never represented as having reviews or ratings.
        if usable_primary_count < quantity:
            osm_rows = await self._osm_candidates(city, canonical_category, quantity)
            for item in osm_rows:
                name_key = _normalize(item.get("name"))
                if name_key and name_key not in primary_ids:
                    rows.append(item)
                    primary_ids.add(name_key)

        scanned = len(rows)
        now = datetime.now(timezone.utc).isoformat()
        accepted: List[ScoutCandidate] = []
        inserted = updated = skipped_duplicates = 0
        rejected: Dict[str, int] = {}

        # Put richer open-data records first. Records remain pending until the
        # review workflow supplies ratings, written feedback, and photos.
        rows.sort(
            key=lambda item: (
                bool(item.get("address")),
                bool(item.get("contact_phone")),
                bool(item.get("website_url")),
                bool(item.get("image_url")),
                bool(item.get("opening_hours")),
            ),
            reverse=True,
        )

        for row in rows:
            if len(accepted) >= quantity:
                break
            name = str(row.get("name") or "").strip()
            address = str(row.get("address") or "").strip()
            lat = row.get("latitude")
            lon = row.get("longitude")
            provider_id = str(row.get("geoapify_place_id") or "").strip()
            osm_id = str(row.get("osm_id") or "").strip()
            if not name:
                rejected["missing_name"] = rejected.get("missing_name", 0) + 1
                continue
            if not address:
                rejected["missing_address"] = rejected.get("missing_address", 0) + 1
                continue
            if lat is None or lon is None:
                rejected["missing_coordinates"] = rejected.get("missing_coordinates", 0) + 1
                continue
            if not provider_id and not osm_id:
                rejected["missing_source_place_id"] = rejected.get("missing_source_place_id", 0) + 1
                continue
            # This is a candidate queue, not the published Community Gems feed.
            # Missing phone/site/image/rating/reviews must remain explicit gaps
            # for manual enrichment; it must not erase a named, addressable,
            # geolocated place with a stable source ID from the review queue.

            external_id = provider_id or osm_id
            dedupe_key = "|".join([_normalize(city), _normalize(canonical_category), _normalize(external_id)])
            candidate = ScoutCandidate(
                dedupe_key=dedupe_key,
                city=city,
                category=canonical_category,
                name=name,
                address=address,
                latitude=float(lat),
                longitude=float(lon),
                website_url=str(row.get("website_url") or "").strip(),
                contact_phone=str(row.get("contact_phone") or "").strip(),
                source="GEOAPIFY_OPEN_DATA" if provider_id else "OPENSTREETMAP_OPEN_DATA",
                source_url=str(row.get("source_url") or _osm_source_url(float(lat), float(lon))),
                google_place_id="",
                fsq_place_id="",
                geoapify_place_id=provider_id,
                confidence=self._evidence_score(row),
                verification_status="open_data_pending_rich_evidence",
                status="candidate",
                discovered_at=now,
            )

            try:
                existing = (
                    self.supabase.table("gem_scout_candidates")
                    .select("id,status,confidence,google_place_id,fsq_place_id,geoapify_place_id,evidence_status")
                    .eq("dedupe_key", dedupe_key)
                    .limit(1)
                    .execute()
                )
                existing_rows = existing.data or []
                if existing_rows:
                    previous = existing_rows[0]
                    record_id = previous.get("id")
                    previous_status = str(previous.get("status") or "candidate").lower()
                    previous_evidence_status = str(previous.get("evidence_status") or "not_submitted").lower()

                    # V10 safety guard: once a reviewer approves manual evidence,
                    # later discovery runs must not overwrite the reviewed business
                    # name, phone, website, source links, photo references or status.
                    # Only refresh the discovery timestamp for such a row.
                    if previous_evidence_status == "approved":
                        self.supabase.table("gem_scout_candidates").update({
                            "last_seen_at": now,
                            "updated_at": now,
                        }).eq("id", record_id).execute()
                        skipped_duplicates += 1
                        continue

                    if previous_status in {"approved", "published", "rejected", "dismissed"}:
                        self.supabase.table("gem_scout_candidates").update({
                            "last_seen_at": now,
                            "updated_at": now,
                        }).eq("id", record_id).execute()
                        skipped_duplicates += 1
                        continue
                    self.supabase.table("gem_scout_candidates").update({
                        "name": candidate.name,
                        "address": candidate.address,
                        "latitude": candidate.latitude,
                        "longitude": candidate.longitude,
                        "website_url": candidate.website_url,
                        "contact_phone": candidate.contact_phone,
                        "source": candidate.source,
                        "source_url": candidate.source_url,
                        "google_place_id": "",
                        "fsq_place_id": "",
                        "geoapify_place_id": candidate.geoapify_place_id,
                        "confidence": candidate.confidence,
                        "verification_status": candidate.verification_status,
                        "last_seen_at": now,
                        "updated_at": now,
                    }).eq("id", record_id).execute()
                    updated += 1
                else:
                    self.supabase.table("gem_scout_candidates").insert(asdict(candidate) | {
                        "last_seen_at": now,
                        "updated_at": now,
                    }).execute()
                    inserted += 1
            except Exception as exc:
                raise RuntimeError(f"Gem Scout Supabase write failed: {exc}") from exc
            accepted.append(candidate)

        source_counts: Dict[str, int] = {
            "DISCOVERY_SCANNED": scanned,
            "GEOAPIFY_DISCOVERED": len(primary_rows),
            "OPENSTREETMAP_SCANNED": max(0, scanned - len(primary_rows)),
            "QUEUED_PENDING_RICH_EVIDENCE": len(accepted),
            "QUEUED_WITH_PHONE": sum(bool(str(item.get("contact_phone") or "").strip()) for item in rows[:quantity]),
            "QUEUED_WITH_WEBSITE": sum(bool(str(item.get("website_url") or "").strip()) for item in rows[:quantity]),
            "QUEUED_WITH_IMAGE_REFERENCE": sum(bool(str(item.get("image_url") or "").strip()) for item in rows[:quantity]),
            "INSERTED": inserted,
            "UPDATED": updated,
            "SKIPPED_EXISTING_DECISIONS": skipped_duplicates,
        }
        for reason, count in rejected.items():
            source_counts[f"REJECTED_{reason.upper()}"] = count
        for reason, count in self._last_geoapify_rejections.items():
            source_counts[f"GEOAPIFY_REJECTED_{reason.upper()}"] = count

        print(
            f"[Gem Scout] city={city!r} category={canonical_category!r} "
            f"scanned={scanned} queued={len(accepted)} inserted={inserted} updated={updated} "
            f"pending_evidence={len(accepted)} rejected={rejected}"
        )
        return ScoutRunResult(
            city=city,
            requested_category=requested_category,
            canonical_category=canonical_category,
            requested_quantity=quantity,
            scanned=scanned,
            inserted=inserted,
            updated=updated,
            skipped_duplicates=skipped_duplicates,
            candidates=accepted,
            source_counts=source_counts,
        )
