from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import httpx


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
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"


# Each profile deliberately targets physical business/place tags rather than
# broad or ambiguous OSM objects such as roads/highways.
CATEGORY_PROFILES: Dict[str, Dict[str, str]] = {
    "pharmacy": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["healthcare"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "chemist": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["healthcare"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "medical store": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["healthcare"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "pharmacy chemist": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["healthcare"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "food": {
        "canonical": "Food",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|cafe|fast_food|food_court|ice_cream",i]["name"];',
    },
    "restaurant": {
        "canonical": "Bar & Restaurant",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|fast_food",i]["name"];',
    },
    "cafe": {
        "canonical": "Chai & Quick Bites",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="cafe"]["name"];',
    },
    "bar": {
        "canonical": "Bar & Restaurant",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"~"bar|pub",i]["name"];',
    },
    "mall": {
        "canonical": "Market, Bazaar & Mall",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"="mall"]["name"];nwr(around:{radius},{lat},{lng})["shop"="department_store"]["name"];nwr(around:{radius},{lat},{lng})["shop"="supermarket"]["name"];',
    },
    "shopping": {
        "canonical": "Market, Bazaar & Mall",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"mall|department_store|supermarket|convenience|clothes|shoes|fashion",i]["name"];',
    },
    "clothing": {
        "canonical": "Clothing & Fashion",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"clothes|fashion|shoes",i]["name"];',
    },
    "dresses": {
        "canonical": "Clothing & Fashion",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"clothes|fashion|fabric|tailor",i]["name"];',
    },
    "hotel": {
        "canonical": "Hotel & Stay",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"hotel|guest_house|hostel|motel",i]["name"];',
    },
    "hotels": {
        "canonical": "Hotel & Stay",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"hotel|guest_house|hostel|motel",i]["name"];',
    },
    "attractions": {
        "canonical": "Heritage & Sight",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"attraction|museum|viewpoint|zoo|theme_park",i]["name"];nwr(around:{radius},{lat},{lng})["historic"]["name"];',
    },
    "parks": {
        "canonical": "Picnic Spot & Landscape",
        "query": 'nwr(around:{radius},{lat},{lng})["leisure"="park"]["name"];nwr(around:{radius},{lat},{lng})["leisure"="garden"]["name"];',
    },
    "barber": {
        "canonical": "Barber & Salon",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"hairdresser|beauty",i]["name"];',
    },
    "salon": {
        "canonical": "Barber & Salon",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"hairdresser|beauty",i]["name"];',
    },
    "electronics": {
        "canonical": "Electronics & Mobile",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"electronics|mobile_phone|computer",i]["name"];',
    },
}


CATEGORY_GOOGLE_QUERIES: Dict[str, str] = {
    "Pharmacy / Chemist": "pharmacies chemists medical stores",
    "Food": "restaurants cafes food places",
    "Bar & Restaurant": "restaurants bars pubs",
    "Chai & Quick Bites": "cafes tea shops snack places",
    "Market, Bazaar & Mall": "malls markets supermarkets shopping",
    "Clothing & Fashion": "clothing stores fashion dress shops",
    "Hotel & Stay": "hotels resorts guest houses",
    "Heritage & Sight": "tourist attractions landmarks",
    "Picnic Spot & Landscape": "parks gardens picnic places",
    "Barber & Salon": "barbers salons beauty parlours",
    "Electronics & Mobile": "electronics mobile phone shops",
}


GOOGLE_DISCOVERY_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.googleMapsUri",
    "places.websiteUri",
    "places.internationalPhoneNumber",
    "places.nationalPhoneNumber",
    "places.rating",
    "places.userRatingCount",
    "places.photos",
    "places.primaryType",
    "places.primaryTypeDisplayName",
    "places.types",
])


def _normalize(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _name_similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    a_tokens, b_tokens = set(na.split()), set(nb.split())
    overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return max(ratio, overlap)


def _resolve_profile(category: str) -> Dict[str, str]:
    raw = _normalize(category)
    if raw in CATEGORY_PROFILES:
        return CATEGORY_PROFILES[raw]
    for key, profile in CATEGORY_PROFILES.items():
        key_norm = _normalize(key)
        if key_norm and (key_norm in raw or raw in key_norm):
            return profile
    if any(token in raw for token in ("pharmacy", "chemist", "medical store")):
        return CATEGORY_PROFILES["pharmacy"]
    if any(token in raw for token in ("hotel", "stay", "resort", "hostel")):
        return CATEGORY_PROFILES["hotel"]
    if any(token in raw for token in ("dress", "clothing", "fashion", "apparel")):
        return CATEGORY_PROFILES["clothing"]
    if any(token in raw for token in ("mall", "shopping", "market")):
        return CATEGORY_PROFILES["shopping"]
    if any(token in raw for token in ("bar", "pub")):
        return CATEGORY_PROFILES["bar"]
    if any(token in raw for token in ("cafe", "tea", "chai")):
        return CATEGORY_PROFILES["cafe"]
    if any(token in raw for token in ("restaurant", "dining", "food", "eat")):
        return CATEGORY_PROFILES["food"]
    return CATEGORY_PROFILES["food"]


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
    """Evidence-first Community Gem discovery engine.

    Discovery starts with permitted open data, then every candidate is checked
    live against Google Places. A candidate is only persisted when the live
    provider confirms the business/place and exposes the minimum evidence needed
    for a useful Community Gem: rating, at least one review, phone and photo.

    Google content is used only for live verification. The persistent candidate
    record keeps open-data fields plus the Google place ID.
    """

    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client

    async def _geocode_city(self, city: str) -> Optional[Tuple[float, float, str]]:
        # Prefer Nominatim for open-data discovery. If it is unavailable from
        # the hosting network, fall back to a lightweight Google place lookup
        # so the Scout can still obtain the search center.
        params = {
            "q": f"{city}, India",
            "format": "jsonv2",
            "limit": "3",
            "accept-language": "en",
        }
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=6)) as client:
                response = await client.get(NOMINATIM_API_URL, params=params, headers=headers)
            if response.status_code == 200:
                rows = response.json() or []
                if rows:
                    chosen = rows[0]
                    return float(chosen["lat"]), float(chosen["lon"]), str(chosen.get("display_name") or city)
            else:
                print(f"[Gem Scout Geocode] Nominatim HTTP {response.status_code}")
        except Exception as exc:
            print(f"[Gem Scout Geocode] Nominatim unavailable: {exc}")

        if not GOOGLE_PLACES_API_KEY:
            return None

        try:
            endpoint = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
            payload = {
                "textQuery": f"{city}, India",
                "pageSize": 1,
                "languageCode": "en",
            }
            g_headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
                response = await client.post(endpoint, json=payload, headers=g_headers)
            if response.status_code != 200:
                print(f"[Gem Scout Geocode] Google fallback HTTP {response.status_code}: {response.text[:300]}")
                return None
            places = (response.json() or {}).get("places") or []
            if not places:
                return None
            place = places[0]
            loc = place.get("location") or {}
            if loc.get("latitude") is None or loc.get("longitude") is None:
                return None
            label = str(place.get("formattedAddress") or city)
            print("[Gem Scout Geocode] Using Google fallback for search center")
            return float(loc["latitude"]), float(loc["longitude"]), label
        except Exception as exc:
            print(f"[Gem Scout Geocode] Google fallback unavailable: {exc}")
            return None

    async def _osm_candidates(
        self,
        city: str,
        profile: Dict[str, str],
        limit: int,
        radius_m: int = 15000,
    ) -> List[Dict[str, Any]]:
        resolved = await self._geocode_city(city)
        if not resolved:
            return []
        lat, lng, _ = resolved
        statement = profile["query"].format(radius=radius_m, lat=lat, lng=lng)
        query = f"[out:json][timeout:40];({statement});out center tags;"
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(50, connect=10)) as client:
                response = await client.post(
                    OVERPASS_API_URL,
                    data={"data": query},
                    headers=headers,
                )
            if response.status_code != 200:
                print(f"[Gem Scout OSM] Overpass HTTP {response.status_code}: {response.text[:300]}")
                return []

            elements = (response.json() or {}).get("elements") or []
            rows: List[Dict[str, Any]] = []
            seen = set()
            for element in elements:
                tags = element.get("tags") or {}
                name = str(tags.get("name") or tags.get("name:en") or "").strip()
                if not name:
                    continue
                dedupe_name = _normalize(name)
                if dedupe_name in seen:
                    continue

                if element.get("lat") is not None and element.get("lon") is not None:
                    item_lat, item_lng = float(element["lat"]), float(element["lon"])
                else:
                    center = element.get("center") or {}
                    if center.get("lat") is None or center.get("lon") is None:
                        continue
                    item_lat, item_lng = float(center["lat"]), float(center["lon"])

                address_parts = [
                    tags.get("addr:housenumber"),
                    tags.get("addr:street"),
                    tags.get("addr:suburb"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode"),
                ]
                address = ", ".join(str(x).strip() for x in address_parts if str(x or "").strip())
                # OSM address data is optional here because Google verification
                # is the rich-evidence gate. Do not discard a real named place
                # just because OSM omitted addr:* tags.

                osm_id = f"osm:{element.get('type', '')}/{element.get('id', '')}"
                source_url = (
                    f"https://www.openstreetmap.org/?mlat={item_lat}&mlon={item_lng}"
                    f"#map=17/{item_lat}/{item_lng}"
                )
                website = str(tags.get("website") or tags.get("contact:website") or "").strip()
                phone = str(tags.get("phone") or tags.get("contact:phone") or "").strip()

                rows.append({
                    "osm_id": osm_id,
                    "name": name,
                    "address": address,
                    "city": city.strip(),
                    "latitude": item_lat,
                    "longitude": item_lng,
                    "website_url": website,
                    "contact_phone": phone,
                    "source_url": source_url,
                })
                seen.add(dedupe_name)
                if len(rows) >= min(max(limit * 5, limit), 100):
                    break
            return rows
        except Exception as exc:
            print(f"[Gem Scout OSM] {exc}")
            return []

    @staticmethod
    def _google_place_matches_category(place: Dict[str, Any], canonical_category: str) -> bool:
        raw = " ".join([
            str(place.get("primaryType") or ""),
            str(place.get("primaryTypeDisplayName") or ""),
            " ".join(str(x) for x in (place.get("types") or [])),
        ]).lower()
        checks = {
            "Pharmacy / Chemist": ("pharmacy", "drugstore", "chemist"),
            "Barber & Salon": ("barber", "hair_care", "beauty", "salon"),
            "Bar & Restaurant": ("restaurant", "bar", "pub", "night_club"),
            "Chai & Quick Bites": ("cafe", "coffee", "bakery", "tea", "fast_food"),
            "Hotel & Stay": ("hotel", "resort", "hostel", "guest_house", "motel"),
            "Market, Bazaar & Mall": ("shopping_mall", "department_store", "supermarket", "market"),
            "Clothing & Fashion": ("clothing", "fashion", "shoe", "boutique"),
            "Electronics & Mobile": ("electronics", "mobile_phone", "computer", "store"),
            "Heritage & Sight": ("tourist_attraction", "museum", "landmark", "point_of_interest"),
            "Picnic Spot & Landscape": ("park", "garden", "picnic", "campground"),
        }
        wanted = checks.get(canonical_category)
        if not wanted:
            return True
        return any(token in raw for token in wanted)

    async def _google_candidates(
        self,
        city: str,
        canonical_category: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Discover named places directly from Google Places when open-data
        providers are unreachable from the hosting network. Results are then
        passed through the same rich-evidence gate before persistence."""
        if not GOOGLE_PLACES_API_KEY:
            return []

        endpoint = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
        category_hint = CATEGORY_GOOGLE_QUERIES.get(canonical_category, canonical_category)

        resolved = await self._geocode_city(city)
        payload: Dict[str, Any] = {
            "textQuery": f"{category_hint} in {city}, India",
            "pageSize": min(max(limit * 2, 10), 20),
            "languageCode": "en",
        }
        if resolved:
            lat, lng, _ = resolved
            payload["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": 20000.0,
                }
            }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": GOOGLE_DISCOVERY_FIELD_MASK,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=6)) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"[Gem Scout Google Discovery] HTTP {response.status_code}: {response.text[:500]}")
                return []

            places = (response.json() or {}).get("places") or []
            rows: List[Dict[str, Any]] = []
            seen = set()
            center = resolved[:2] if resolved else None

            for place in places:
                if not self._google_place_matches_category(place, canonical_category):
                    continue
                display = str((place.get("displayName") or {}).get("text") or "").strip()
                if not display:
                    continue
                dedupe = _normalize(display)
                if not dedupe or dedupe in seen:
                    continue
                loc = place.get("location") or {}
                if loc.get("latitude") is None or loc.get("longitude") is None:
                    continue
                lat, lng = float(loc["latitude"]), float(loc["longitude"])
                if center and _distance_m(center[0], center[1], lat, lng) > 25000:
                    continue

                photos = place.get("photos") or []
                phone = str(place.get("internationalPhoneNumber") or place.get("nationalPhoneNumber") or "").strip()
                rating = place.get("rating")
                reviews = place.get("userRatingCount")
                address = str(place.get("formattedAddress") or "").strip()
                place_id = str(place.get("id") or "").strip()
                website = str(place.get("websiteUri") or "").strip()

                rows.append({
                    "osm_id": "",
                    "name": display,
                    "address": address,
                    "city": city,
                    "latitude": lat,
                    "longitude": lng,
                    "website_url": website,
                    "contact_phone": phone,
                    "source_url": str(place.get("googleMapsUri") or ""),
                    "google_place": place,
                    "google_place_id": place_id,
                    "rating": rating,
                    "review_count": int(reviews) if isinstance(reviews, (int, float)) else 0,
                    "photo_count": len(photos),
                })
                seen.add(dedupe)
                if len(rows) >= max(limit * 2, limit):
                    break

            print(f"[Gem Scout Google Discovery] discovered={len(rows)} category={canonical_category!r} city={city!r}")
            return rows
        except Exception as exc:
            print(f"[Gem Scout Google Discovery] {exc}")
            return []

    async def _google_verify(
        self,
        candidate: Dict[str, Any],
        city: str,
        canonical_category: str,
    ) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
        """Return (best_place, match_score, evidence) for live Google verification."""
        if not GOOGLE_PLACES_API_KEY:
            return None, 0.0, {"reason": "google_key_missing"}

        endpoint = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.googleMapsUri",
            "places.websiteUri",
            "places.internationalPhoneNumber",
            "places.nationalPhoneNumber",
            "places.rating",
            "places.userRatingCount",
            "places.photos",
            "places.primaryType",
            "places.primaryTypeDisplayName",
            "places.types",
        ])
        category_hint = CATEGORY_GOOGLE_QUERIES.get(canonical_category, canonical_category)
        payload = {
            "textQuery": f"{candidate['name']} {category_hint} {city}",
            "pageSize": 5,
            "languageCode": "en",
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": field_mask,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"[Gem Scout Google] HTTP {response.status_code}: {response.text[:500]}")
                return None, 0.0, {"reason": f"http_{response.status_code}"}

            places = (response.json() or {}).get("places") or []
            best_place: Optional[Dict[str, Any]] = None
            best_score = 0.0
            for place in places:
                display = (place.get("displayName") or {}).get("text") or ""
                loc = place.get("location") or {}
                plat, plng = loc.get("latitude"), loc.get("longitude")
                name_score = _name_similarity(candidate["name"], display)
                dist_score = 0.0
                distance_m = None
                if plat is not None and plng is not None:
                    distance_m = _distance_m(
                        float(candidate["latitude"]),
                        float(candidate["longitude"]),
                        float(plat),
                        float(plng),
                    )
                    dist_score = max(0.0, 1.0 - min(distance_m, 7500.0) / 7500.0)

                score = name_score * 0.70 + dist_score * 0.30
                if score > best_score:
                    best_score = score
                    best_place = place

            if not best_place or best_score < 0.68:
                return None, best_score, {"reason": "weak_match"}

            phone = str(
                best_place.get("internationalPhoneNumber")
                or best_place.get("nationalPhoneNumber")
                or ""
            ).strip()
            rating = best_place.get("rating")
            reviews = best_place.get("userRatingCount")
            photos = best_place.get("photos") or []
            formatted_address = str(best_place.get("formattedAddress") or "").strip()
            website = str(best_place.get("websiteUri") or "").strip()
            place_id = str(best_place.get("id") or "").strip()

            evidence = {
                "rating": rating,
                "review_count": int(reviews) if isinstance(reviews, (int, float)) else None,
                "phone": phone,
                "photo_count": len(photos),
                "formatted_address_present": bool(formatted_address),
                "website_present": bool(website),
                "place_id_present": bool(place_id),
                "match_score": round(best_score, 3),
            }
            return best_place, best_score, evidence
        except Exception as exc:
            print(f"[Gem Scout Google] {exc}")
            return None, 0.0, {"reason": "exception"}

    @staticmethod
    def _rich_evidence_gate(evidence: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Require the business evidence the user asked the Scout to prioritize."""
        rating = evidence.get("rating")
        reviews = evidence.get("review_count") or 0
        phone = str(evidence.get("phone") or "").strip()
        photo_count = int(evidence.get("photo_count") or 0)
        address_present = bool(evidence.get("formatted_address_present"))
        place_id_present = bool(evidence.get("place_id_present"))
        match_score = float(evidence.get("match_score") or 0.0)

        # Hard gate: without these signals we do not persist the candidate.
        if rating is None:
            return False, 0.0, "missing_rating"
        if int(reviews) < 1:
            return False, 0.0, "no_reviews"
        if not phone:
            return False, 0.0, "missing_phone"
        if photo_count < 1:
            return False, 0.0, "missing_photo"
        if not address_present:
            return False, 0.0, "missing_google_address"
        if not place_id_present:
            return False, 0.0, "missing_place_id"
        if match_score < 0.68:
            return False, 0.0, "weak_match"

        # Soft quality score for ranking candidates that passed the gate.
        score = 0.30
        score += 0.16 if float(rating) >= 4.0 else 0.10
        score += 0.08 if int(reviews) >= 10 else 0.04
        score += 0.08 if int(reviews) >= 50 else 0.00
        score += 0.10  # phone
        score += 0.10  # photo
        score += 0.08 if address_present else 0.00
        score += min(0.10, max(0.0, match_score) * 0.10)
        return True, round(min(0.99, score), 3), "rich_google_evidence"

    async def run_community_scout(
        self,
        city: str,
        category: str,
        quantity: int = 10,
        verify_google: bool = True,
    ) -> ScoutRunResult:
        city = city.strip()
        requested_category = category.strip() or "Food"
        quantity = max(1, min(int(quantity), 50))
        profile = _resolve_profile(requested_category)
        canonical_category = profile["canonical"]

        if not verify_google:
            print("[Gem Scout] verify_google=false is intentionally ignored for evidence-first mode.")
            verify_google = True

        # Google is the primary discovery path because this hosted service may
        # not be able to reach Overpass/Nominatim reliably. OSM remains an
        # optional secondary source when Google discovery returns too few rows.
        google_candidates = await self._google_candidates(city, canonical_category, quantity)
        candidates = google_candidates
        discovery_source = "GOOGLE" if google_candidates else "OSM"
        if len(candidates) < quantity:
            osm_candidates = await self._osm_candidates(city, profile, quantity)
            existing_names = {_normalize(row.get("name")) for row in candidates}
            for row in osm_candidates:
                if _normalize(row.get("name")) not in existing_names:
                    candidates.append(row)
                    existing_names.add(_normalize(row.get("name")))
            if osm_candidates:
                discovery_source = "GOOGLE+OPENSTREETMAP" if google_candidates else "OPENSTREETMAP"

        scanned = len(candidates)
        if scanned == 0:
            print(f"[Gem Scout] No candidates discovered for city={city!r}, category={requested_category!r}")
        accepted: List[ScoutCandidate] = []
        skipped_duplicates = 0
        inserted = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()

        # Strong local completeness first; then we will use live Google evidence
        # as the actual publication gate.
        candidates.sort(
            key=lambda row: (
                bool(row.get("address")),
                bool(row.get("contact_phone")),
                bool(row.get("website_url")),
                int(row.get("review_count") or 0),
                float(row.get("rating") or 0.0),
                int(row.get("photo_count") or 0),
            ),
            reverse=True,
        )

        rejection_counts: Dict[str, int] = {}
        for row in candidates:
            if len(accepted) >= quantity:
                break

            if row.get("google_place"):
                google_place = row.get("google_place")
                google_match_score = 1.0
                evidence = {
                    "rating": row.get("rating"),
                    "review_count": row.get("review_count") or 0,
                    "phone": row.get("contact_phone") or "",
                    "photo_count": row.get("photo_count") or 0,
                    "formatted_address_present": bool(row.get("address")),
                    "website_present": bool(row.get("website_url")),
                    "place_id_present": bool(row.get("google_place_id")),
                    "match_score": 1.0,
                }
            else:
                google_place, google_match_score, evidence = await self._google_verify(
                    row,
                    city,
                    canonical_category,
                )
            passed, evidence_score, evidence_reason = self._rich_evidence_gate(evidence)
            if not passed:
                rejection_counts[evidence_reason] = rejection_counts.get(evidence_reason, 0) + 1
                continue

            google_place_id = str(google_place.get("id") or "") if google_place else ""
            confidence = round(min(0.995, evidence_score), 3)

            dedupe_key = "|".join([
                _normalize(city),
                _normalize(canonical_category),
                _normalize(row["name"]),
            ])

            candidate = ScoutCandidate(
                dedupe_key=dedupe_key,
                city=city,
                category=canonical_category,
                name=row["name"],
                address=row["address"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                website_url=str(row.get("website_url") or ""),
                contact_phone=str(row.get("contact_phone") or ""),
                source=f"{discovery_source}+GOOGLE_VERIFIED",
                source_url=str(row.get("source_url") or ""),
                google_place_id=google_place_id,
                confidence=confidence,
                verification_status="google_verified_rich_evidence",
                status="candidate",
                discovered_at=now,
            )

            try:
                existing = (
                    self.supabase.table("gem_scout_candidates")
                    .select("id,status,confidence,google_place_id")
                    .eq("dedupe_key", dedupe_key)
                    .limit(1)
                    .execute()
                )
                existing_rows = existing.data or []
                if existing_rows:
                    record_id = existing_rows[0].get("id")
                    self.supabase.table("gem_scout_candidates").update({
                        "address": candidate.address,
                        "latitude": candidate.latitude,
                        "longitude": candidate.longitude,
                        "website_url": candidate.website_url,
                        "contact_phone": candidate.contact_phone,
                        "source": candidate.source,
                        "source_url": candidate.source_url,
                        "google_place_id": candidate.google_place_id or existing_rows[0].get("google_place_id"),
                        "confidence": candidate.confidence,
                        "verification_status": candidate.verification_status,
                        "last_seen_at": now,
                    }).eq("id", record_id).execute()
                    updated += 1
                else:
                    self.supabase.table("gem_scout_candidates").insert(asdict(candidate) | {
                        "last_seen_at": now,
                    }).execute()
                    inserted += 1
            except Exception as exc:
                raise RuntimeError(f"Gem Scout Supabase write failed: {exc}") from exc

            accepted.append(candidate)

        source_counts: Dict[str, int] = {
            "DISCOVERY_SCANNED": scanned,
            "GOOGLE_DISCOVERED": len(google_candidates),
            "OPENSTREETMAP_SCANNED": max(0, scanned - len(google_candidates)),
            "GOOGLE_VERIFIED_RICH": len(accepted),
        }
        for reason, count in rejection_counts.items():
            source_counts[f"REJECTED_{reason.upper()}"] = count

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
