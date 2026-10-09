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
FSQ_SERVICE_API_KEY = os.environ.get("FSQ_SERVICE_API_KEY", "").strip()
FSQ_PLACES_BASE_URL = "https://places-api.foursquare.com"
FSQ_PLACES_API_VERSION = "2025-06-17"


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


CATEGORY_FOURSQUARE_QUERIES: Dict[str, str] = {
    "Pharmacy / Chemist": "pharmacy",
    "Food": "restaurant",
    "Bar & Restaurant": "restaurant",
    "Chai & Quick Bites": "cafe",
    "Market, Bazaar & Mall": "shopping mall",
    "Clothing & Fashion": "clothing store",
    "Hotel & Stay": "hotel",
    "Heritage & Sight": "tourist attraction",
    "Picnic Spot & Landscape": "park",
    "Barber & Salon": "salon",
    "Electronics & Mobile": "electronics store",
}

FOURSQUARE_QUERY_VARIANTS: Dict[str, List[str]] = {
    "Pharmacy / Chemist": ["pharmacy", "chemist", "medical store"],
    "Food": ["restaurant", "cafe", "food"],
    "Bar & Restaurant": ["restaurant", "bar", "pub"],
    "Chai & Quick Bites": ["cafe", "tea shop", "bakery"],
    "Market, Bazaar & Mall": ["shopping mall", "supermarket", "market"],
    "Clothing & Fashion": ["clothing store", "fashion store", "boutique"],
    "Hotel & Stay": ["hotel", "resort", "guest house"],
    "Heritage & Sight": ["tourist attraction", "museum", "landmark"],
    "Picnic Spot & Landscape": ["park", "garden", "picnic spot"],
    "Barber & Salon": ["barber", "salon", "beauty salon"],
    "Electronics & Mobile": ["electronics store", "mobile phone shop", "computer shop"],
}



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
    fsq_place_id: str
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
    """Foursquare-first evidence gate for Community Gem candidates.

    Foursquare Places is the primary discovery and verification provider.
    OpenStreetMap is an optional fallback for candidate discovery only; any
    OSM candidate must still match a Foursquare place with the required evidence
    before it enters the candidate table. Google is not called by this module.
    The returned Foursquare photo/tip payload is evaluated in memory and is not
    copied into Supabase; only the fields required by the existing candidate
    queue plus the stable Foursquare place ID are persisted.
    """

    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client

    async def _geocode_city(self, city: str) -> Optional[Tuple[float, float, str]]:
        """Resolve city coordinates with Nominatim for the optional OSM fallback."""
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
                print(f"[Gem Scout Geocode] Nominatim HTTP {response.status_code}: {response.text[:250]}")
        except Exception as exc:
            print(f"[Gem Scout Geocode] Nominatim unavailable: {exc}")
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
                # OSM address data is optional here because Foursquare verification
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
    def _foursquare_place_matches_category(place: Dict[str, Any], canonical_category: str) -> bool:
        categories = place.get("categories") or []
        raw = " ".join(
            str(item.get("name") or item.get("short_name") or "")
            if isinstance(item, dict) else str(item)
            for item in categories
        ).lower()
        checks = {
            "Pharmacy / Chemist": ("pharmacy", "chemist", "drugstore", "medical store"),
            "Barber & Salon": ("barber", "hair", "beauty", "salon"),
            "Bar & Restaurant": ("restaurant", "bar", "pub", "nightclub", "night club"),
            "Chai & Quick Bites": ("cafe", "coffee", "tea", "bakery", "snack", "fast food"),
            "Hotel & Stay": ("hotel", "resort", "hostel", "guest house", "motel"),
            "Market, Bazaar & Mall": ("shopping mall", "department store", "supermarket", "market"),
            "Clothing & Fashion": ("clothing", "fashion", "shoe", "boutique", "apparel"),
            "Electronics & Mobile": ("electronics", "mobile phone", "computer", "electronic store"),
            "Heritage & Sight": ("tourist attraction", "museum", "landmark", "historic"),
            "Picnic Spot & Landscape": ("park", "garden", "picnic", "campground"),
        }
        wanted = checks.get(canonical_category)
        # Category metadata can be absent in some response tiers; the targeted
        # query remains useful, but the evidence gate below is still mandatory.
        if not raw:
            return True
        return not wanted or any(token in raw for token in wanted)

    @staticmethod
    def _foursquare_location(place: Dict[str, Any]) -> Dict[str, Any]:
        return place.get("location") or {}

    @classmethod
    def _foursquare_address(cls, place: Dict[str, Any]) -> str:
        location = cls._foursquare_location(place)
        formatted = str(location.get("formatted_address") or "").strip()
        if formatted:
            return formatted
        parts = [
            location.get("address"),
            location.get("locality"),
            location.get("region"),
            location.get("postcode"),
            location.get("country"),
        ]
        return ", ".join(str(part).strip() for part in parts if str(part or "").strip())

    @staticmethod
    def _foursquare_evidence(place: Dict[str, Any], match_score: float = 1.0) -> Dict[str, Any]:
        stats = place.get("stats") or {}
        photos = place.get("photos") or []
        tips = place.get("tips") or []
        rating_count = stats.get("total_ratings") or 0
        tip_count = max(int(stats.get("total_tips") or 0), len(tips))
        photo_count = max(int(stats.get("total_photos") or 0), len(photos))
        location = place.get("location") or {}
        return {
            "rating": place.get("rating"),
            "rating_count": int(rating_count or 0),
            # Foursquare's text feedback is returned as user tips rather than
            # Google-style written reviews; it serves as the written-feedback signal.
            "review_count": int(tip_count),
            "phone": str(place.get("tel") or "").strip(),
            "photo_count": int(photo_count),
            "photo_urls_present": bool(photos),
            "formatted_address_present": bool(
                str(location.get("formatted_address") or "").strip()
                or str(location.get("address") or "").strip()
            ),
            "place_id_present": bool(str(place.get("fsq_place_id") or "").strip()),
            "match_score": round(float(match_score), 3),
        }

    @classmethod
    def _foursquare_row(cls, place: Dict[str, Any], city: str, match_score: float = 1.0) -> Dict[str, Any]:
        location = cls._foursquare_location(place)
        evidence = cls._foursquare_evidence(place, match_score)
        categories = place.get("categories") or []
        category_name = ", ".join(
            str(item.get("name") or item.get("short_name") or "")
            for item in categories if isinstance(item, dict)
            if item.get("name") or item.get("short_name")
        )
        lat = place.get("latitude")
        lng = place.get("longitude")
        if lat is None:
            lat = location.get("latitude")
        if lng is None:
            lng = location.get("longitude")
        return {
            "name": str(place.get("name") or "").strip(),
            "address": cls._foursquare_address(place),
            "city": city.strip(),
            "latitude": float(lat) if lat is not None else None,
            "longitude": float(lng) if lng is not None else None,
            "website_url": str(place.get("website") or "").strip(),
            "contact_phone": str(place.get("tel") or "").strip(),
            "source_url": str(place.get("link") or "").strip(),
            "fsq_place_id": str(place.get("fsq_place_id") or "").strip(),
            "source_category": category_name,
            "rating": evidence["rating"],
            "rating_count": evidence["rating_count"],
            "review_count": evidence["review_count"],
            "photo_count": evidence["photo_count"],
            "foursquare_place": place,
            "evidence": evidence,
        }

    async def _foursquare_search(self, city: str, query: str, limit: int) -> List[Dict[str, Any]]:
        """Call the current Foursquare Places Search API using a Service API Key."""
        if not FSQ_SERVICE_API_KEY:
            print("[Gem Scout Foursquare] FSQ_SERVICE_API_KEY is not configured")
            return []
        endpoint = f"{FSQ_PLACES_BASE_URL}/places/search"
        params = {
            "query": query,
            "near": f"{city}, India",
            "limit": str(max(1, min(int(limit), 50))),
            "sort": "RATING",
            "tel_format": "NATIONAL",
        }
        headers = {
            "Authorization": f"Bearer {FSQ_SERVICE_API_KEY}",
            "X-Places-Api-Version": FSQ_PLACES_API_VERSION,
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=6)) as client:
                response = await client.get(endpoint, params=params, headers=headers)
            if response.status_code != 200:
                print(f"[Gem Scout Foursquare] HTTP {response.status_code}: {response.text[:500]}")
                return []
            data = response.json() or {}
            results = data.get("results") or []
            return [row for row in results if isinstance(row, dict)]
        except Exception as exc:
            print(f"[Gem Scout Foursquare] Search failed for query={query!r}: {exc}")
            return []

    async def _foursquare_candidates(
        self, city: str, canonical_category: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Discover a small user-requested category set through Foursquare."""
        variants = FOURSQUARE_QUERY_VARIANTS.get(
            canonical_category,
            [CATEGORY_FOURSQUARE_QUERIES.get(canonical_category, canonical_category.lower())],
        )
        per_query = min(max(limit * 2, 10), 30)
        seen_ids = set()
        seen_names = set()
        rows: List[Dict[str, Any]] = []
        for term in variants[:3]:
            places = await self._foursquare_search(city, term, per_query)
            for place in places:
                place_id = str(place.get("fsq_place_id") or "").strip()
                name = str(place.get("name") or "").strip()
                if not place_id or not name or not self._foursquare_place_matches_category(place, canonical_category):
                    continue
                normalized_name = _normalize(name)
                identity = place_id or normalized_name
                if identity in seen_ids or normalized_name in seen_names:
                    continue
                row = self._foursquare_row(place, city)
                if row["latitude"] is None or row["longitude"] is None or not row["address"]:
                    continue
                row["osm_id"] = ""
                rows.append(row)
                seen_ids.add(identity)
                seen_names.add(normalized_name)
        rows.sort(
            key=lambda row: (
                int(row.get("review_count") or 0),
                int(row.get("rating_count") or 0),
                int(row.get("photo_count") or 0),
                float(row.get("rating") or 0.0),
            ),
            reverse=True,
        )
        print(f"[Gem Scout Foursquare Discovery] discovered={len(rows)} category={canonical_category!r} city={city!r}")
        return rows[: max(limit * 5, limit)]

    async def _foursquare_verify(
        self, candidate: Dict[str, Any], city: str, canonical_category: str
    ) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
        """Match a secondary OSM candidate to a live Foursquare place."""
        name = str(candidate.get("name") or "").strip()
        if not name or not FSQ_SERVICE_API_KEY:
            return None, 0.0, {"reason": "name_or_foursquare_key_missing"}
        category_hint = CATEGORY_FOURSQUARE_QUERIES.get(canonical_category, canonical_category.lower())
        results = await self._foursquare_search(city, f"{name} {category_hint}", 5)
        best_place: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for place in results:
            display = str(place.get("name") or "").strip()
            if not self._foursquare_place_matches_category(place, canonical_category):
                continue
            name_score = _name_similarity(name, display)
            plat = place.get("latitude")
            plng = place.get("longitude")
            if plat is None or plng is None:
                continue
            distance_m = _distance_m(
                float(candidate["latitude"]), float(candidate["longitude"]), float(plat), float(plng)
            )
            distance_score = max(0.0, 1.0 - min(distance_m, 7500.0) / 7500.0)
            score = name_score * 0.70 + distance_score * 0.30
            if score > best_score:
                best_score, best_place = score, place
        if not best_place or best_score < 0.68:
            return None, best_score, {"reason": "weak_foursquare_match", "match_score": round(best_score, 3)}
        evidence = self._foursquare_evidence(best_place, best_score)
        return best_place, best_score, evidence

    @staticmethod
    def _rich_evidence_gate(evidence: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Accept only records with a location, rating activity, text feedback, phone and image evidence."""
        rating = evidence.get("rating")
        rating_count = int(evidence.get("rating_count") or 0)
        tips_count = int(evidence.get("review_count") or 0)
        phone = str(evidence.get("phone") or "").strip()
        photo_count = int(evidence.get("photo_count") or 0)
        address_present = bool(evidence.get("formatted_address_present"))
        place_id_present = bool(evidence.get("place_id_present"))
        match_score = float(evidence.get("match_score") or 0.0)

        if rating is None:
            return False, 0.0, "missing_rating"
        if rating_count < 1:
            return False, 0.0, "no_rating_activity"
        if tips_count < 1:
            return False, 0.0, "no_written_feedback"
        if not phone:
            return False, 0.0, "missing_phone"
        if photo_count < 1:
            return False, 0.0, "missing_photo"
        if not evidence.get("photo_urls_present"):
            return False, 0.0, "photo_asset_not_returned"
        if not address_present:
            return False, 0.0, "missing_address"
        if not place_id_present:
            return False, 0.0, "missing_foursquare_place_id"
        if match_score < 0.68:
            return False, 0.0, "weak_match"

        # Foursquare's rating scale is different from Google's. Reward ratings
        # on the upper end of its scale while still retaining the evidence counts.
        score = 0.30
        score += 0.14 if float(rating) >= 8.0 else 0.08 if float(rating) >= 6.0 else 0.00
        score += 0.07 if rating_count >= 10 else 0.03
        score += 0.07 if tips_count >= 5 else 0.03
        score += 0.10  # phone
        score += 0.10  # returned photo assets
        score += 0.08 if address_present else 0.00
        score += min(0.10, max(0.0, match_score) * 0.10)
        return True, round(min(0.99, score), 3), "foursquare_rich_evidence"

    async def run_community_scout(
        self,
        city: str,
        category: str,
        quantity: int = 10,
        verify_google: bool = True,
    ) -> ScoutRunResult:
        # verify_google remains in the signature for backwards compatibility with
        # main.py callers, but V5 never sends requests to Google.
        city = city.strip()
        requested_category = category.strip() or "Food"
        quantity = max(1, min(int(quantity), 50))
        profile = _resolve_profile(requested_category)
        canonical_category = profile["canonical"]
        if verify_google:
            print("[Gem Scout] V5 uses Foursquare; the legacy verify_google flag is ignored.")

        fsq_candidates = await self._foursquare_candidates(city, canonical_category, quantity)
        candidates = list(fsq_candidates)
        discovery_source = "FOURSQUARE"

        # Optional OSM fallback only expands the candidate pool. Any OSM result
        # must still be verified against Foursquare before it can be saved.
        if len(candidates) < quantity:
            osm_candidates = await self._osm_candidates(city, profile, quantity)
            existing_names = {_normalize(row.get("name")) for row in candidates}
            for row in osm_candidates:
                key = _normalize(row.get("name"))
                if key and key not in existing_names:
                    candidates.append(row)
                    existing_names.add(key)
            if osm_candidates:
                discovery_source = "FOURSQUARE+OPENSTREETMAP" if fsq_candidates else "OPENSTREETMAP"

        scanned = len(candidates)
        if scanned == 0:
            print(f"[Gem Scout] No candidates discovered for city={city!r}, category={requested_category!r}")
        accepted: List[ScoutCandidate] = []
        skipped_duplicates = 0
        inserted = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()
        candidates.sort(
            key=lambda row: (
                bool(row.get("address")),
                bool(row.get("contact_phone")),
                bool(row.get("website_url")),
                int(row.get("review_count") or 0),
                int(row.get("rating_count") or 0),
                int(row.get("photo_count") or 0),
                float(row.get("rating") or 0.0),
            ),
            reverse=True,
        )

        rejection_counts: Dict[str, int] = {}
        for row in candidates:
            if len(accepted) >= quantity:
                break
            place = row.get("foursquare_place")
            evidence = row.get("evidence")
            match_score = float((evidence or {}).get("match_score") or 1.0)
            if not place:
                place, match_score, evidence = await self._foursquare_verify(row, city, canonical_category)
            if not place:
                reason = str((evidence or {}).get("reason") or "foursquare_match_not_found")
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            verified_row = self._foursquare_row(place, city, match_score)
            evidence = verified_row.get("evidence") or evidence or {}
            passed, evidence_score, evidence_reason = self._rich_evidence_gate(evidence)
            if not passed:
                rejection_counts[evidence_reason] = rejection_counts.get(evidence_reason, 0) + 1
                continue

            fsq_place_id = str(place.get("fsq_place_id") or "").strip()
            if not fsq_place_id:
                rejection_counts["missing_foursquare_place_id"] = rejection_counts.get("missing_foursquare_place_id", 0) + 1
                continue
            # Use the verified provider's current address/contact/coordinates. The
            # image and text-feedback payloads are used only in this run's gate.
            name = str(verified_row.get("name") or row.get("name") or "").strip()
            address = str(verified_row.get("address") or row.get("address") or "").strip()
            latitude = verified_row.get("latitude") if verified_row.get("latitude") is not None else row.get("latitude")
            longitude = verified_row.get("longitude") if verified_row.get("longitude") is not None else row.get("longitude")
            phone = str(verified_row.get("contact_phone") or row.get("contact_phone") or "").strip()
            website = str(verified_row.get("website_url") or row.get("website_url") or "").strip()
            if not name or not address or latitude is None or longitude is None:
                rejection_counts["missing_core_place_fields"] = rejection_counts.get("missing_core_place_fields", 0) + 1
                continue

            dedupe_key = "|".join([
                _normalize(city),
                _normalize(canonical_category),
                _normalize(fsq_place_id),
            ])
            candidate = ScoutCandidate(
                dedupe_key=dedupe_key,
                city=city,
                category=canonical_category,
                name=name,
                address=address,
                latitude=float(latitude),
                longitude=float(longitude),
                website_url=website,
                contact_phone=phone,
                source=f"{discovery_source}+FOURSQUARE_VERIFIED",
                source_url=str(verified_row.get("source_url") or row.get("source_url") or ""),
                google_place_id="",
                fsq_place_id=fsq_place_id,
                confidence=round(min(0.995, evidence_score), 3),
                verification_status="foursquare_verified_rich_evidence",
                status="candidate",
                discovered_at=now,
            )

            try:
                existing = (
                    self.supabase.table("gem_scout_candidates")
                    .select("id,status,confidence,google_place_id,fsq_place_id")
                    .eq("dedupe_key", dedupe_key)
                    .limit(1)
                    .execute()
                )
                existing_rows = existing.data or []
                if existing_rows:
                    record_id = existing_rows[0].get("id")
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
                        "fsq_place_id": candidate.fsq_place_id,
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
            "FOURSQUARE_DISCOVERED": len(fsq_candidates),
            "OPENSTREETMAP_SCANNED": max(0, scanned - len(fsq_candidates)),
            "FOURSQUARE_VERIFIED_RICH": len(accepted),
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
