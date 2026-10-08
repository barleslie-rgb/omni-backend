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


CATEGORY_PROFILES: Dict[str, Dict[str, str]] = {
    "pharmacy": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "chemist": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "food": {
        "canonical": "Food",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|cafe|fast_food|food_court|ice_cream",i];',
    },
    "restaurant": {
        "canonical": "Bar & Restaurant",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"~"restaurant|fast_food",i];',
    },
    "cafe": {
        "canonical": "Chai & Quick Bites",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="cafe"];',
    },
    "mall": {
        "canonical": "Market, Bazaar & Mall",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"="mall"];nwr(around:{radius},{lat},{lng})["shop"="department_store"];nwr(around:{radius},{lat},{lng})["shop"="supermarket"];',
    },
    "shopping": {
        "canonical": "Market, Bazaar & Mall",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"mall|department_store|supermarket|convenience|clothes|shoes|fashion",i];',
    },
    "clothing": {
        "canonical": "Clothing & Fashion",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"clothes|fashion|shoes",i];',
    },
    "dresses": {
        "canonical": "Clothing & Fashion",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"clothes|fashion|fabric|tailor",i];',
    },
    "hotel": {
        "canonical": "Hotel & Stay",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"hotel|guest_house|hostel|motel",i];',
    },
    "hotels": {
        "canonical": "Hotel & Stay",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"hotel|guest_house|hostel|motel",i];',
    },
    "pharmacy chemist": {
        "canonical": "Pharmacy / Chemist",
        "query": 'nwr(around:{radius},{lat},{lng})["amenity"="pharmacy"];nwr(around:{radius},{lat},{lng})["shop"="chemist"];',
    },
    "attractions": {
        "canonical": "Heritage & Sight",
        "query": 'nwr(around:{radius},{lat},{lng})["tourism"~"attraction|museum|viewpoint|zoo|theme_park",i];nwr(around:{radius},{lat},{lng})["historic"];',
    },
    "parks": {
        "canonical": "Picnic Spot & Landscape",
        "query": 'nwr(around:{radius},{lat},{lng})["leisure"="park"];nwr(around:{radius},{lat},{lng})["leisure"="garden"];',
    },
    "barber": {
        "canonical": "Barber & Salon",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"hairdresser|beauty",i];',
    },
    "salon": {
        "canonical": "Barber & Salon",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"hairdresser|beauty",i];',
    },
    "electronics": {
        "canonical": "Electronics & Mobile",
        "query": 'nwr(around:{radius},{lat},{lng})["shop"~"electronics|mobile_phone|computer",i];',
    },
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
        if key in raw or raw in key:
            return profile
    if any(token in raw for token in ("pharmacy", "chemist", "medical store")):
        return CATEGORY_PROFILES["pharmacy"]
    if any(token in raw for token in ("hotel", "stay", "resort", "hostel")):
        return CATEGORY_PROFILES["hotel"]
    if any(token in raw for token in ("dress", "clothing", "fashion", "apparel")):
        return CATEGORY_PROFILES["clothing"]
    if any(token in raw for token in ("mall", "shopping", "market")):
        return CATEGORY_PROFILES["shopping"]
    if any(token in raw for token in ("food", "restaurant", "eat", "dining")):
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
    """Community Gem discovery engine.

    V1 intentionally persists open-data-derived candidate records. Google is
    used as live verification/enrichment and only the Google place ID is stored.
    This keeps the scout useful without turning Google Places responses into a
    permanent scraped database.
    """

    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client

    async def _geocode_city(self, city: str) -> Optional[Tuple[float, float, str]]:
        params = {
            "q": f"{city}, India",
            "format": "jsonv2",
            "limit": "3",
            "accept-language": "en",
        }
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=6)) as client:
            response = await client.get(NOMINATIM_API_URL, params=params, headers=headers)
        if response.status_code != 200:
            return None
        rows = response.json() or []
        if not rows:
            return None
        chosen = rows[0]
        return float(chosen["lat"]), float(chosen["lon"]), str(chosen.get("display_name") or city)

    async def _osm_candidates(self, city: str, profile: Dict[str, str], limit: int, radius_m: int = 15000) -> List[Dict[str, Any]]:
        resolved = await self._geocode_city(city)
        if not resolved:
            return []
        lat, lng, _ = resolved
        statement = profile["query"].format(radius=radius_m, lat=lat, lng=lng)
        query = f"[out:json][timeout:40];({statement});out center tags;"
        headers = {"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(50, connect=10)) as client:
            response = await client.post(
                OVERPASS_API_URL,
                data={"data": query},
                headers=headers,
            )
        if response.status_code != 200:
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
            address = ", ".join(str(x).strip() for x in address_parts if str(x or "").strip()) or city
            osm_id = f"osm:{element.get('type', '')}/{element.get('id', '')}"
            source_url = f"https://www.openstreetmap.org/?mlat={item_lat}&mlon={item_lng}#map=17/{item_lat}/{item_lng}"
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
            if len(rows) >= min(max(limit * 3, limit), 60):
                break
        return rows

    async def _google_verify(self, candidate: Dict[str, Any], city: str) -> Tuple[str, float]:
        if not GOOGLE_PLACES_API_KEY:
            return "", 0.0
        endpoint = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
        ])
        payload = {
            "textQuery": f"{candidate['name']} {city}",
            "pageSize": 5,
            "languageCode": "en",
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": field_mask,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(12, connect=4)) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                return "", 0.0
            places = (response.json() or {}).get("places") or []
            best_id, best_score = "", 0.0
            for place in places:
                display = (place.get("displayName") or {}).get("text") or ""
                loc = place.get("location") or {}
                plat, plng = loc.get("latitude"), loc.get("longitude")
                name_score = _name_similarity(candidate["name"], display)
                dist_score = 0.0
                if plat is not None and plng is not None:
                    distance = _distance_m(
                        float(candidate["latitude"]), float(candidate["longitude"]),
                        float(plat), float(plng),
                    )
                    dist_score = max(0.0, 1.0 - min(distance, 5000.0) / 5000.0)
                score = name_score * 0.75 + dist_score * 0.25
                if score > best_score:
                    best_score = score
                    best_id = str(place.get("id") or "")
            return (best_id if best_score >= 0.60 else "", best_score)
        except Exception:
            return "", 0.0

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

        candidates = await self._osm_candidates(city, profile, quantity)
        scanned = len(candidates)
        accepted: List[ScoutCandidate] = []
        skipped_duplicates = 0
        inserted = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()

        # Prefer records with addresses and official websites/phones.
        candidates.sort(
            key=lambda row: (
                bool(row.get("address") and row.get("address") != city),
                bool(row.get("website_url")),
                bool(row.get("contact_phone")),
            ),
            reverse=True,
        )

        for row in candidates:
            if len(accepted) >= quantity:
                break
            google_place_id, google_score = ("", 0.0)
            if verify_google:
                google_place_id, google_score = await self._google_verify(row, city)

            completeness = 0.55
            if row.get("address") and row.get("address") != city:
                completeness += 0.12
            if row.get("website_url"):
                completeness += 0.07
            if row.get("contact_phone"):
                completeness += 0.06
            if google_place_id:
                completeness += 0.12
                completeness += min(0.06, google_score * 0.06)

            confidence = round(min(0.98, completeness), 3)
            verification_status = "google_matched" if google_place_id else "open_data_unverified"
            status = "candidate" if confidence >= 0.60 else "needs_review"
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
                source="OPENSTREETMAP",
                source_url=str(row.get("source_url") or ""),
                google_place_id=google_place_id,
                confidence=confidence,
                verification_status=verification_status,
                status=status,
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
                # A bad table/schema should not turn a partially completed scout
                # run into misleading success. Re-raise to make deployment logs loud.
                raise RuntimeError(f"Gem Scout Supabase write failed: {exc}") from exc

            accepted.append(candidate)

        source_counts = {"OPENSTREETMAP": len(accepted)}
        if any(c.google_place_id for c in accepted):
            source_counts["GOOGLE_VERIFIED"] = sum(1 for c in accepted if c.google_place_id)

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
