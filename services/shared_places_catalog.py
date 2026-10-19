"""Shared Supabase catalogue for validated, provider-neutral place records.

This module is additive. It does not replace community_places, gem_scout_candidates,
or destination_catalog_entries. Call it from existing route handlers after the SQL
migration has been applied. The caller supplies the already-configured Supabase client.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

TABLE = "shared_places_catalog"
PUBLIC_STATUSES = ("published",)

# Exact generic labels are not useful public place names on their own.
# Exact matching avoids rejecting legitimate names such as "Central Park".
_GENERIC_PLACE_NAMES = {
    "park",
    "garden",
    "temple",
    "church",
    "mosque",
    "museum",
    "market",
    "mall",
    "hotel",
    "restaurant",
    "beach",
    "school",
    "hospital",
    "station",
    "monument",
    "attraction",
    "place",
    "building",
    "unnamed",
    "unknown",
    "no name",
    "poi",
    "point of interest",
}


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def is_generic_place_name(value: Any) -> bool:
    """Return True when a name is only a generic feature type, not a place name."""
    cleaned = _clean(value)
    if not cleaned:
        return True
    normalized = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
    return normalized in _GENERIC_PLACE_NAMES


def _valid_coordinate_pair(latitude: Any, longitude: Any) -> bool:
    if latitude is None or longitude is None:
        return latitude is None and longitude is None
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def normalize_place(record: Dict[str, Any], *, publish_if_valid: bool = True) -> Optional[Dict[str, Any]]:
    """Normalize a provider result. Return None if it lacks safe identity essentials.

    Public publishing requires name, city, source provider, and either a stable
    provider place ID or a valid coordinate pair. Weak records are rejected rather
    than automatically published. This does not verify provider licensing.
    """
    name = _clean(record.get("name"))
    city = _clean(record.get("city"))
    category = _clean(record.get("category")) or "Other"
    provider = _clean(record.get("source_provider") or record.get("provider"))
    place_id = _clean(record.get("source_place_id") or record.get("place_id") or record.get("osm_id"))
    latitude = record.get("latitude", record.get("lat"))
    longitude = record.get("longitude", record.get("lng", record.get("lon")))

    if not name or is_generic_place_name(name) or not city or not provider:
        # Do not automatically publish weak records whose only name is a generic
        # feature type (for example, "Park"). Existing database rows are untouched.
        return None
    if not _valid_coordinate_pair(latitude, longitude):
        return None
    has_coordinates = latitude is not None and longitude is not None
    if not place_id and not has_coordinates:
        return None

    try:
        lat_value = float(latitude) if latitude is not None else None
        lon_value = float(longitude) if longitude is not None else None
    except (TypeError, ValueError):
        return None

    metadata_source = record.get("metadata")
    metadata_source = metadata_source if isinstance(metadata_source, dict) else {}
    metadata = {
        key: value
        for key in ("attribution_required", "database_license", "provider_variant", "discovery_scope")
        if (value := _clean(metadata_source.get(key)))
    }

    # Avoid persisting large/raw provider payloads. Only allow the named fields below.
    clean = {
        "name": name,
        "category": category,
        "description": _clean(record.get("description")),
        "city": city,
        "state_region": _clean(record.get("state_region") or record.get("state")),
        "country": _clean(record.get("country")),
        "country_code": _clean(record.get("country_code")),
        "address": _clean(record.get("address")),
        "latitude": lat_value,
        "longitude": lon_value,
        "source_provider": provider.lower(),
        "source_place_id": place_id,
        "external_url": _clean(record.get("external_url") or record.get("website") or record.get("url")),
        "image_url": _clean(record.get("image_url")),
        "image_source_url": _clean(record.get("image_source_url") or record.get("image_page_url")),
        "image_attribution": _clean(record.get("image_attribution") or record.get("attribution")),
        "image_license": _clean(record.get("image_license") or record.get("license")),
        "publication_status": "published" if publish_if_valid else "pending_review",
        "metadata": metadata,
    }
    return {key: value for key, value in clean.items() if value is not None}


def upsert_validated_place(supabase: Any, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Upsert a validated place without overriding an existing moderation status.

    Call only for providers whose terms permit storage of the fields being written.
    A newly discovered, validated record can be published; an existing non-published
    record keeps its current status when refreshed.
    """
    normalized = normalize_place(record, publish_if_valid=True)
    if not normalized:
        return None

    provider = normalized["source_provider"]
    source_id = normalized.get("source_place_id")
    if source_id:
        try:
            existing = (
                supabase.table(TABLE)
                .select("id,publication_status")
                .eq("source_provider", provider)
                .eq("source_place_id", source_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            if existing:
                previous_status = _clean(existing[0].get("publication_status"))
                if previous_status and previous_status.lower() != "published":
                    normalized["publication_status"] = previous_status
        except Exception:
            # Let the write return the useful database error if the table/schema is absent.
            pass

    payload = dict(normalized)
    payload["last_seen_at"] = datetime.now(timezone.utc).isoformat()
    if source_id:
        result = supabase.table(TABLE).upsert(
            payload, on_conflict="source_provider,source_place_id"
        ).execute()
    else:
        # Without stable provider ID, insert only; coordinate/name dedupe is deliberately
        # not asserted because that can incorrectly merge nearby distinct businesses.
        result = supabase.table(TABLE).insert(payload).execute()
    rows = getattr(result, "data", None) or []
    return rows[0] if rows else None


def search_shared_places(
    supabase: Any,
    *,
    city: str = "",
    query: str = "",
    category: str = "All",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Search published records only. Safe to use as an additive source in existing feeds."""
    if not supabase:
        return []
    limit = max(1, min(int(limit), 100))
    safe_query = re.sub(r"[,()%*\\\"']", " ", query or "").strip()
    try:
        request = (
            supabase.table(TABLE).select("*")
            .eq("publication_status", "published")
        )
        if city.strip():
            request = request.ilike("city", city.strip())
        if category.strip() and category.strip().lower() != "all":
            request = request.ilike("category", category.strip())
        if safe_query:
            request = request.or_(
                f"name.ilike.%{safe_query}%,address.ilike.%{safe_query}%,"
                f"category.ilike.%{safe_query}%,description.ilike.%{safe_query}%,city.ilike.%{safe_query}%"
            )
        # Fetch extra rows so generic names do not consume the entire result limit.
        fetch_limit = min(100, max(limit, limit * 3))
        response = request.order("last_seen_at", desc=True).limit(fetch_limit).execute()
        records = list(getattr(response, "data", None) or [])
        # Filter generic-only labels from every shared-catalogue consumer, while
        # leaving the stored rows intact for future review or correction.
        return [
            record for record in records
            if not is_generic_place_name(record.get("name"))
        ][:limit]
    except Exception as exc:
        # Preserve existing live-search behavior if the migration has not been deployed yet.
        print(f"[Shared Places Catalogue notice] {exc}")
        return []


def merge_places(*groups: Iterable[Dict[str, Any]], limit: int = 50) -> List[Dict[str, Any]]:
    """Merge rows without assuming IDs from different providers are globally equivalent."""
    merged: List[Dict[str, Any]] = []
    seen_provider_ids = set()
    seen_fallback = set()
    for group in groups:
        for row in group or []:
            provider = _clean(row.get("source_provider") or row.get("provider"))
            place_id = _clean(row.get("source_place_id") or row.get("place_id"))
            if provider and place_id:
                key = (provider.lower(), place_id)
                if key in seen_provider_ids:
                    continue
                seen_provider_ids.add(key)
            else:
                name = (_clean(row.get("name")) or "").lower()
                city = (_clean(row.get("city")) or "").lower()
                address = (_clean(row.get("address")) or "").lower()
                key = (name, city, address)
                if key != ("", "", "") and key in seen_fallback:
                    continue
                if key != ("", "", ""):
                    seen_fallback.add(key)
            merged.append(dict(row))
            if len(merged) >= max(1, limit):
                return merged
    return merged
