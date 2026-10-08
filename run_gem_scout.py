from __future__ import annotations

import argparse
import asyncio
import json
import os

from supabase import create_client

from gem_scout import GemScout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Omni TouristOS Community Gem Scout job")
    parser.add_argument("--city", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--quantity", type=int, default=10)
    parser.add_argument("--no-google", action="store_true")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_KEY must be configured")

    client = create_client(url, key)
    result = asyncio.run(
        GemScout(client).run_community_scout(
            city=args.city,
            category=args.category,
            quantity=args.quantity,
            verify_google=not args.no_google,
        )
    )
    print(json.dumps({
        "city": result.city,
        "requested_category": result.requested_category,
        "canonical_category": result.canonical_category,
        "requested_quantity": result.requested_quantity,
        "scanned": result.scanned,
        "inserted": result.inserted,
        "updated": result.updated,
        "candidates": [c.__dict__ for c in result.candidates],
        "source_counts": result.source_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
