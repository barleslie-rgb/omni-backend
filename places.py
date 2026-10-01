import os
import httpx
from fastapi import APIRouter, HTTPException
from supabase import create_client

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None

@router.post("/api/v1/sync-city-gems")
async def sync_city_gems(city: str, category: str = "pharmacy"):
    if not GOOGLE_PLACES_KEY:
        raise HTTPException(status_code=500, detail="Google Places API key not configured on server.")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized on backend.")
    
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={category}+in+{city}&key={GOOGLE_PLACES_KEY}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch from Google Places API")
        
        data = response.json()
        results = data.get("results", [])
        
        inserted_count = 0
        for place in results:
            name = place.get("name")
            address = place.get("formatted_address")
            lat = place.get("geometry", {}).get("location", {}).get("lat")
            lon = place.get("geometry", {}).get("location", {}).get("lng")
            
            # Check if place already exists in Supabase
            existing = supabase.table("community_places").select("id").eq("name", name).execute()
            if not existing.data:
                supabase.table("community_places").insert({
                    "name": name,
                    "city": city.title(),
                    "category": "Pharmacy / Chemist" if "pharmacy" in category.lower() else "Diner & Seafood",
                    "address": address or f"{city} Hub",
                    "latitude": lat,
                    "longitude": lon,
                    "contributor_name": "Google Verified Scout",
                    "upvotes": 12,
                    "must_try_tip": "Auto-synced from Google Places Registry.",
                    "endorsement_tags": ["🕒 Verified Location", "💳 Digital Pay"],
                }).execute()
                inserted_count += 1
                
        return {"status": "success", "city": city, "imported_places": inserted_count}