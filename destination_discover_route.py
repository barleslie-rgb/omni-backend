@app.get("/api/v1/destination-discover")
async def destination_discover(
    q: str = Query("", max_length=160),
    country: str = Query("", max_length=100),
    country_code: str = Query("", max_length=3),
    region: str = Query("", max_length=120),
    category: str = Query("", max_length=80),
    destination_type: str = Query("", max_length=40),
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(20, ge=1, le=50),
):
    """Search published curated destination entries; never invent results."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Destination catalogue storage is not configured.")
    try:
        query = supabase.table("destination_catalog_entries").select(
            "id,slug,name,alternate_names,destination_type,category,description,city,state_region,country,country_code,latitude,longitude,address,image_url,source_name,source_url,source_license,source_attribution,verified_at",
            count="exact"
        ).eq("is_published", True)
        if country.strip(): query = query.ilike("country", f"%{country.strip()}%")
        if country_code.strip(): query = query.ilike("country_code", country_code.strip().upper())
        if region.strip(): query = query.ilike("state_region", f"%{region.strip()}%")
        if category.strip() and category.strip().lower() != "all": query = query.ilike("category", f"%{category.strip()}%")
        if destination_type.strip(): query = query.ilike("destination_type", f"%{destination_type.strip()}%")
        if q.strip():
            term = re.sub(r"[,()]", " ", q.strip())[:160]
            pattern = f"%{term}%"
            query = query.or_(f"name.ilike.{pattern},city.ilike.{pattern},state_region.ilike.{pattern},country.ilike.{pattern}")
        start=(page-1)*page_size
        response=query.order("name").range(start,start+page_size-1).execute()
        rows=response.data or []
        return {"status":"success","query":q.strip(),"filters":{"country":country.strip(),"country_code":country_code.strip().upper(),"region":region.strip(),"category":category.strip(),"destination_type":destination_type.strip()},"page":page,"page_size":page_size,"total":getattr(response,"count",None) if getattr(response,"count",None) is not None else len(rows),"results":rows,"source":"Omni TouristOS curated destination catalogue","attribution_required":True}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[destination-discover] catalogue query failed: {exc}")
        raise HTTPException(status_code=503, detail="Destination catalogue is temporarily unavailable.")
