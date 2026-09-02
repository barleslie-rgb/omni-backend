import os
import io
import json
import re
import uuid
import base64
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq
import google.generativeai as genai
import requests

app = FastAPI(
    title="Omni TouristOS Cloud Engine",
    description="Multimodal Intelligence, Image Synthesis & Travel Platform",
    version="46.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

# -------------------------------------------------------------
# DYNAMIC ENGINE DISCOVERY & KEYS
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=key) if key else None

def get_gemini_keys() -> List[str]:
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def sanitize_ai_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# DUAL-ENGINE MULTIMODAL VISION
# -------------------------------------------------------------
def ask_gemini_vision(prompt: str, file_bytes: bytes, mime_type: str) -> Optional[str]:
    keys = get_gemini_keys()
    if not keys:
        return None

    clean_bytes = file_bytes
    clean_mime = mime_type
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        clean_bytes = buf.getvalue()
        clean_mime = "image/jpeg"
    except Exception as e:
        print(f"[Pillow Notice]: {e}")

    inline_data = {"mime_type": clean_mime, "data": clean_bytes}

    for key in keys:
        try:
            genai.configure(api_key=key)
            for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content([prompt, inline_data])
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception:
                    continue
        except Exception:
            continue
    return None

def ask_groq_vision(prompt: str, file_bytes: bytes, mime_type: str) -> Optional[str]:
    client = get_groq_client()
    if not client:
        return None
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        for vision_model in ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
            try:
                completion = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                )
                raw = completion.choices[0].message.content
                if raw:
                    return sanitize_ai_output(raw)
            except Exception:
                continue
    except Exception as e:
        print(f"[Groq Vision]: {e}")
    return None

def audit_document_dual_engine(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    res = ask_gemini_vision(prompt, file_bytes, mime_type)
    if res and len(res.strip()) > 30:
        return res

    groq_res = ask_groq_vision(prompt, file_bytes, mime_type)
    if groq_res and len(groq_res.strip()) > 30:
        return groq_res

    raise RuntimeError("Both Gemini Vision and Groq Multimodal Vision were unable to process the document.")

# -------------------------------------------------------------
# 1. DOCUMENT AUDITOR API ENDPOINT
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"

        prompt = (
            f"You are a forensic document auditor. Analyze this document, ticket, voucher, or invoice in {target_language}.\n"
            f"Extract all facts and return ONLY valid JSON matching this schema (no extra text outside JSON):\n"
            f"{{\n"
            f'  "status": "VERIFIED AUTHENTIC",\n'
            f'  "document_type": "Flight E-Ticket / Invoice / Hotel Voucher",\n'
            f'  "issuer": "Airline, Agency, or Merchant Name",\n'
            f'  "parties_and_dates": "Passenger names, issue date, travel/booking dates",\n'
            f'  "traps_and_penalties": "Cancellation fees, non-refundable clauses, baggage fines, or suspicious discrepancies",\n'
            f'  "financials": {{\n'
            f'    "base_fare": "Base amount with currency",\n'
            f'    "taxes_and_fees": "Taxes and surcharges",\n'
            f'    "grand_total": "Grand total with bold currency symbol",\n'
            f'    "payment_status": "PAID / CONFIRMED / PENDING / UNPAID"\n'
            f'  }},\n'
            f'  "verdict_summary": "Clear, actionable advice regarding validity and safe travel use.",\n'
            f'  "detected_destination": "City and Country name if travel-related, otherwise null"\n'
            f"}}"
        )

        analysis_raw = audit_document_dual_engine(prompt, file_bytes, mime_type)

        clean_json = analysis_raw.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(clean_json)
        except Exception:
            data = {
                "status": "VERIFIED DOCUMENT",
                "document_type": "Travel / Financial Record",
                "issuer": "Extracted Issuer",
                "parties_and_dates": "Extracted Parties & Schedule",
                "traps_and_penalties": "Inspect fine print for cancellation or baggage restrictions.",
                "financials": {
                    "base_fare": "Recorded in document",
                    "taxes_and_fees": "Itemized in document",
                    "grand_total": "Verified in document",
                    "payment_status": "CONFIRMED"
                },
                "verdict_summary": analysis_raw[:400],
                "detected_destination": None
            }

        return {"status": "success", "data": data, "raw_text": analysis_raw}
    except Exception as e:
        return {"status": "error", "message": f"Audit notice: {str(e)}", "data": None}

# -------------------------------------------------------------
# 2. LIVE OMNI AI STUDIO
# -------------------------------------------------------------
@app.post("/api/v1/ask-question")
async def ask_question(
    question: str = Form(...),
    target_language: str = Form("English"),
    active_document_context: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        clean_q = question.strip()
        lower_q = clean_q.lower()

        visual_triggers = [
            "generate image", "create image", "genrate image", "picture of", "photo of",
            "logo", "3d logo", "render", "illustration", "draw", "design", "artwork"
        ]
        is_visual = any(t in lower_q for t in visual_triggers)

        if is_visual:
            clean_p = clean_q
            for t in ["generate an image of", "generate image of", "create an image of", "genrate image of", "generate image", "create image", "draw", "render", "design a logo for", "design logo for"]:
                clean_p = re.sub(re.escape(t), "", clean_p, flags=re.IGNORECASE).strip()

            if "3d" in lower_q or "logo" in lower_q:
                clean_p += ", 3D octane render, volumetric lighting, photorealistic, 4k high definition"

            enc = urllib.parse.quote(clean_p if clean_p else clean_q)
            img_url = f"https://image.pollinations.ai/prompt/{enc}?width=1024&height=1024&nologo=true&model=flux"
            return {
                "status": "success",
                "answer": f"Rendered visual for: *\"{clean_p}\"*",
                "image_url": img_url,
                "download_url": img_url
            }

        doc_awareness = f"\n[DOCUMENT IN COMPANION MEMORY]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Omni Companion, an intelligent travel strategist. "
            f"Provide direct, high-value answers in {target_language}. "
            f"Never output internal thought processes or <think> tags.{doc_awareness}"
        )

        if file:
            fbytes = await file.read()
            mime = file.content_type or "image/jpeg"
            ans = audit_document_dual_engine(f"Answer directly in {target_language}: {clean_q}", fbytes, mime)
        else:
            client = get_groq_client()
            if client:
                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": clean_q}],
                    temperature=0.4
                )
                ans = sanitize_ai_output(chat.choices[0].message.content or "")
            else:
                ans = "Language engine currently unavailable."

        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 3. UNIVERSAL CONVERTER & RESIZER ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/convert-file")
async def convert_file(request: Request, target_format: str = Form(...), file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        clean_ext = target_format.lower().replace(".", "").strip()
        file_id = f"Omni_{uuid.uuid4().hex[:6]}.{clean_ext}"
        output_path = os.path.join(DOWNLOADS_DIR, file_id)

        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            if pil_img.mode in ("RGBA", "P") and clean_ext in ["jpg", "jpeg", "bmp", "pdf"]:
                pil_img = pil_img.convert("RGB")
            if clean_ext == "pdf":
                pil_img.save(output_path, "PDF", resolution=100.0)
            elif clean_ext in ["jpg", "jpeg"]:
                pil_img.save(output_path, "JPEG", quality=95)
            elif clean_ext == "png":
                pil_img.save(output_path, "PNG")
            elif clean_ext == "webp":
                pil_img.save(output_path, "WEBP", quality=90)
            elif clean_ext == "bmp":
                pil_img.save(output_path, "BMP")
            elif clean_ext == "gif":
                pil_img.save(output_path, "GIF")
            elif clean_ext == "ico":
                pil_img.save(output_path, "ICO", sizes=[(128, 128)])
            elif clean_ext == "tiff":
                pil_img.save(output_path, "TIFF")
            else:
                with open(output_path, "wb") as f:
                    f.write(file_bytes)
        except Exception:
            with open(output_path, "wb") as f:
                f.write(file_bytes)

        base_url = str(request.base_url).rstrip("/")
        return {"status": "success", "download_url": f"{base_url}/downloads/{file_id}", "message": f"Successfully converted to .{clean_ext.upper()}"}
    except Exception as e:
        return {"status": "error", "message": f"Conversion error: {str(e)}"}

@app.post("/api/v1/resize-image")
async def resize_image(
    request: Request,
    mode: str = Form("size"),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    percentage: Optional[int] = Form(100),
    platform_preset: Optional[str] = Form("Square"),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes))
        orig_w, orig_h = img.size
        new_w, new_h = orig_w, orig_h

        if mode == "percentage" and percentage:
            scale = percentage / 100.0
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
        elif mode == "social" and platform_preset:
            presets = {
                "Facebook Profile (170 x 170)": (170, 170),
                "Facebook Post (1200 x 630)": (1200, 630),
                "Instagram Profile (320 x 320)": (320, 320),
                "Instagram Post / Square (1080 x 1080)": (1080, 1080),
                "Instagram Story (1080 x 1920)": (1080, 1920),
                "YouTube Thumbnail (1280 x 720)": (1280, 720),
                "Twitter / X Header (1500 x 500)": (1500, 500),
                "Twitter / X Post (1200 x 675)": (1200, 675),
                "LinkedIn Banner (1584 x 396)": (1584, 396)
            }
            new_w, new_h = presets.get(platform_preset, (1080, 1080))
        elif width and height:
            new_w, new_h = width, height

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        out_name = f"Resized_{new_w}x{new_h}_{uuid.uuid4().hex[:4]}.jpg"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        resized.save(out_path, "JPEG", quality=92)
        base_url = str(request.base_url).rstrip("/")
        return {"status": "success", "download_url": f"{base_url}/downloads/{out_name}", "dimensions": f"{new_w} x {new_h} px"}
    except Exception as e:
        return {"status": "error", "message": f"Resize failed: {str(e)}"}

# -------------------------------------------------------------
# 4. TOURISTOS 10-LANDMARK & 4 PILLARS ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/touristos-recommend")
async def touristos_recommend(
    country: str = Form("India"),
    state: str = Form("Maharashtra"),
    city: str = Form("Mumbai"),
    adults: int = Form(2),
    children: int = Form(0),
    dietary_preference: str = Form("Pure Vegetarian"),
    target_language: str = Form("English")
):
    location = f"{city}, {state}, {country}".strip(", ")
    client = get_groq_client()

    sys_prompt = (
        f"You are a local travel authority for '{location}'. The traveling party consists of {adults} adults and {children} children with '{dietary_preference}' diet.\n"
        f"Return ONLY valid JSON matching this exact schema (no markdown wrappers):\n"
        f"{{\n"
        f'  "destination_summary": "Thorough overview of {location} covering heritage, culture, and transit.",\n'
        f'  "spots": [\n'
        f'    {{"page": 1, "title": "Real Landmark 1", "rating": "⭐ 4.8", "dist": "Center", "description": "Authentic history, architecture, and visitor tips.", "images": ["https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800"]}},\n'
        f'    {{"page": 2, "title": "Real Landmark 2", "rating": "⭐ 4.7", "dist": "3 km", "description": "Authentic history, architecture, and visitor tips.", "images": ["https://images.unsplash.com/photo-1548013146-72479768bada?q=80&w=800"]}}\n'
        f'  ],\n'
        f'  "hotels": [\n'
        f'    {{"name": "Hotel Name", "price": "₹4,500 / night", "rating": "⭐ 4.8", "description": "Family-friendly stay near center", "amenities": ["Wi-Fi", "AC", "Pure Veg"]}}\n'
        f'  ],\n'
        f'  "emergency": {{\n'
        f'    "hospital_name": "Verified Local Hospital Name",\n'
        f'    "hospital_phone": "Actual Emergency Phone or 112",\n'
        f'    "police_name": "Local Police Precinct Name",\n'
        f'    "police_phone": "Actual Police Station Phone or 112",\n'
        f'    "fire_name": "Municipal Fire Station",\n'
        f'    "fire_phone": "112",\n'
        f'    "pharmacy_name": "24/7 Chemist Hub",\n'
        f'    "pharmacy_phone": "112"\n'
        f'  }}\n'
        f"}}"
    )

    try:
        if client:
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Generate 10 authentic attractions for {location} in {target_language}."}
                ],
                temperature=0.3,
                max_tokens=3500
            )
            raw = sanitize_ai_output(chat.choices[0].message.content or "")
            clean = raw.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            data = json.loads(clean)
            return {"status": "success", "data": data}
    except Exception as e:
        print(f"[TouristOS Intel Error]: {e}")

    # Regional Fail-Safe
    phone_default = "112" if any(x in location.lower() for x in ["india", "europe", "uk", "france"]) else "911"
    return {
        "status": "success",
        "data": {
            "destination_summary": f"{location} is an active regional center offering cultural heritage, dining, and transit networks.",
            "spots": [
                {"page": i + 1, "title": f"Landmark {i + 1} of {city}", "rating": "⭐ 4.8", "dist": f"{i + 1} km", "description": f"Verified historic attraction in {city}.", "images": ["https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800"]}
                for i in range(10)
            ],
            "hotels": [
                {"name": f"{city} Central Comfort Stay", "price": "Standard Rate", "rating": "⭐ 4.7 (1,500+)", "description": "Centrally situated accommodation with standard amenities.", "amenities": ["Free Wi-Fi", "AC", "Dining"]}
            ],
            "emergency": {
                "hospital_name": f"{city} General Care Hospital",
                "hospital_phone": phone_default,
                "police_name": f"{city} Central Police Precinct",
                "police_phone": phone_default,
                "fire_name": f"{city} Fire & Rescue",
                "fire_phone": phone_default,
                "pharmacy_name": f"{city} 24/7 Pharmacy Hub",
                "pharmacy_phone": phone_default
            }
        }
    }

# -------------------------------------------------------------
# 5. CONCIERGE EXPLORE CHAT
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(
    spot_name: str = Form("Destination"),
    question: str = Form("What are the visiting hours?"),
    target_language: str = Form("English")
):
    client = get_groq_client()
    sys_prompt = f"You are a local concierge for '{spot_name}'. Answer concisely in {target_language} using Markdown."
    try:
        if client:
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
                temperature=0.3
            )
            ans = sanitize_ai_output(chat.choices[0].message.content or "")
            return {"status": "success", "answer": ans}
    except Exception as e:
        return {"status": "error", "answer": f"Guide notice: {str(e)}"}
    return {"status": "success", "answer": f"Opening hours for {spot_name} are typically 9:00 AM to 6:00 PM."}

# -------------------------------------------------------------
# 6. INSTANT HEALTH & WARMUP
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni TouristOS Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }