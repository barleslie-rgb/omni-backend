import os
import io
import json
import re
import uuid
import tempfile
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq
import google.generativeai as genai
import requests

app = FastAPI(
    title="Omni TouristOS Cloud Engine",
    description="Multimodal Intelligence, Image Synthesis & Travel Companion API",
    version="45.0.0"
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
    raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

def get_active_groq_models(client: Groq) -> List[str]:
    try:
        models_data = client.models.list().data
        active_ids = [m.id for m in models_data if "whisper" not in m.id and "distil" not in m.id]
        priority = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]
        sorted_models = [p for p in priority if p in active_ids]
        for m_id in active_ids:
            if m_id not in sorted_models:
                sorted_models.append(m_id)
        return sorted_models if sorted_models else ["llama-3.3-70b-versatile"]
    except Exception as e:
        print(f"[Groq Discovery]: {e}")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

def sanitize_ai_output(text: str) -> str:
    """Strips <think>...</think> reasoning tags and cleans trailing formatting."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# CORE LLM EXECUTORS
# -------------------------------------------------------------
def ask_groq(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY is not configured.")
    
    for model_name in get_active_groq_models(client):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000,
            )
            raw = completion.choices[0].message.content
            if raw:
                return sanitize_ai_output(raw)
        except Exception:
            continue
    raise Exception("Groq execution failed across all models.")

def ask_gemini_multimodal(prompt: str, file_bytes: bytes, original_mime: str = "") -> str:
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY configured.")

    # Image byte verification
    is_image = False
    img = None
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        img = Image.open(io.BytesIO(file_bytes))
        is_image = True
    except Exception:
        is_image = False

    temp_path = None
    try:
        if is_image and img is not None:
            parts = [prompt, img]
        else:
            ext = ".pdf" if "pdf" in original_mime.lower() else ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

        for key in keys:
            genai.configure(api_key=key)
            models = ["gemini-1.5-flash", "gemini-1.5-pro"]
            if not is_image and temp_path:
                try:
                    uploaded = genai.upload_file(temp_path, mime_type=original_mime or "application/pdf")
                    parts = [prompt, uploaded]
                except Exception:
                    parts = [prompt]

            for model_name in models:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(parts)
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception:
                    continue

        raise Exception("Multimodal Vision analysis timed out on available API keys.")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    try:
        return ask_groq(prompt, system_prompt)
    except Exception:
        pass

    for key in get_gemini_keys():
        genai.configure(api_key=key)
        for m in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            try:
                model = genai.GenerativeModel(m)
                res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}")
                if res and res.text:
                    return sanitize_ai_output(res.text)
            except Exception:
                continue
    return "Local travel intelligence is updating. Please tap submit again."

# -------------------------------------------------------------
# 1. LIVE OMNI AI STUDIO (With Real Image Synthesis)
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

        # REAL IMAGE SYNTHESIS ROUTER
        image_triggers = ["generate image", "create image", "draw", "generate an image", "picture of", "photo of"]
        is_image_request = any(lower_q.startswith(trigger) or f" {trigger}" in lower_q for trigger in image_triggers)

        if is_image_request:
            clean_prompt = clean_q
            for trigger in ["generate an image of", "generate image of", "create an image of", "generate image", "create image", "draw"]:
                clean_prompt = re.sub(re.escape(trigger), "", clean_prompt, flags=re.IGNORECASE).strip()

            encoded_prompt = urllib.parse.quote(clean_prompt if clean_prompt else clean_q)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"

            return {
                "status": "success",
                "answer": f"Generated artwork based on your prompt: *\"{clean_prompt}\"*",
                "image_url": image_url,
                "download_url": image_url
            }

        # COMPANION TEXT & MULTIMODAL QUERY
        doc_awareness = ""
        if active_document_context and active_document_context.strip():
            doc_awareness = f"\n[ACTIVE USER DOCUMENT DOSSIER IN MEMORY]:\n{active_document_context.strip()}\n"

        system_msg = (
            f"You are a companion AI and travel strategist. "
            f"Provide direct, high-value, and proportional answers in {target_language}. "
            f"Never output internal thought processes, <think> tags, or greetings. "
            f"If an active document is provided below, answer all questions using its verified dates, parties, and fees.{doc_awareness}"
        )

        if file:
            file_bytes = await file.read()
            mime = file.content_type or "image/jpeg"
            prompt = f"Review this document and answer directly in {target_language}: {clean_q}"
            answer = ask_gemini_multimodal(prompt, file_bytes, mime)
        else:
            answer = ask_hybrid_text(clean_q, system_msg)

        return {
            "status": "success",
            "answer": answer,
            "image_url": "",
            "download_url": ""
        }
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 2. PAPER PILOT MULTIMODAL AUDITOR (Structured High-Legibility)
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"

        audit_prompt = (
            f"You are a senior document auditor. Visually inspect every printed stamp, signature, table, barcode, and total in {target_language}.\n"
            f"Format your response with these exact structural headers:\n"
            f"### 📄 Document Overview\n"
            f"• Type & Issuer: (e.g., E-Ticket, Tax Invoice, Hotel Reservation)\n"
            f"• Primary Parties & Dates: (Passenger/Customer names, issue date, travel dates)\n\n"
            f"### 🛡️ Forensic Authenticity & Risk\n"
            f"• Authenticity Status: (VERIFIED AUTHENTIC / SUSPICIOUS / UNVERIFIED)\n"
            f"• Traps & Penalty Analysis: (State cancellation policies, non-refundable clauses, and hidden surcharges)\n\n"
            f"### 💰 Financial Breakdown\n"
            f"• Base Fare / Subtotal: (State exact amount with currency symbol)\n"
            f"• Taxes & Fees: (State exact breakdown)\n"
            f"• Grand Total: (State bold total amount with currency)\n"
            f"• Payment State: (CONFIRMED / PENDING / UNPAID)\n\n"
            f"### ✅ Verification Verdict\n"
            f"(Provide an actionable closing statement on validity and safe travel use)\n\n"
            f"CLASSIFICATION TAG:\n"
            f"If this is any travel booking, flight, train, or hotel voucher, append at the very bottom:\n"
            f"[SILENT_DESTINATION: <Destination City Name>]\n"
        )

        analysis = ask_gemini_multimodal(audit_prompt, file_bytes, mime_type)

        detected_dest = None
        if "[SILENT_DESTINATION:" in analysis:
            m = re.search(r"\[SILENT_DESTINATION:\s*([^\]]+)\]", analysis)
            if m and m.group(1).strip():
                detected_dest = m.group(1).strip()
            analysis = re.sub(r"\[SILENT_DESTINATION:\s*[^\]]+\]", "", analysis).strip()

        return {
            "status": "success",
            "data": {
                "plain_summary": analysis,
                "detected_destination": detected_dest
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "data": {
                "plain_summary": f"### ⚠️ Document Verification Notice\nUnable to complete visual audit: {str(e)}",
                "detected_destination": None
            }
        }

# -------------------------------------------------------------
# 3. UNIVERSAL FILE CONVERTER & RESIZER ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/convert-file")
async def convert_file(
    request: Request,
    target_format: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()
        clean_ext = target_format.lower().replace(".", "").strip()
        file_id = f"Omni_{uuid.uuid4().hex[:6]}.{clean_ext}"
        output_path = os.path.join(DOWNLOADS_DIR, file_id)

        # Image-to-Image / Image-to-PDF Conversion via Pillow
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
            # Fallback for text/binary streams
            with open(output_path, "wb") as f:
                f.write(file_bytes)

        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/downloads/{file_id}"

        return {
            "status": "success",
            "download_url": download_url,
            "message": f"Successfully converted to .{clean_ext.upper()}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Conversion notice: {str(e)}"}

@app.post("/api/v1/resize-image")
async def resize_image(
    request: Request,
    mode: str = Form("size"),  # "size", "percentage", "social"
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
                "Facebook Profile": (170, 170),
                "Facebook Post": (1200, 630),
                "Instagram Profile": (320, 320),
                "Instagram Post / Square": (1080, 1080),
                "Instagram Story": (1080, 1920),
                "YouTube Thumbnail": (1280, 720),
                "Twitter / X Header": (1500, 500),
                "Twitter / X Post": (1200, 675),
                "LinkedIn Banner": (1584, 396)
            }
            target_dims = presets.get(platform_preset, (1080, 1080))
            new_w, new_h = target_dims
        elif width and height:
            new_w, new_h = width, height

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        out_name = f"Resized_{new_w}x{new_h}_{uuid.uuid4().hex[:4]}.jpg"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        resized.save(out_path, "JPEG", quality=92)

        base_url = str(request.base_url).rstrip("/")
        return {
            "status": "success",
            "download_url": f"{base_url}/downloads/{out_name}",
            "dimensions": f"{new_w} x {new_h} px",
            "message": f"Exported successfully at {new_w}x{new_h} px"
        }
    except Exception as e:
        return {"status": "error", "message": f"Resize failed: {str(e)}"}

# -------------------------------------------------------------
# 4. TOURISTOS DESTINATION EXPLORER & REAL LOCAL INTEL
# -------------------------------------------------------------
@app.post("/api/v1/touristos-recommend")
async def touristos_recommend(
    time_available: str = Form("Full Day"),
    location: str = Form("San Jose, California, United States"),
    group_type: str = Form("Family"),
    dietary_preference: str = Form("All Foods"),
    target_language: str = Form("English")
):
    system_prompt = (
        f"You are a global travel concierge. "
        f"Generate authentic travel intelligence for '{location}' in {target_language}.\n"
        f"Provide 10 REAL attractions, authentic regional hotels with prices, local delicacies, and national emergency contacts (e.g., 911 in US, 112 in India/EU, 999 in UK).\n"
        f"Return ONLY valid JSON matching this schema:\n"
        f"{{\n"
        f'  "destination_summary": "Comprehensive overview of {location}...",\n'
        f'  "transport_availability": "Local transport details...",\n'
        f'  "spots": [\n'
        f'    {{"title": "Real Spot 1", "rating": "⭐ 4.8", "dist": "1.2 km", "description": "Authentic history...", "images": ["https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800"]}},\n'
        f'    {{"title": "Real Spot 2", "rating": "⭐ 4.7", "dist": "3.5 km", "description": "Local culture...", "images": ["https://images.unsplash.com/photo-1548013146-72479768bada?q=80&w=800"]}}\n'
        f'  ],\n'
        f'  "hotels": [\n'
        f'    {{"name": "Hotel 1", "price": "Standard Rate", "rating": "⭐ 4.7 (1,200+)", "description": "Verified stay...", "amenities": ["Wi-Fi", "AC"]}}\n'
        f'  ],\n'
        f'  "best_things_to_do": ["Activity 1", "Activity 2", "Activity 3"],\n'
        f'  "best_food_to_try": ["Local specialty 1", "Local specialty 2"],\n'
        f'  "emergency": {{\n'
        f'    "hospital_name": "Regional Care Hospital",\n'
        f'    "hospital_phone": "112",\n'
        f'    "police_name": "Local Police Precinct",\n'
        f'    "police_phone": "112",\n'
        f'    "fire_name": "Central Fire Station",\n'
        f'    "fire_phone": "112",\n'
        f'    "pharmacy_name": "24/7 Medico Center",\n'
        f'    "pharmacy_phone": "112"\n'
        f'  }}\n'
        f"}}"
    )

    try:
        raw_res = ask_hybrid_text(f"Dossier for '{location}', Duration: {time_available}", system_prompt)
        clean = raw_res.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        data = json.loads(clean)
        return {"status": "success", "data": data}
    except Exception:
        # High-Fidelity Regional Fail-Safe
        phone_default = "112" if any(x in location.lower() for x in ["india", "europe", "uk", "france"]) else "911"
        return {
            "status": "success",
            "data": {
                "destination_summary": f"{location} is an active regional center offering cultural heritage, dining, and transit networks.",
                "transport_availability": "Taxis, auto-rickshaws/rideshares, metro, and local bus routes",
                "spots": [
                    {"title": f"Historic Center & Landmarks of {location}", "rating": "⭐ 4.8", "dist": "City Center", "description": "Central heritage district with architectural sites and markets.", "images": ["https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800"]},
                    {"title": f"{location} Waterfront Promenade & Parks", "rating": "⭐ 4.7", "dist": "2.1 km", "description": "Open scenic promenades and recreational park areas.", "images": ["https://images.unsplash.com/photo-1548013146-72479768bada?q=80&w=800"]}
                ],
                "hotels": [
                    {"name": f"{location} Central Comfort Stay", "price": "Standard Rate", "rating": "⭐ 4.7 (1,500+)", "description": "Centrally situated accommodation with standard amenities.", "amenities": ["Free Wi-Fi", "AC", "Breakfast"]}
                ],
                "best_things_to_do": [f"Visit local architectural landmarks in {location}", f"Experience neighborhood street markets and cultural dining"],
                "best_food_to_try": ["Traditional regional dishes", "Fresh vegetarian specialties"],
                "emergency": {
                    "hospital_name": f"{location} General Care Hospital",
                    "hospital_phone": phone_default,
                    "police_name": f"{location} Central Police Precinct",
                    "police_phone": phone_default,
                    "fire_name": f"{location} Fire & Rescue",
                    "fire_phone": phone_default,
                    "pharmacy_name": f"{location} 24/7 Pharmacy Hub",
                    "pharmacy_phone": phone_default
                }
            }
        }

# -------------------------------------------------------------
# 5. TOURIST CONCIERGE EXPLORE CHAT
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(request: Request):
    try:
        spot_name = "Destination"
        question = "Tell me more about this landmark."
        group_type = "Family"
        dietary_preference = "All Foods"
        target_language = "English"

        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            body = await request.json()
            spot_name = body.get("spot_name", spot_name)
            question = body.get("question", question)
            group_type = body.get("group_type", group_type)
            dietary_preference = body.get("dietary_preference", dietary_preference)
            target_language = body.get("target_language", target_language)
        else:
            form = await request.form()
            spot_name = form.get("spot_name", spot_name)
            question = form.get("question", question)
            group_type = form.get("group_type", group_type)
            dietary_preference = form.get("dietary_preference", dietary_preference)
            target_language = form.get("target_language", target_language)

        system_msg = (
            f"You are a local guide for '{spot_name}'. "
            f"The visitor is traveling as a {group_type} with dietary preference '{dietary_preference}'. "
            f"Answer concisely in {target_language} using Markdown. No boilerplate."
        )
        answer = ask_hybrid_text(f"Landmark '{spot_name}': {question}", system_msg)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "answer": f"Guide notice: {str(e)}"}

# -------------------------------------------------------------
# 6. INSTANT SERVER WARMUP & HEALTH PING
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