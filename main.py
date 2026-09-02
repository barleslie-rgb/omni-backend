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
from PIL import Image
from groq import Groq
import google.generativeai as genai

app = FastAPI(
    title="Omni TouristOS Cloud Backend",
    description="Engine powered by Groq LPU & Google Gemini",
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
# DYNAMIC MODEL DISCOVERY HELPERS
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
        priority = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192"]
        sorted_models = [p for p in priority if p in active_ids]
        for m_id in active_ids:
            if m_id not in sorted_models:
                sorted_models.append(m_id)
        return sorted_models if sorted_models else ["llama-3.1-8b-instant"]
    except Exception as e:
        print(f"[Groq Discovery Notice]: {e}")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def get_active_gemini_models() -> List[str]:
    try:
        valid_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                valid_models.append(m.name.replace("models/", ""))
        return valid_models if valid_models else ["gemini-1.5-flash", "gemini-1.5-pro"]
    except Exception as e:
        print(f"[Gemini Discovery Notice]: {e}")
        return ["gemini-1.5-flash", "gemini-1.5-pro"]


# -------------------------------------------------------------
# EXECUTION ENGINES
# -------------------------------------------------------------
def ask_groq(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY is not configured.")
    
    active_models = get_active_groq_models(client)
    last_err = None

    for model_name in active_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            res = completion.choices[0].message.content
            if res:
                return res.strip()
        except Exception as e:
            last_err = e
            continue

    raise Exception(f"Groq execution failed: {last_err}")


def ask_gemini_multimodal(prompt: str, file_bytes: bytes, original_mime: str = "") -> str:
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY configured.")

    # Auto-detect image formats by attempting PIL decode on raw bytes
    is_image = False
    img = None
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        img = Image.open(io.BytesIO(file_bytes))  # Reopen after verification
        is_image = True
    except Exception:
        is_image = False

    temp_file_path = None
    try:
        if is_image and img is not None:
            parts = [prompt, img]
        else:
            # Handle PDF and other text documents
            ext = ".pdf" if "pdf" in original_mime.lower() else ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                temp_file_path = tmp.name

        last_err = None
        for key in keys:
            genai.configure(api_key=key)
            active_models = get_active_gemini_models()

            if not is_image and temp_file_path:
                try:
                    uploaded = genai.upload_file(temp_file_path, mime_type=original_mime or "application/pdf")
                    parts = [prompt, uploaded]
                except Exception:
                    parts = [prompt]

            for model_name in active_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(parts)
                    if res and res.text:
                        return res.text.strip()
                except Exception as e:
                    last_err = e
                    continue

        raise Exception(f"Vision API Error: {last_err}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    try:
        return ask_groq(prompt, system_prompt)
    except Exception:
        pass

    keys = get_gemini_keys()
    for key in keys:
        genai.configure(api_key=key)
        for model_name in get_active_gemini_models():
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}")
                if res and res.text:
                    return res.text.strip()
            except Exception:
                continue

    return "Service is momentarily updating. Please tap submit again."


# -------------------------------------------------------------
# 1. AI STUDIO CHAT (Concise & Direct)
# -------------------------------------------------------------
@app.post("/api/v1/ask-question")
async def ask_question(
    question: str = Form(...),
    target_language: str = Form("English"),
    export_format: str = Form("none"),
    file: Optional[UploadFile] = File(None)
):
    try:
        clean_q = question.strip()
        system_msg = (
            f"You are a helpful and direct AI assistant. "
            f"Provide crisp, structured, and proportional answers in {target_language}. "
            f"Never pad simple queries with unnecessary boilerplate, greetings, or filler essays. "
            f"Use compact bullet points where appropriate."
        )

        if file:
            file_bytes = await file.read()
            mime = file.content_type or "image/jpeg"
            prompt = f"Analyze this file in {target_language} and answer directly: {clean_q}"
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
# 2. DOCUMENT AUDITOR (Universal Vision + Classifier)
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
            f"You are a document auditor. Analyze the attached image, ticket, invoice, or document visually in {target_language}.\n"
            f"Read every printed text line, stamp, price, and date visible.\n"
            f"Never state that you received binary or unreadable data if this is an image. Parse all visible text directly.\n\n"
            f"Format your response using these exact markdown headers:\n"
            f"### 📄 Document Overview\n"
            f"(Break down: Document type, issuer, date of issue/travel, and primary parties involved)\n\n"
            f"### 🛡️ Fraud, Penalty & Trap Analysis\n"
            f"• Integrity Status: (Authentic / Suspicious / Tampered / Incomplete)\n"
            f"• Hidden Penalties / Traps: (Cancellation rules, non-refundable clauses, blackout dates, or zero-risk)\n"
            f"• Security Assessment: (Visual typography, barcodes, and authenticity indicators)\n\n"
            f"### 💰 Financial & Transaction Breakdown\n"
            f"• Extracted Total: (Base charges, taxes/fees, and grand total with original currency symbol)\n"
            f"• Payment Status: (Confirmed / Pending / Unpaid)\n\n"
            f"### ✅ Verification Verdict\n"
            f"(Clear statement on whether this document is valid, actionable, or requires clarification)\n\n"
            f"TRAVEL CLASSIFICATION TAG:\n"
            f"If and ONLY IF this file is an Air Ticket, Train Reservation, Bus Ticket, or Hotel Booking, "
            f"append at the end on a new line:\n"
            f"[SILENT_DESTINATION: <Destination City Name>]\n"
        )

        analysis_text = ask_gemini_multimodal(audit_prompt, file_bytes, mime_type)

        detected_destination = None
        if "[SILENT_DESTINATION:" in analysis_text:
            match = re.search(r"\[SILENT_DESTINATION:\s*([^\]]+)\]", analysis_text)
            if match and match.group(1).strip():
                detected_destination = match.group(1).strip()
            analysis_text = re.sub(r"\[SILENT_DESTINATION:\s*[^\]]+\]", "", analysis_text).strip()

        return {
            "status": "success",
            "data": {
                "plain_summary": analysis_text,
                "detected_destination": detected_destination
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "data": {
                "plain_summary": f"### ⚠️ Document Audit Notice\nUnable to process document: {str(e)}",
                "detected_destination": None
            }
        }


# -------------------------------------------------------------
# 3. DESTINATION & HOTEL PLANNER
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
        f"You are a global travel planner. "
        f"Generate a comprehensive travel dossier strictly for '{location}'. "
        f"Provide real attractions located directly in '{location}', authentic hotels with local prices, "
        f"accurate emergency numbers for that country (e.g. 911 in US, 112 in India/EU, 999 in UK), and transit in {target_language}.\n"
        f"Return ONLY valid JSON matching this schema:\n"
        f"{{\n"
        f'  "destination_summary": "Description of {location}...",\n'
        f'  "distance_from_center": "Central District",\n'
        f'  "transport_availability": "Transit options in {location}",\n'
        f'  "facilities": ["Wi-Fi", "Accommodations", "ATM Access"],\n'
        f'  "spots": [\n'
        f'    {{"title": "Spot 1", "rating": "⭐ 4.8 (15k+)", "dist": "Downtown", "phone": "Public Area", "images": ["https://images.unsplash.com/photo-1506744038136-46273834b3fb?q=80&w=800"], "tag": "Top Attraction"}},\n'
        f'    {{"title": "Spot 2", "rating": "⭐ 4.7 (9k+)", "dist": "Center", "phone": "Public Area", "images": ["https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=800"], "tag": "Historic"}}\n'
        f'  ],\n'
        f'  "hotels": [\n'
        f'    {{"name": "Hotel 1", "type": "Hotel", "price": "Standard Rate", "rating": "⭐ 4.8", "reviews": "1,000+ Reviews", "address": "{location}", "phone": "Front Desk", "description": "Quality stay.", "amenities": ["Wi-Fi", "AC"]}}\n'
        f'  ],\n'
        f'  "best_things_to_do": ["Activity 1", "Activity 2"],\n'
        f'  "best_food_to_try": ["Dish 1", "Dish 2"],\n'
        f'  "emergency": {{\n'
        f'    "hospital_name": "Major Hospital",\n'
        f'    "hospital_phone": "Emergency Number",\n'
        f'    "police_name": "Police Department",\n'
        f'    "police_phone": "Emergency Number",\n'
        f'    "fire_name": "Fire & Rescue",\n'
        f'    "fire_phone": "Emergency Number",\n'
        f'    "pharmacy_name": "24/7 Pharmacy",\n'
        f'    "pharmacy_phone": "Emergency Number"\n'
        f'  }}\n'
        f"}}"
    )

    try:
        raw_res = ask_hybrid_text(f"Plan for location: '{location}', time: '{time_available}'", system_prompt)
        clean = raw_res.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        data = json.loads(clean)
        return {"status": "success", "data": data}
    except Exception as e:
        # Structured fallback
        return {
            "status": "success",
            "data": {
                "destination_summary": f"{location} is an active regional hub offering cultural landmarks, dining, and transit connections.",
                "distance_from_center": f"Central {location}",
                "transport_availability": "Buses, local trains, rideshares, and taxis",
                "facilities": ["Wi-Fi", "Verified Accommodations", "ATM Access"],
                "spots": [
                    {"title": f"Historic Center of {location}", "rating": "⭐ 4.8", "dist": "City Center", "phone": "Public", "images": ["https://images.unsplash.com/photo-1506744038136-46273834b3fb?q=80&w=800"], "tag": "Historic Landmark"},
                    {"title": f"{location} Promenade & Gardens", "rating": "⭐ 4.7", "dist": "1.5 km", "phone": "Public", "images": ["https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=800"], "tag": "Scenic"}
                ],
                "hotels": [
                    {"name": f"{location} Central Stay", "type": "Hotel", "price": "Standard Rate", "rating": "⭐ 4.8", "reviews": "1,000+", "address": f"Prime District, {location}", "phone": "112", "description": "Centrally located quality stay.", "amenities": ["Free Wi-Fi", "AC"]}
                ],
                "best_things_to_do": [f"Explore heritage sites in {location}", f"Sample regional street food and dining in {location}"],
                "best_food_to_try": ["Local specialties", "Vegetarian and global options"],
                "emergency": {
                    "hospital_name": f"{location} Hospital",
                    "hospital_phone": "112",
                    "police_name": f"{location} Police",
                    "police_phone": "112",
                    "fire_name": "Fire & Rescue",
                    "fire_phone": "112",
                    "pharmacy_name": "24/7 Pharmacy Hub",
                    "pharmacy_phone": "112"
                }
            }
        }


# -------------------------------------------------------------
# 4. TOUR CONCIERGE (Form + JSON Hybrid Support)
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(request: Request):
    try:
        spot_name = "Destination"
        question = "Tell me more about this place."
        group_type = "Family"
        dietary_preference = "All Foods"
        target_language = "English"

        # Check content type to accept both Form and JSON safely
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
            f"Answer concisely in {target_language} using Markdown. No corporate introductions."
        )
        answer = ask_hybrid_text(f"Regarding {spot_name}: {question}", system_msg)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "answer": f"Guide notice: {str(e)}"}


@app.get("/")
def health_check():
    return {
        "status": "Operational",
        "groq_lpu": bool(os.environ.get("GROQ_API_KEY")),
        "gemini_keys": len(get_gemini_keys())
    }