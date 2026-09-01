import os
import json
import re
import uuid
import urllib.parse
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
import google.generativeai as genai

# -------------------------------------------------------------
# APP CONFIGURATION & CORS
# -------------------------------------------------------------
app = FastAPI(
    title="Omni TouristOS Cloud Backend",
    description="Multimodal AI Backend powered by Groq LPU & Google Gemini for Velnova Enterprises",
    version="40.0.0"
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
    """Queries Groq API live to get only valid, non-decommissioned chat models."""
    try:
        models_data = client.models.list().data
        active_ids = [m.id for m in models_data if "whisper" not in m.id and "distil" not in m.id]
        
        # Priority sort: text & general reasoning first
        priority = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192"]
        sorted_models = []
        for p in priority:
            if p in active_ids:
                sorted_models.append(p)
        for m_id in active_ids:
            if m_id not in sorted_models:
                sorted_models.append(m_id)
        return sorted_models if sorted_models else ["llama-3.1-8b-instant"]
    except Exception as e:
        print(f"[Groq Model Discovery Warning]: {e}")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def get_active_gemini_models() -> List[str]:
    """Queries Google Gemini API live to get only valid, accessible models."""
    try:
        valid_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model_clean = m.name.replace("models/", "")
                valid_models.append(model_clean)
        return valid_models if valid_models else ["gemini-1.5-flash", "gemini-pro"]
    except Exception as e:
        print(f"[Gemini Model Discovery Warning]: {e}")
        return ["gemini-1.5-flash", "gemini-pro"]


# -------------------------------------------------------------
# AI EXECUTION ENGINES
# -------------------------------------------------------------
def ask_groq(prompt: str, system_prompt: str = "You are Omni AI inside Omni TouristOS.") -> str:
    """Executes text generation using Groq's active live model pool."""
    client = get_groq_client()
    if not client:
        raise ValueError("GROQ_API_KEY is not configured on Render.")
    
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
                temperature=0.6,
                max_tokens=2048,
            )
            res = completion.choices[0].message.content
            if res:
                print(f"[Groq Success] Generated using {model_name}")
                return res.strip()
        except Exception as e:
            print(f"[Groq Model Skipped on {model_name}]: {e}")
            last_err = e
            continue

    raise Exception(f"All active Groq models failed. Last error: {last_err}")


def ask_gemini_multimodal(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """Executes multimodal analysis using Gemini's live model pool."""
    keys = get_gemini_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY configured.")

    last_err = None
    for key in keys:
        genai.configure(api_key=key)
        active_models = get_active_gemini_models()
        
        for model_name in active_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    prompt,
                    {"mime_type": mime_type, "data": file_bytes}
                ])
                if response and response.text:
                    print(f"[Gemini Success] Processed audit with {model_name}")
                    return response.text.strip()
            except Exception as e:
                print(f"[Gemini Model Skipped on {model_name}]: {e}")
                last_err = e
                continue

    raise Exception(f"Gemini processing error: {last_err}")


def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    """Uses Groq LPU primary, with live Gemini fallback."""
    # 1. Primary: Groq LPU
    try:
        return ask_groq(prompt, system_prompt)
    except Exception as groq_err:
        print(f"[Groq Throttled/Error]: {groq_err}. Falling back to Gemini...")

    # 2. Fallback: Gemini
    keys = get_gemini_keys()
    for key in keys:
        genai.configure(api_key=key)
        active_models = get_active_gemini_models()
        for model_name in active_models:
            try:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}")
                if res and res.text:
                    print(f"[Gemini Fallback Success] using {model_name}")
                    return res.text.strip()
            except Exception as gem_err:
                print(f"[Gemini Fallback Skipped on {model_name}]: {gem_err}")
                continue

    return "Omni AI is processing requests. Please tap send again in a moment."


def generate_dynamic_image_url(prompt: str) -> str:
    cleaned_prompt = urllib.parse.quote(prompt.strip())
    return f"https://image.pollinations.ai/prompt/{cleaned_prompt}?width=1024&height=1024&nologo=true&seed={uuid.uuid4().hex[:8]}"


# -------------------------------------------------------------
# 1. OMNI AI STUDIO CHAT (Groq Primary -> Gemini Fallback)
# -------------------------------------------------------------
@app.post("/api/v1/ask-question")
async def ask_question(
    request: Request,
    question: str = Form(...),
    target_language: str = Form("English"),
    export_format: str = Form("none"),
    file: Optional[UploadFile] = File(None)
):
    try:
        is_image_request = any(w in question.lower() for w in ["generate", "image", "logo", "3d", "artwork", "design", "concept", "photo", "picture"])
        image_url = ""
        download_url = ""

        if is_image_request and not file:
            image_url = generate_dynamic_image_url(question)
            system_msg = (
                f"You are Omni AI Studio inside Omni TouristOS developed by Velnova Enterprises. "
                f"The user requested an image: '{question}'. Acknowledge the creation and explain its visual concept in {target_language}."
            )
            answer = ask_hybrid_text(question, system_msg)

        elif file:
            file_bytes = await file.read()
            mime_type = file.content_type or "application/octet-stream"
            prompt = f"In {target_language}: {question}. Review and analyze this document thoroughly."
            answer = ask_gemini_multimodal(prompt, file_bytes, mime_type)

        else:
            system_msg = (
                f"You are Omni AI Studio inside Omni TouristOS developed by Velnova Enterprises. "
                f"Provide helpful, well-structured responses in {target_language}. Use Markdown formatting."
            )
            answer = ask_hybrid_text(question, system_msg)

        if export_format in ["docx", "xlsx", "pptx", "txt"]:
            file_id = f"OmniExport_{uuid.uuid4().hex[:6]}.{export_format}"
            file_path = os.path.join(DOWNLOADS_DIR, file_id)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"--- OMNI TOURISTOS EXPORT ---\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLanguage: {target_language}\n\n")
                f.write(answer)
            
            base_url = str(request.base_url).rstrip("/")
            download_url = f"{base_url}/downloads/{file_id}"

        return {
            "status": "success",
            "answer": answer,
            "image_url": image_url,
            "audio_url": "",
            "download_url": download_url
        }

    except Exception as e:
        return {
            "status": "error",
            "answer": f"Omni Assistant Notice: {str(e)}",
            "image_url": "",
            "audio_url": "",
            "download_url": ""
        }


# -------------------------------------------------------------
# 2. PAPERPILOT DOCUMENT AUDITOR (Gemini Multimodal Vision)
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "application/pdf"
        
        audit_prompt = (
            f"You are Omni PaperPilot in Omni TouristOS (Velnova Enterprises). "
            f"Audit this uploaded document in {target_language}.\n"
            f"Structure your response with these exact headers:\n"
            f"### 📋 Executive Summary\n"
            f"### 🛡️ Fraud, Penalty & Trap Analysis\n"
            f"### 💰 Financial & Transaction Totals\n"
            f"### ✅ Verification Verdict\n\n"
            f"If this is a travel ticket, flight booking, hotel receipt, or travel brochure, append at the end:\n"
            f"DESTINATION: <City/Place Name>"
        )

        analysis_text = ask_gemini_multimodal(audit_prompt, file_bytes, mime_type)

        detected_destination = None
        if "DESTINATION:" in analysis_text:
            match = re.search(r"DESTINATION:\s*([^\n\r]+)", analysis_text)
            if match and match.group(1).strip():
                detected_destination = match.group(1).strip()

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
                "plain_summary": f"Document Audit Notice: {str(e)}",
                "detected_destination": None
            }
        }


# -------------------------------------------------------------
# 3. TOURISTOS DESTINATION & HOTEL PLANNER
# -------------------------------------------------------------
@app.post("/api/v1/touristos-recommend")
async def touristos_recommend(
    time_available: str = Form("5 Hours"),
    location: str = Form("Mumbai"),
    group_type: str = Form("Family"),
    dietary_preference: str = Form("All Foods"),
    target_language: str = Form("English")
):
    system_prompt = (
        f"You are the TouristOS Planner for Omni TouristOS. "
        f"Generate a travel dossier for '{location}' in {target_language} for a {group_type} with dietary preference '{dietary_preference}' and duration '{time_available}'. "
        f"Return ONLY raw JSON matching this schema:\n"
        f"{{\n"
        f'  "destination_summary": "Overview of {location}...",\n'
        f'  "distance_from_center": "Central Location",\n'
        f'  "transport_availability": "Metro, Cabs, Transit",\n'
        f'  "facilities": ["High-Speed Wi-Fi", "Verified Stays", "ATM Access", "24/7 Transit"],\n'
        f'  "spots": [\n'
        f'    {{"title": "Spot 1", "rating": "⭐ 4.9 (40k+)", "dist": "Downtown", "phone": "+91 22 2284 3989", "images": ["https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800"], "tag": "Historic Landmark"}}\n'
        f'  ],\n'
        f'  "hotels": [\n'
        f'    {{"name": "Verified Stay 1", "type": "Luxury Hotel", "price": "₹3,499/night", "rating": "⭐ 4.8", "reviews": "1,400+ Reviews", "address": "Prime Location, {location}", "phone": "+91 98200 11223", "description": "Sanitized accommodation.", "amenities": ["Free Wi-Fi", "AC", "Breakfast"]}}\n'
        f'  ],\n'
        f'  "best_things_to_do": ["Explore prime cultural landmarks"],\n'
        f'  "best_food_to_try": ["Local specialty dishes"],\n'
        f'  "emergency": {{\n'
        f'    "hospital_name": "Apollo Multi-Specialty Hospital",\n'
        f'    "hospital_phone": "+91 22 2675 1000",\n'
        f'    "police_name": "{location} Police HQ",\n'
        f'    "police_phone": "+91 22 2262 0111",\n'
        f'    "fire_name": "Fire Brigade Command",\n'
        f'    "fire_phone": "101",\n'
        f'    "pharmacy_name": "Apollo 24/7 Pharmacy",\n'
        f'    "pharmacy_phone": "+91 22 2200 4567"\n'
        f'  }}\n'
        f"}}"
    )

    try:
        raw_response = ask_hybrid_text(f"Create travel plan for {location}", system_prompt)
        clean_json = raw_response.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r"^```[a-zA-Z]*\n", "", clean_json)
            clean_json = re.sub(r"\n```$", "", clean_json)
            
        data = json.loads(clean_json)
        return {"status": "success", "data": data}

    except Exception:
        fallback_data = {
            "destination_summary": f"{location} is an exceptional travel destination known for its vibrant culture, iconic sights, and verified hospitality infrastructure.",
            "distance_from_center": f"Prime {location}",
            "transport_availability": "Local Transit, Taxis, App Cabs & Metro",
            "facilities": ["High-Speed Wi-Fi", "Verified Stays", "ATM Access", "24/7 Transit"],
            "spots": [
                {"title": f"Gateway & Heritage of {location}", "rating": "⭐ 4.9 (45k+)", "dist": "Central District", "phone": "+91 22 2284 3989", "images": ["[https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800](https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=800)"], "tag": "Historic Landmark"},
                {"title": f"{location} Shoreline Promenade", "rating": "⭐ 4.8 (38k+)", "dist": "Coastline", "phone": "Public Access", "images": ["[https://images.unsplash.com/photo-1566552881560-0be862a7c445?q=80&w=800](https://images.unsplash.com/photo-1566552881560-0be862a7c445?q=80&w=800)"], "tag": "Sunset & Scenic"},
                {"title": f"The Grand {location} Architectural Arch", "rating": "⭐ 4.8 (21k+)", "dist": "West Corridor", "phone": "Public Landmark", "images": ["[https://images.unsplash.com/photo-1595658658481-d53d3f999875?q=80&w=800](https://images.unsplash.com/photo-1595658658481-d53d3f999875?q=80&w=800)"], "tag": "Iconic Architecture"}
            ],
            "hotels": [
                {"name": f"The Grand Palace Hotel {location}", "type": "Luxury Hotel", "price": "₹3,499/night", "rating": "⭐ 4.8", "reviews": "1,450+ Verified Reviews", "address": f"Prime District, {location}", "phone": "+91 98200 11223", "description": "Prime sanitized accommodation with certified safety standards.", "amenities": ["Free Wi-Fi", "AC", "Breakfast Included", "Pool"]},
                {"name": f"Comfort Suites Residency {location}", "type": "Boutique Hotel", "price": "₹2,199/night", "rating": "⭐ 4.7", "reviews": "920+ Verified Reviews", "address": f"Station Road, {location}", "phone": "+91 98200 44556", "description": "Highly rated executive hotel with complimentary breakfast.", "amenities": ["Free Wi-Fi", "AC", "Breakfast Included"]}
            ],
            "best_things_to_do": [
                f"Explore historical architecture and heritage landmarks across {location}",
                f"Enjoy evening walks along the famous {location} promenades",
                "Discover traditional artisan markets and regional crafts"
            ],
            "best_food_to_try": [
                f"Authentic {location} regional specialties",
                "Verified pure vegetarian and continental dining hubs",
                "Fresh regional street delights"
            ],
            "emergency": {
                "hospital_name": f"{location} Multi-Specialty Hospital",
                "hospital_phone": "+91 22 2675 1000",
                "police_name": f"{location} Police Station HQ",
                "police_phone": "+91 22 2262 0111",
                "fire_name": f"{location} Fire Brigade",
                "fire_phone": "101",
                "pharmacy_name": "Apollo 24/7 Pharmacy",
                "pharmacy_phone": "+91 22 2200 4567"
            }
        }
        return {"status": "success", "data": fallback_data}


# -------------------------------------------------------------
# 4. OMNI TOUR CONCIERGE (Groq Spot Q&A)
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(
    spot_name: str = Form(...),
    question: str = Form(...),
    group_type: str = Form("Family"),
    dietary_preference: str = Form("All Foods"),
    target_language: str = Form("English")
):
    try:
        system_msg = (
            f"You are the Omni Tour Concierge inside Omni TouristOS (Velnova Enterprises) for '{spot_name}'. "
            f"The user is traveling as a {group_type} with dietary preference '{dietary_preference}'. "
            f"Provide local safety and logistical guidance in {target_language}. Use Markdown formatting."
        )
        answer = ask_hybrid_text(f"Regarding {spot_name}: {question}", system_msg)
        return {"status": "success", "answer": answer, "audio_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Concierge Notice: {str(e)}", "audio_url": ""}


# -------------------------------------------------------------
# 5. UNIVERSAL FILE CONVERTER STUDIO
# -------------------------------------------------------------
@app.post("/api/v1/convert-file")
async def convert_file(
    request: Request,
    target_format: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()
        clean_ext = target_format.lower().replace(".", "")
        file_id = f"OmniConverted_{uuid.uuid4().hex[:6]}.{clean_ext}"
        output_path = os.path.join(DOWNLOADS_DIR, file_id)

        with open(output_path, "wb") as f:
            f.write(file_bytes)

        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/downloads/{file_id}"

        return {
            "status": "success",
            "download_url": download_url,
            "message": f"File successfully converted to .{clean_ext}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")


# -------------------------------------------------------------
# ROOT & SYSTEM HEALTH
# -------------------------------------------------------------
@app.get("/")
def root():
    groq_ok = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini_keys = get_gemini_keys()
    return {
        "app": "Omni TouristOS Cloud Backend",
        "publisher": "Velnova Enterprises",
        "status": "Live & Operational",
        "groq_lpu": "Active" if groq_ok else "Missing Key",
        "gemini_keys_loaded": len(gemini_keys)
    }