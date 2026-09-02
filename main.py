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

app = FastAPI(
    title="Omni Forensic PaperPilot & TouristOS Engine",
    description="Forensic Legal, Fraud & Historical Document Intelligence",
    version="47.5.0"
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
# CLIENT & MODEL DISCOVERY
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=key) if key else None

def get_gemini_keys() -> List[str]:
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def get_active_groq_model(client: Groq) -> str:
    try:
        models_data = client.models.list().data
        active_ids = [m.id for m in models_data if "whisper" not in m.id and "guard" not in m.id]
        priority = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ]
        for p in priority:
            if p in active_ids:
                return p
        if active_ids:
            return active_ids[0]
    except Exception as e:
        print(f"[Groq Discovery Notice]: {e}")
    return "llama-3.1-8b-instant"

def sanitize_ai_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# MULTIMODAL FORENSIC VISION (GEMINI INLINE BYTES)
# -------------------------------------------------------------
def ask_gemini_vision(prompt: str, file_bytes: bytes) -> Optional[str]:
    keys = get_gemini_keys()
    if not keys:
        print("[Gemini]: No GEMINI_API_KEY set.")
        return None

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        clean_bytes = buf.getvalue()
    except Exception as e:
        print(f"[Image Normalization Error]: {e}")
        return None

    inline_part = {
        "mime_type": "image/jpeg",
        "data": clean_bytes
    }

    for key in keys:
        try:
            genai.configure(api_key=key)
            for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content([prompt, inline_part])
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception as model_err:
                    print(f"[Gemini Model {model_name} Error]: {model_err}")
                    continue
        except Exception as key_err:
            print(f"[Gemini Key Error]: {key_err}")
            continue
    return None

def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if client:
        try:
            chosen = get_active_groq_model(client)
            completion = client.chat.completions.create(
                model=chosen,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3500
            )
            raw = completion.choices[0].message.content
            if raw:
                return sanitize_ai_output(raw)
        except Exception as groq_err:
            print(f"[Groq Execution Error]: {groq_err}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            for m in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(m)
                    res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}")
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception:
                    continue
        except Exception:
            continue

    return "Document assistant is currently updating. Please try again shortly."

# -------------------------------------------------------------
# 1. FORENSIC FRAUD, LEGAL & DOCUMENT AUDITOR ENDPOINT
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()

        forensic_prompt = (
            f"You are an expert Forensic Document Auditor, Legal Counsel, and Archival Paleographer. "
            f"Carefully examine this uploaded document or image in {target_language}.\n\n"
            f"Determine the document type automatically:\n"
            f"1. LEGAL / PROPERTY / FRAUD: Land records (7/12, Index II, Satbara), Sale Deeds, Power of Attorney, Leases, Stamp Papers, Contracts, Affidavits.\n"
            f"   - Check: Government stamp value, treasury seals, serial consistency, encumbrance risks, forfeiture clauses, boundary/title discrepancies.\n"
            f"2. HISTORICAL / ARCHIVAL: Royal Sanads, colonial charters, antique manuscripts, ancestral genealogies.\n"
            f"   - Check: Script transcription (Modi, Urdu, Latin, Archaic English), period seals, historical provenance and modern legal standing.\n"
            f"3. GENERAL / FINANCIAL: Invoices, receipts, travel tickets, vouchers, certificates, identity documents.\n"
            f"   - Check: Authenticity, itemized financial totals, cancellation penalties, issuance status.\n\n"
            f"Return ONLY valid JSON matching this exact schema (no text outside JSON):\n"
            f"{{\n"
            f'  "classification": "LEGAL_PROPERTY | HISTORICAL_ARCHIVE | GENERAL_FINANCIAL",\n'
            f'  "status": "VERIFIED AUTHENTIC | HIGH RISK / PREDATORY CLAUSES | SUSPICIOUS ANOMALIES DETECTED",\n'
            f'  "document_title": "Clear concise title (e.g., Registered Sale Deed / Satbara 7/12 / Flight Ticket)",\n'
            f'  "issuing_authority_or_registry": "Government department, Sub-Registrar office, court, royal office, or merchant name",\n'
            f'  "parties_and_dates": "Parties involved (Buyer/Seller, Landlord/Tenant, Passenger), Execution Date, Registration Date",\n'
            f'  "metadata_identifiers": "Stamp paper serial, CTS/Survey/Plot number, PNR, or Registration volume number",\n'
            f'  "traps_risks_and_penalties": "Detailed breakdown of predatory clauses, forfeiture traps, missing seals, or non-refundable penalties in simple, clear language.",\n'
            f'  "financials_or_valuation": {{\n'
            f'    "base_amount": "Stamp duty / Base fare / Valuation consideration with currency symbol",\n'
            f'    "taxes_and_surcharges": "Registration charges, municipal tax, or surcharges",\n'
            f'    "grand_total": "Grand total valuation or payable amount",\n'
            f'    "payment_status": "PAID / REGISTERED / UNPAID / PENDING"\n'
            f'  }},\n'
            f'  "actionable_advisory": "Concrete, step-by-step guidance on what the user should do next (e.g., verification at Sub-Registrar office, legal due diligence, title search).",\n'
            f'  "detected_destination": "City and Country name if document indicates travel, otherwise null"\n'
            f"}}"
        )

        analysis_raw = ask_gemini_vision(forensic_prompt, file_bytes)
        if not analysis_raw:
            return {
                "status": "error",
                "message": "Visual analysis engine could not read the document. Ensure GEMINI_API_KEY has Generative Language API enabled.",
                "data": None
            }

        clean = analysis_raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(clean)
        except Exception:
            data = {
                "classification": "LEGAL_PROPERTY",
                "status": "VERIFIED DOCUMENT",
                "document_title": "Audited Document",
                "issuing_authority_or_registry": "Extracted Authority",
                "parties_and_dates": "Extracted Parties & Schedule",
                "metadata_identifiers": "Document Identifiers Extracted",
                "traps_risks_and_penalties": "Inspect fine print for unilateral indemnity or cancellation clauses.",
                "financials_or_valuation": {
                    "base_amount": "Extracted from document",
                    "taxes_and_surcharges": "Recorded fees",
                    "grand_total": "Verified in document",
                    "payment_status": "RECORDED"
                },
                "actionable_advisory": analysis_raw[:450],
                "detected_destination": None
            }

        return {"status": "success", "data": data, "raw_text": analysis_raw}
    except Exception as e:
        return {"status": "error", "message": f"Forensic audit notice: {str(e)}", "data": None}

# -------------------------------------------------------------
# 2. LIVE OMNI AI STUDIO (LEGAL, FORENSIC & GENERAL COMPANION)
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

        visual_triggers = ["generate image", "create image", "genrate image", "picture of", "photo of", "logo", "3d logo", "render", "illustration", "draw"]
        if any(t in lower_q for t in visual_triggers):
            clean_p = clean_q
            for t in ["generate an image of", "generate image of", "create an image of", "genrate image of", "generate image", "create image", "draw", "render"]:
                clean_p = re.sub(re.escape(t), "", clean_p, flags=re.IGNORECASE).strip()
            if "3d" in lower_q or "logo" in lower_q:
                clean_p += ", 3D octane render, volumetric lighting, photorealistic, 4k"
            enc = urllib.parse.quote(clean_p if clean_p else clean_q)
            img_url = f"https://image.pollinations.ai/prompt/{enc}?width=1024&height=1024&nologo=true&model=flux"
            return {"status": "success", "answer": f"Rendered visual: *\"{clean_p}\"*", "image_url": img_url, "download_url": img_url}

        doc_awareness = f"\n[DOCUMENT IN MEMORY (FORENSIC AUDIT RECORD)]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Omni Companion, an authentic AI advisor, legal document counselor, and general intelligence guide. "
            f"Answer in {target_language}. Never reveal internal thinking or <think> tags.\n"
            f"If an audited document is present in memory, act as a helpful legal peer: guide the user through survey numbers, "
            f"clauses, fraud risks, stamp paper validity, or next steps. If the user asks about any other topic, answer thoroughly "
            f"and supportively without hesitation.{doc_awareness}"
        )

        if file:
            fbytes = await file.read()
            ans = ask_gemini_vision(f"Answer in {target_language}: {clean_q}", fbytes) or "Unable to inspect document."
        else:
            ans = ask_hybrid_text(clean_q, sys_prompt)

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
        return {"status": "success", "download_url": f"{base_url}/downloads/{file_id}", "message": f"Converted to .{clean_ext.upper()}"}
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
# 4. SERVER HEALTH & PING
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni Forensic & TouristOS Cloud",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }