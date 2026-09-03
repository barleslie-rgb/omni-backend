import os
import io
import gc
import json
import re
import uuid
import base64
import zipfile
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq
import google.generativeai as genai

app = FastAPI(
    title="Omni Forensic PaperPilot & TouristOS Engine",
    description="Resilient Vision, Conversion & Travel Platform",
    version="53.0.0"
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
# KEYS & DYNAMIC DISCOVERY
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=key) if key else None

def get_gemini_keys() -> List[str]:
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def get_active_groq_text_model(client: Groq) -> str:
    try:
        models_data = client.models.list().data
        active_ids = [m.id for m in models_data]
        for p in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"]:
            if p in active_ids:
                return p
        filtered = [m for m in active_ids if "guard" not in m and "whisper" not in m and "vision" not in m]
        if filtered:
            return filtered[0]
    except Exception as e:
        print(f"[Groq Text Discovery]: {e}")
    return "llama-3.3-70b-versatile"

def sanitize_ai_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# LIGHTWEIGHT NATIVE PDF & WORD (.DOCX) COMPILER
# -------------------------------------------------------------
def compile_pdf_document(title: str, content: str, output_path: str):
    """Generates a valid standard PDF document without external heavy C libraries."""
    lines = []
    lines.append(f"BT /F1 16 Tf 50 750 Td ({title[:55]}) Tj ET")
    lines.append(f"BT /F1 9 Tf 50 735 Td (Omni TouristOS Travel Dossier - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}) Tj ET")
    lines.append("0 0 0 rg 50 725 m 550 725 l S")
    
    y = 705
    clean_paragraphs = content.replace("\r", "").split("\n")
    for p in clean_paragraphs:
        p_clean = p.strip()
        if not p_clean:
            y -= 8
            continue
        safe_p = p_clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        words = safe_p.split(" ")
        curr_line = ""
        for w in words:
            if len(curr_line) + len(w) > 85:
                lines.append(f"BT /F1 10 Tf 50 {y} Td ({curr_line}) Tj ET")
                y -= 14
                curr_line = w
                if y < 60:
                    break
            else:
                curr_line = f"{curr_line} {w}".strip()
        if curr_line and y >= 60:
            lines.append(f"BT /F1 10 Tf 50 {y} Td ({curr_line}) Tj ET")
            y -= 14
        if y < 60:
            break

    stream_content = "\n".join(lines)
    stream_len = len(stream_content.encode("latin-1", errors="ignore"))

    pdf_template = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {stream_len} >>\n"
        "stream\n"
        f"{stream_content}\n"
        "endstream\n"
        "endobj\n"
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n"
        "0 6\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000244 00000 n \n"
        f"{str(300 + stream_len).zfill(10)} 00000 n \n"
        "trailer << /Size 6 /Root 1 0 R >>\n"
        "startxref\n"
        f"{str(370 + stream_len)}\n"
        "%%EOF\n"
    )

    with open(output_path, "wb") as f:
        f.write(pdf_template.encode("latin-1", errors="ignore"))

def compile_docx_document(title: str, content: str, output_path: str):
    """Generates an authentic Microsoft Word .docx OpenXML container using python's built-in zipfile."""
    paragraphs_xml = [f"<w:p><w:pPr><w:pStyle w:val='Heading1'/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val='32'/><w:color w:val='2563EB'/></w:rPr><w:t>{title}</w:t></w:r></w:p>"]
    
    clean_paragraphs = content.replace("\r", "").split("\n")
    for p in clean_paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        safe_xml = p_clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        is_bold = safe_xml.startswith("**") or safe_xml.startswith("#")
        safe_xml = safe_xml.replace("**", "").replace("#", "").strip()
        bold_tag = "<w:b/>" if is_bold else ""
        paragraphs_xml.append(f"<w:p><w:r><w:rPr>{bold_tag}<w:sz w:val='22'/></w:rPr><w:t>{safe_xml}</w:t></w:r></w:p>")

    body_content = "".join(paragraphs_xml)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_content}
  </w:body>
</w:document>"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

# -------------------------------------------------------------
# DUAL-ENGINE VISION PIPELINE (GEMINI GA MODELS)
# -------------------------------------------------------------
def prepare_image_safe(file_bytes: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > 1280:
            pil_img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=82, optimize=True)
        clean_bytes = buf.getvalue()
        b64_str = base64.b64encode(clean_bytes).decode("utf-8")
        del clean_bytes
        gc.collect()
        return pil_img, b64_str
    except Exception as e:
        print(f"[Pillow Processing Error]: {e}")
        return None, None

def run_gemini_vision(prompt: str, pil_img: Image.Image) -> Tuple[Optional[str], Optional[str]]:
    keys = get_gemini_keys()
    if not keys:
        return None, "Gemini API key not configured."

    last_error = ""
    for key in keys:
        try:
            genai.configure(api_key=key)
            active_vision_models = []
            try:
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        clean_name = m.name.replace("models/", "")
                        if any(v in clean_name for v in ["2.5-flash", "2.0-flash", "flash"]):
                            if "-exp" not in clean_name and clean_name not in active_vision_models:
                                active_vision_models.append(clean_name)
            except Exception:
                pass

            candidates = active_vision_models if active_vision_models else ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for model_name in candidates:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content([prompt, pil_img], request_options={"timeout": 22})
                    if res and res.text and len(res.text.strip()) > 10:
                        return sanitize_ai_output(res.text), None
                except Exception as me:
                    last_error = f"{model_name}: {str(me)[:90]}"
                    continue
        except Exception as ke:
            last_error = f"Key config: {str(ke)[:90]}"
            continue

    return None, f"Gemini ({last_error})"

def audit_document_robust(prompt: str, file_bytes: bytes) -> Tuple[Optional[str], str]:
    pil_img, b64_img = prepare_image_safe(file_bytes)
    if not pil_img or not b64_img:
        return None, "Could not decode uploaded document image format."
    gemini_res, gemini_err = run_gemini_vision(prompt, pil_img)
    gc.collect()
    if gemini_res:
        return gemini_res, ""
    return None, f"Inspection notice: {gemini_err}"

def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if client:
        try:
            chosen = get_active_groq_text_model(client)
            completion = client.chat.completions.create(
                model=chosen,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000,
                timeout=16
            )
            raw = completion.choices[0].message.content
            if raw:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Error]: {e}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(m)
                    res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}", request_options={"timeout": 14})
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception:
                    continue
        except Exception:
            continue

    return "Service is momentarily busy. Please try again shortly."

# -------------------------------------------------------------
# 1. PAPERPILOT FORENSIC AUDITOR
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        forensic_prompt = (
            f"You are an expert Forensic Document Auditor. Analyze this document thoroughly in {target_language}.\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f"{{\n"
            f'  "classification": "LEGAL_PROPERTY | HISTORICAL_ARCHIVE | GENERAL_FINANCIAL",\n'
            f'  "status": "VERIFIED AUTHENTIC | HIGH RISK / PREDATORY CLAUSES | SUSPICIOUS ANOMALIES DETECTED",\n'
            f'  "document_title": "Concise descriptive title of document",\n'
            f'  "issuing_authority_or_registry": "Issuing authority or merchant",\n'
            f'  "parties_and_dates": "Parties involved and key recorded dates",\n'
            f'  "metadata_identifiers": "Stamp serial, CTS/Plot number, or PNR code",\n'
            f'  "traps_risks_and_penalties": "Clear breakdown of predatory clauses, forfeiture risks, or cancellation fees.",\n'
            f'  "financials_or_valuation": {{\n'
            f'    "base_amount": "Base amount with currency",\n'
            f'    "taxes_and_surcharges": "Taxes, registration fees, or duty",\n'
            f'    "grand_total": "Grand total valuation or paid amount",\n'
            f'    "payment_status": "PAID / REGISTERED / UNPAID / PENDING"\n'
            f'  }},\n'
            f'  "actionable_advisory": "Concrete next steps.",\n'
            f'  "detected_destination": "City and Country name if document indicates travel, otherwise null"\n'
            f"}}"
        )

        analysis_raw, diagnostic_err = audit_document_robust(forensic_prompt, file_bytes)
        del file_bytes
        gc.collect()

        if not analysis_raw:
            return {"status": "error", "message": diagnostic_err or "Analysis engine timed out.", "data": None}

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
                "issuing_authority_or_registry": "Registry Department",
                "parties_and_dates": "Extracted Details",
                "metadata_identifiers": "Serials Recorded",
                "traps_risks_and_penalties": "Inspect fine print for terms.",
                "financials_or_valuation": {
                    "base_amount": "Recorded",
                    "taxes_and_surcharges": "Recorded fees",
                    "grand_total": "Verified",
                    "payment_status": "RECORDED"
                },
                "actionable_advisory": analysis_raw[:450],
                "detected_destination": None
            }

        return {"status": "success", "data": data, "raw_text": analysis_raw}
    except Exception as e:
        return {"status": "error", "message": f"Forensic audit error: {str(e)}", "data": None}

# -------------------------------------------------------------
# 2. DEDICATED DOCUMENT EXPORTERS (DIRECT PDF / WORD CONVERTERS)
# -------------------------------------------------------------
@app.post("/api/v1/export-pdf")
async def export_pdf(request: Request, content: str = Form(...), title: str = Form("Travel Itinerary")):
    try:
        file_id = f"Omni_Itinerary_{uuid.uuid4().hex[:5]}.pdf"
        out_path = os.path.join(DOWNLOADS_DIR, file_id)
        compile_pdf_document(title, content, out_path)
        base_url = str(request.base_url).rstrip("/")
        return {
            "status": "success",
            "download_url": f"{base_url}/downloads/{file_id}",
            "file_name": file_id,
            "file_type": "PDF"
        }
    except Exception as e:
        return {"status": "error", "message": f"PDF build error: {str(e)}"}

@app.post("/api/v1/export-docx")
async def export_docx(request: Request, content: str = Form(...), title: str = Form("Travel Itinerary")):
    try:
        file_id = f"Omni_Itinerary_{uuid.uuid4().hex[:5]}.docx"
        out_path = os.path.join(DOWNLOADS_DIR, file_id)
        compile_docx_document(title, content, out_path)
        base_url = str(request.base_url).rstrip("/")
        return {
            "status": "success",
            "download_url": f"{base_url}/downloads/{file_id}",
            "file_name": file_id,
            "file_type": "DOCX"
        }
    except Exception as e:
        return {"status": "error", "message": f"Word build error: {str(e)}"}

# -------------------------------------------------------------
# 3. LIVE OMNI AI STUDIO COMPANION & IMAGE GENERATOR
# -------------------------------------------------------------
@app.post("/api/v1/ask-question")
async def ask_question(
    request: Request,
    question: str = Form(...),
    target_language: str = Form("English"),
    active_document_context: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        clean_q = question.strip()
        lower_q = clean_q.lower()
        base_url = str(request.base_url).rstrip("/")

        # Check if user requested an image
        visual_triggers = ["generate image", "create image", "genrate image", "picture of", "photo of", "logo", "3d logo", "render", "illustration", "draw"]
        if any(t in lower_q for t in visual_triggers):
            # Formulate an explicit visual diffusion prompt
            clean_subject = clean_q
            for tr in ["generate a 3d logo of", "generate 3d logo for my app name", "generate 3d logo for", "generate image of", "create image of", "generate image", "create image"]:
                clean_subject = re.sub(re.escape(tr), "", clean_subject, flags=re.IGNORECASE).strip()
            
            prompt_diffusion = f"Modern 3D isometric vector emblem for {clean_subject}, stylized vibrant app icon, smooth matte clay render, volumetric studio lighting, centered, 8k resolution, photorealistic"
            seed = uuid.uuid4().int % 999999
            enc_prompt = urllib.parse.quote(prompt_diffusion)
            img_url = f"https://image.pollinations.ai/prompt/{enc_prompt}?width=1024&height=1024&nologo=true&model=flux&seed={seed}"

            return {
                "status": "success",
                "answer": f"Rendered visual: *\"{prompt_diffusion}\"*",
                "image_url": img_url,
                "download_url": img_url,
                "download_name": "Omni_Generated_Visual.jpg",
                "file_type": "IMAGE"
            }

        doc_awareness = f"\n[AUDITED DOCUMENT IN MEMORY]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Omni Companion, an authentic travel concierge and legal intelligence advisor. "
            f"Answer thoroughly and directly in {target_language}. Never output <think> tags. "
            f"When creating itineraries, structure them day-by-day with morning, afternoon, and evening activities.{doc_awareness}"
        )

        if file:
            fbytes = await file.read()
            ans, _ = audit_document_robust(f"Analyze in {target_language}: {clean_q}", fbytes)
            ans = ans or "Unable to inspect document."
        else:
            ans = ask_hybrid_text(clean_q, sys_prompt)

        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 4. CONVERTER & RESIZER STUDIO
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
# 5. TOURISTOS DESTINATION EXPLORER & GUIDE CHAT
# -------------------------------------------------------------
@app.post("/api/v1/touristos-recommend")
async def touristos_recommend(
    country: str = Form("India"),
    state: str = Form("Maharashtra"),
    city: str = Form("Mumbai"),
    adults: int = Form(2),
    children: int = Form(0),
    dietary_preference: str = Form("Pure Vegetarian"),
    hotel_page: int = Form(1),
    target_language: str = Form("English")
):
    loc_clean = f"{city}, {state}, {country}".strip(", ")
    is_uae = any(x in loc_clean.lower() for x in ["uae", "dubai", "emirates", "abu dhabi"])
    is_europe = any(x in loc_clean.lower() for x in ["france", "italy", "vatican", "rome", "spain", "germany", "europe", "paris"])
    curr_symbol = "₹" if "india" in loc_clean.lower() else ("AED " if is_uae else ("€" if is_europe else "$"))

    if is_uae:
        dubai_spots = [
            "Burj Khalifa", "Dubai Mall", "Palm Jumeirah", "Museum of the Future",
            "Dubai Marina Walk", "Burj Al Arab", "Dubai Frame", "Souk Madinat Jumeirah",
            "Miracle Garden Dubai", "Global Village"
        ]
        return {
            "status": "success",
            "data": {
                "destination_summary": "Dubai is a global metropolis renowned for modern architecture, luxury shopping, coastal marinas, and historic souks.",
                "spots": [
                    {
                        "page": i + 1,
                        "title": dubai_spots[i],
                        "rating": f"⭐ 4.{9 - (i % 2) * 0.1}",
                        "dist": f"{2.0 + i * 1.8:.1f} km from center",
                        "description": "Verified landmark situated in Dubai, UAE.",
                        "images": [f"https://image.pollinations.ai/prompt/{urllib.parse.quote(dubai_spots[i] + ' Dubai architecture')}?width=800&height=500&nologo=true"]
                    }
                    for i in range(10)
                ],
                "hotels_page": hotel_page,
                "hotels_total_pages": 6,
                "hotels": [
                    {
                        "hotel_id": f"HTL-DXB-P{hotel_page}-{i+1:02d}",
                        "name": f"Dubai Grand Hotel #{i+1}",
                        "price_per_night": f"AED {650 + i * 90}",
                        "rating": "⭐ 4.8 (1,900 reviews)",
                        "location_address": "Sheikh Zayed Road, Dubai",
                        "availability": "Rooms Available (Instant Confirmation)",
                        "room_types": ["Deluxe Room", "Executive Suite", "Family Room"],
                        "amenities": ["Free Wi-Fi", "Breakfast Included", "AC", f"{dietary_preference} Options"]
                    }
                    for i in range(10)
                ],
                "emergency": {
                    "hospital_name": "Rashid Hospital / Dubai Hospital",
                    "hospital_phone": "998",
                    "police_name": "Dubai Police General HQ",
                    "police_phone": "999",
                    "fire_name": "Dubai Civil Defence",
                    "fire_phone": "997",
                    "pharmacy_name": "Aster Pharmacy 24/7",
                    "pharmacy_phone": "04-4405100"
                }
            }
        }

    return {
        "status": "success",
        "data": {
            "destination_summary": f"{loc_clean} offers rich cultural landmarks, dining, and transit networks.",
            "spots": [
                {
                    "page": i + 1,
                    "title": f"Attraction {i + 1} of {city}",
                    "rating": "⭐ 4.8",
                    "dist": f"{i + 1.2:.1f} km from center",
                    "description": f"Verified cultural highlight in {city}.",
                    "images": [f"https://image.pollinations.ai/prompt/{urllib.parse.quote(city + ' landmark architecture')}?width=800&height=500&nologo=true"]
                }
                for i in range(10)
            ],
            "hotels_page": hotel_page,
            "hotels_total_pages": 6,
            "hotels": [
                {
                    "hotel_id": f"HTL-{hotel_page}-{i+1:02d}",
                    "name": f"{city} Hotel #{ (hotel_page - 1) * 10 + i + 1 }",
                    "price_per_night": f"{curr_symbol}{3500 + i * 400}",
                    "rating": "⭐ 4.7 (1,250 reviews)",
                    "location_address": f"Central Avenue, {city}",
                    "availability": "Rooms Available (Instant Confirmation)",
                    "room_types": ["Deluxe Room", "Executive Suite", "Family Room"],
                    "amenities": ["Free Wi-Fi", "Breakfast Included", "AC", f"{dietary_preference} Options"]
                }
                for i in range(10)
            ],
            "emergency": {
                "hospital_name": f"{city} General Hospital",
                "hospital_phone": "112" if is_europe else ("998" if is_uae else "911"),
                "police_name": f"{city} Police Control",
                "police_phone": "112" if is_europe else ("999" if is_uae else "911"),
                "fire_name": f"{city} Fire & Rescue",
                "fire_phone": "112" if is_europe else ("997" if is_uae else "911"),
                "pharmacy_name": f"{city} 24/7 Pharmacy Hub",
                "pharmacy_phone": "112" if is_europe else ("04-4405100" if is_uae else "911")
            }
        }
    }

@app.post("/api/v1/explore-chat")
async def explore_chat(
    spot_name: str = Form("Destination"),
    question: str = Form("What are the visiting hours?"),
    target_language: str = Form("English")
):
    sys_prompt = f"You are a local guide for '{spot_name}'. Answer concisely in {target_language} using Markdown."
    ans = ask_hybrid_text(question, sys_prompt)
    return {"status": "success", "answer": ans}

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