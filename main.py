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
import xml.etree.ElementTree as ET

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq

# Digital & Scanned PDF Processors
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

app = FastAPI(
    title="Omni Paper Pilot Scanner & Unified Intelligence Cloud",
    description="Direct REST Vision, Multi-Page Legal Document Engine & Forensic Auditor",
    version="76.0.0"
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
# CREDENTIAL MANAGEMENT & SANITIZATION
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    raw = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    return Groq(api_key=raw) if raw else None

def get_gemini_keys() -> List[str]:
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    keys = []
    for k in raw.split(","):
        cleaned = k.strip().strip('"').strip("'")
        if cleaned:
            keys.append(cleaned)
    return keys

def sanitize_ai_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# DIRECT REST CALL FOR ACTIVE GEMINI FLASH PRODUCTION MODELS
# -------------------------------------------------------------
async def call_gemini_rest_vision(prompt: str, img_bytes: bytes, mime_type: str = "image/jpeg") -> Tuple[Optional[str], str]:
    keys = get_gemini_keys()
    if not keys:
        return None, "Gemini API key is not configured on Render. Check GEMINI_API_KEY."

    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    last_err = ""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096
        }
    }

    # Active production models on Google AI Studio
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash",
    ]

    async with httpx.AsyncClient(timeout=35.0) as client:
        for key in keys:
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                try:
                    res = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text_pieces = [p.get("text", "") for p in parts if "text" in p]
                            ans = "".join(text_pieces).strip()
                            if len(ans) > 20:
                                return sanitize_ai_output(ans), ""
                    else:
                        last_err = f"HTTP {res.status_code} ({model_name}): {res.text[:120]}"
                        print(f"[Gemini REST Error {model_name}]: {last_err}")
                except Exception as ex:
                    last_err = f"{model_name} exception: {str(ex)[:100]}"
                    print(f"[Gemini REST Exception {model_name}]: {last_err}")
                    continue

    return None, f"Vision notice ({last_err})"

# -------------------------------------------------------------
# FAST TEXT ENGINE (GROQ LLAMA-3.3-70B WITH REST FALLBACK)
# -------------------------------------------------------------
async def ask_fast_text(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if client:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
                timeout=18
            )
            raw = completion.choices[0].message.content
            if raw and len(raw.strip()) > 10:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Notice]: {e}")

    keys = get_gemini_keys()
    if keys:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\nUser Query: {prompt}"}
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 3500}
        }
        async with httpx.AsyncClient(timeout=20.0) as http_client:
            for key in keys:
                for m in ["gemini-2.5-flash", "gemini-3.5-flash"]:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
                    try:
                        res = await http_client.post(url, json=payload)
                        if res.status_code == 200:
                            candidates = res.json().get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                ans = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
                                if len(ans) > 10:
                                    return sanitize_ai_output(ans)
                    except Exception:
                        continue

    return "Document inspection complete. Review the forensic breakdown above or ask a specific follow-up question."

async def ask_fast_json(prompt: str, system_prompt: str) -> Optional[dict]:
    client = get_groq_client()
    if client:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3500,
                response_format={"type": "json_object"},
                timeout=18
            )
            raw = completion.choices[0].message.content
            if raw:
                return json.loads(sanitize_ai_output(raw))
        except Exception as e:
            print(f"[Groq JSON Notice]: {e}")
    return None

# -------------------------------------------------------------
# MASSIVE MULTI-PAGE DOCUMENT PARSERS (UP TO 200+ PAGES)
# -------------------------------------------------------------
def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            xml_content = zf.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        print(f"[DOCX error]: {e}")
        return ""

def extract_text_from_pptx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            slides = sorted([n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
            all_text = []
            for slide_name in slides:
                xml_content = zf.read(slide_name)
                tree = ET.fromstring(xml_content)
                slide_texts = [node.text for node in tree.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t') if node.text]
                if slide_texts:
                    all_text.append(" • " + " ".join(slide_texts))
            return "\n\n".join(all_text)
    except Exception as e:
        print(f"[PPTX error]: {e}")
        return ""

def extract_text_from_xlsx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                ss_xml = zf.read("xl/sharedStrings.xml")
                tree = ET.fromstring(ss_xml)
                for si in tree.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    t_nodes = [node.text for node in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if node.text]
                    shared_strings.append("".join(t_nodes))

            sheets = sorted([n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")])
            table_output = []
            for s_name in sheets:
                xml_content = zf.read(s_name)
                tree = ET.fromstring(xml_content)
                for row in tree.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                    row_vals = []
                    for c in row.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                        cell_type = c.attrib.get('t')
                        v_node = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                        if v_node is not None and v_node.text:
                            val = v_node.text
                            if cell_type == 's' and val.isdigit():
                                idx = int(val)
                                if idx < len(shared_strings):
                                    val = shared_strings[idx]
                            row_vals.append(val)
                    if row_vals:
                        table_output.append(" | ".join(row_vals))
            return "\n".join(table_output)
    except Exception as e:
        print(f"[XLSX error]: {e}")
        return ""

def extract_massive_pdf_text(file_bytes: bytes, max_pages: int = 250) -> Tuple[str, int]:
    if PdfReader is None:
        return "", 0
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)

        extracted_chunks = []
        for i in range(pages_to_read):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text and page_text.strip():
                    extracted_chunks.append(f"--- [PAGE {i+1} OF {total_pages}] ---\n{page_text.strip()}")
            except Exception:
                continue

        full_extracted = "\n\n".join(extracted_chunks)
        return full_extracted.strip(), total_pages
    except Exception as e:
        print(f"[pypdf extraction error]: {e}")
        return "", 0

def render_scanned_pdf_first_page(file_bytes: bytes) -> Optional[bytes]:
    if pdfium is None:
        return None
    try:
        pdf = pdfium.PdfDocument(file_bytes)
        if len(pdf) == 0:
            return None
        page = pdf[0]
        pil_img = page.render(scale=1.5).to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > 1200:
            pil_img.thumbnail((1200, 1200), Image.Resampling.BILINEAR)
        out_buf = io.BytesIO()
        pil_img.save(out_buf, format="JPEG", quality=90)
        return out_buf.getvalue()
    except Exception as e:
        print(f"[pdfium render error]: {e}")
        return None

def prepare_image_bytes(file_bytes: bytes) -> Optional[bytes]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > 1200:
            pil_img.thumbnail((1200, 1200), Image.Resampling.BILINEAR)
        out_buf = io.BytesIO()
        pil_img.save(out_buf, format="JPEG", quality=90)
        return out_buf.getvalue()
    except Exception as e:
        print(f"[Pillow error]: {e}")
        return None

# -------------------------------------------------------------
# 1. UNIVERSAL DOCUMENT SCANNER & FORENSIC / LEGAL AUDITOR
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        filename = (file.filename or "").lower()

        extracted_text = ""
        total_pages_detected = 1

        if filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".pptx"):
            extracted_text = extract_text_from_pptx(file_bytes)
        elif filename.endswith(".xlsx"):
            extracted_text = extract_text_from_xlsx(file_bytes)
        elif any(filename.endswith(ext) for ext in [".csv", ".txt", ".json", ".md"]):
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                pass
        elif filename.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
            extracted_text, total_pages_detected = extract_massive_pdf_text(file_bytes, max_pages=250)

        dual_role_prompt = (
            f"You are Paper Pilot, an authentic Forensic Legal Auditor and Historical Facts Examiner. "
            f"Thoroughly analyze and deconstruct this document in {target_language}.\n\n"
            f"PRESENTATION & STYLE RULES:\n"
            f"1. Make ONLY headlines and key labels bold (e.g. **Document Type:**, **Issuing Authority:**, **Effective Date:**). Descriptions must be in regular weight.\n"
            f"2. Any rates, dimensions, schedules, penalties, or numerical comparisons MUST be rendered in a clean Markdown Table (like | Clause / Section | Detail | Liability |).\n"
            f"3. CRITICAL: Whenever you identify ANY legal liability, penalty, suspicious clause, indemnity risk, arbitration trap, or statutory catch, prefix that line with '🚨 **[SUSPICIOUS / RISK]:**'. This renders in bright red for the user.\n\n"
            f"STRUCTURE:\n"
            f"• **Document Identity:** Type, Issuing Body, Document Date, Parties Involved, Official Seals, and Primary Headline.\n"
            f"• **Scope & Multi-Page Summary:** Outline the overall legal covenants across sections.\n"
            f"• **Key Clauses, Tables & Directives:** Provide structured bullets and tables of terms, dates, and covenants.\n"
            f"• **Liabilities, Traps & Fine Print:** List every risky item prefixed with '🚨 **[SUSPICIOUS / RISK]:**'.\n"
            f"• **Actionable Roadmap:** Concrete next steps for the citizen, advocate, or signatory.\n\n"
            f"At the very end of your response, output a single line:\n"
            f"EXPLORE_SUGGESTIONS: [\"Verify issuing authority credentials\", \"Examine legal precedents\", \"Save document voucher to Family Travel Vault\"]"
        )

        analysis_raw = None
        diagnostic_err = ""

        # Path A: Digital Document
        if len(extracted_text.strip()) > 30:
            doc_context_header = f"DOCUMENT FILE: {filename} (Total Pages: {total_pages_detected})\n\n"
            truncated_content = extracted_text[:80000]
            analysis_raw = await ask_fast_text(
                f"{doc_context_header}{truncated_content}\n\nConduct full forensic legal audit according to your directives.",
                dual_role_prompt
            )
        else:
            # Path B: Scanned PDF or Image File
            img_bytes = None
            if filename.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
                img_bytes = render_scanned_pdf_first_page(file_bytes)
            if img_bytes is None:
                img_bytes = prepare_image_bytes(file_bytes)

            if img_bytes:
                analysis_raw, diagnostic_err = await call_gemini_rest_vision(
                    prompt=dual_role_prompt,
                    img_bytes=img_bytes,
                    mime_type="image/jpeg"
                )
            else:
                diagnostic_err = "Could not decode this file format. Please ensure it is a valid PDF, Word, Excel, PowerPoint, or Image."

        del file_bytes
        gc.collect()

        if not analysis_raw:
            return {
                "status": "error",
                "message": diagnostic_err or "Analysis engine encountered a timeout. Please retry.",
                "data": None
            }

        suggestions = [
            "Verify official authority contact numbers",
            "Examine legal precedent and historical records",
            "Save document voucher to Family Travel Vault"
        ]

        clean_text = analysis_raw
        if "EXPLORE_SUGGESTIONS:" in analysis_raw:
            parts = analysis_raw.split("EXPLORE_SUGGESTIONS:")
            clean_text = parts[0].strip()
            try:
                parsed_sugg = json.loads(parts[1].strip())
                if isinstance(parsed_sugg, list) and len(parsed_sugg) > 0:
                    suggestions = [str(s) for s in parsed_sugg[:4]]
            except Exception:
                pass

        detected_destination = None
        lower_raw = clean_text.lower()
        if "vasai" in lower_raw or "virar" in lower_raw:
            detected_destination = "Vasai, Maharashtra, India"
        elif "pune" in lower_raw:
            detected_destination = "Pune, Maharashtra, India"
        elif "mumbai" in lower_raw:
            detected_destination = "Mumbai, Maharashtra, India"
        elif "dubai" in lower_raw:
            detected_destination = "Dubai, UAE"

        return {
            "status": "success",
            "data": {
                "document_title": f"Forensic Audit ({filename})",
                "actionable_advisory": clean_text,
                "detected_destination": detected_destination,
                "suggestions": suggestions
            },
            "raw_text": clean_text
        }
    except Exception as e:
        return {"status": "error", "message": f"Scan error: {str(e)}", "data": None}

# -------------------------------------------------------------
# 2. STANDALONE INDIAN RAILWAYS TRANSIT API
# -------------------------------------------------------------
@app.post("/api/v1/railway-inquiry")
async def railway_inquiry(
    query_type: str = Form(...),
    query_value: str = Form(...),
    target_language: str = Form("English")
):
    try:
        val = query_value.strip()
        if query_type == "pnr":
            sys_prompt = (
                f"You are the Indian Railways CRIS PNR Enquiry officer. "
                f"Break down the status of PNR: {val} in {target_language}.\n"
                f"Include Train Name, Number, Journey Date, Class, Boarding/Destination, Booking Status vs Current Status (CNF/WL/RAC), and Chart Status in a clean Grok Markdown table."
            )
            ans = await ask_fast_text(f"PNR Status inquiry: {val}", sys_prompt)
        elif query_type == "live_train":
            sys_prompt = (
                f"You are the Indian Railways NTES live tracking officer. "
                f"Provide running status for Train: {val} in {target_language}.\n"
                f"Provide current station location, delay in minutes, next halt, platform number, and upcoming schedule table in Grok style."
            )
            ans = await ask_fast_text(f"Live status of train: {val}", sys_prompt)
        else:
            sys_prompt = (
                f"You are the Station Master for Indian Railways station: {val}. "
                f"Generate the Live Station Display Board for the next 4 hours in {target_language} with Markdown columns: | Train No & Name | Expected Time | Platform | Status |."
            )
            ans = await ask_fast_text(f"Station board for station: {val}", sys_prompt)

        return {"status": "success", "answer": ans}
    except Exception as e:
        return {"status": "error", "answer": f"Transit error: {str(e)}"}

# -------------------------------------------------------------
# 3. INTERACTIVE CHAT & INQUIRY
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
        doc_awareness = f"\n[AUDITED DOCUMENT CONTEXT]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Paper Pilot Companion, an authentic forensic legal auditor and document expert. "
            f"Answer the user's specific inquiry directly in {target_language}. "
            f"Maintain Grok presentation: bold headers only, normal body text, clean Markdown tables for numbers or clauses.{doc_awareness}"
        )
        ans = await ask_fast_text(clean_q, sys_prompt)
        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 4. INSTANT MULTILINGUAL REPORT TRANSLATOR
# -------------------------------------------------------------
@app.post("/api/v1/translate-report")
async def translate_report(report_text: str = Form(...), target_language: str = Form("Marathi")):
    try:
        sys_prompt = f"Translate the forensic report into {target_language}. Retain bold labels, markdown tables, and red alerts."
        translated = await ask_fast_text(report_text, sys_prompt)
        return {"status": "success", "translated_report": translated}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 5. PURE TOURISTOS DESTINATION EXPLORER (NO RAILWAY ADS)
# -------------------------------------------------------------
@app.post("/api/v1/touristos-recommend")
async def touristos_recommend(
    country: str = Form("India"),
    state: str = Form("Maharashtra"),
    city: str = Form("Pune"),
    adults: int = Form(2),
    children: int = Form(0),
    dietary_preference: str = Form("All / Any"),
    target_language: str = Form("English")
):
    loc_clean = f"{city}, {state}, {country}".strip(", ")
    lower_loc = loc_clean.lower()

    if any(x in lower_loc for x in ["denmark", "copenhagen"]):
        curr_sym = "DKK "
        nat_police, nat_hosp, nat_fire, nat_pharm = "112", "1813 / 112", "112", "Steno Apotek 24/7"
    elif any(x in lower_loc for x in ["israel", "jerusalem", "tel aviv"]):
        curr_sym = "₪"
        nat_police, nat_hosp, nat_fire, nat_pharm = "100", "101", "102", "Super-Pharm 24/7"
    elif any(x in lower_loc for x in ["united states", "usa", "us", "new york"]):
        curr_sym = "$"
        nat_police, nat_hosp, nat_fire, nat_pharm = "911", "911", "911", "311 / 1-800-222-1222"
    elif any(x in lower_loc for x in ["france", "italy", "germany", "spain", "europe"]):
        curr_sym = "€"
        nat_police, nat_hosp, nat_fire, nat_pharm = "112", "112", "112", "112 / 24/7 Chemist"
    else:
        curr_sym = "₹"
        nat_police, nat_hosp, nat_fire, nat_pharm = "112 / 100", "108 / 102", "101", "1800-200-1234"

    system_prompt = (
        f"You are the senior local tourism officer for '{loc_clean}'.\n"
        f"Generate authentic landmarks, heritage spots, cultural highlights, and emergency contacts in strict JSON. Do NOT include train advertisements."
    )

    user_query = f"Scan geographic database for {loc_clean}. Provide 6-8 real landmarks and municipal emergency facilities."
    extracted_data = await ask_fast_json(user_query, system_prompt)

    if extracted_data and "spots" in extracted_data and len(extracted_data["spots"]) > 0:
        spots = extracted_data["spots"]
        for sp in spots:
            t_title = sp.get("title", city)
            seed = abs(hash(t_title + city)) % 999999
            enc_t = urllib.parse.quote(f"Scenic daylight photography of {t_title} {city} {country}, 8k")
            sp["images"] = [f"https://image.pollinations.ai/prompt/{enc_t}?width=800&height=500&nologo=true&seed={seed}&model=flux"]

        emg = extracted_data.get("emergency", {})
        if not emg.get("police_phone"): emg["police_phone"] = nat_police
        if not emg.get("hospital_phone"): emg["hospital_phone"] = nat_hosp
        if not emg.get("fire_phone"): emg["fire_phone"] = nat_fire
        if not emg.get("pharmacy_phone"): emg["pharmacy_phone"] = nat_pharm

        return {"status": "success", "data": extracted_data}

    return {
        "status": "success",
        "data": {
            "destination_summary": f"{city} ({country}) travel guide.",
            "spots": [
                {
                    "page": 1,
                    "title": f"Historic Center & Citadel of {city}",
                    "category": "Cultural Heritage",
                    "rating": "⭐ 4.9",
                    "dist": "Central District",
                    "description": f"The iconic architectural and cultural heart of {city}.",
                    "history": f"Recorded extensively in historical annals.",
                    "sightseeing_rules": "Respect local cultural etiquette.",
                    "culinary": f"Traditional cuisine matching {dietary_preference}.",
                    "transit": f"Accessible via {city} municipal cabs and public buses.",
                    "speciality": f"Historic landmarks and architecture.",
                    "images": [f"https://image.pollinations.ai/prompt/Scenic%20daylight%20photography%20of%20historic%20{urllib.parse.quote(city)}?width=800&height=500&nologo=true&seed=101&model=flux"]
                }
            ],
            "emergency": {
                "hospital_name": f"{city} Central Hospital",
                "hospital_phone": nat_hosp,
                "police_name": f"{city} Police",
                "police_phone": nat_police,
                "fire_name": f"{city} Fire Station",
                "fire_phone": nat_fire,
                "pharmacy_name": f"{city} 24/7 Chemist",
                "pharmacy_phone": nat_pharm
            }
        }
    }

# -------------------------------------------------------------
# 6. CONCIERGE GUIDE CHAT
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(
    request: Request,
    city: str = Form("Pune"),
    country: str = Form("India"),
    party_summary: str = Form("2 Adults"),
    dietary_preference: str = Form("All / Any"),
    question: str = Form("Plan itinerary"),
    target_language: str = Form("English")
):
    try:
        clean_q = question.strip()
        loc_label = f"{city}, {country}".strip(", ")
        sys_prompt = f"You are Omni Guide for '{loc_label}'. Travelers: {party_summary}. Diet: '{dietary_preference}'. Language: {target_language}."
        response_text = await ask_fast_text(clean_q, sys_prompt)
        return {"status": "success", "answer": response_text, "has_document": False}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 7. SERVER HEALTH
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni Paper Pilot Scanner & Unified Intelligence Cloud",
        "version": "76.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }