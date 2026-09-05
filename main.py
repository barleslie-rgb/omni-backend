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

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq
import google.generativeai as genai

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

app = FastAPI(
    title="Omni Paper Pilot Scanner & TouristOS Engine",
    description="Dedicated Multimodal Forensic Analysis & Pure Tourism Discovery",
    version="73.0.0"
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
# KEYS & FAST INFERENCE SETUP
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
# DOCUMENT EXTRACTORS
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

def extract_text_from_pdf_stream(file_bytes: bytes) -> str:
    extracted = ""
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages[:12]:
                t = page.extract_text() or ""
                extracted += t + "\n"
        except Exception as e:
            print(f"[pypdf error]: {e}")
    return extracted.strip()

def convert_pdf_first_page_to_pil(file_bytes: bytes) -> Optional[Image.Image]:
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
        return pil_img
    except Exception as e:
        print(f"[pdfium error]: {e}")
        return None

def prepare_image_pil(file_bytes: bytes) -> Optional[Image.Image]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > 1024:
            pil_img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        return pil_img
    except Exception as e:
        print(f"[Pillow error]: {e}")
        return None

# -------------------------------------------------------------
# DIRECT GEMINI 2.0 FLASH VISION ONLY (NO RETIRED 1.5 CALLS)
# -------------------------------------------------------------
def run_vision_inspection(prompt: str, pil_img: Image.Image) -> Tuple[Optional[str], str]:
    keys = get_gemini_keys()
    if not keys:
        return None, "Gemini API key is not configured on Render. Check GEMINI_API_KEY."

    # Strict target on Gemini 2.0 Flash (active generation model)
    target_models = ["gemini-2.0-flash", "gemini-2.0-flash-exp"]
    last_err = ""

    for key in keys:
        try:
            genai.configure(api_key=key)
            for m_name in target_models:
                try:
                    model = genai.GenerativeModel(m_name)
                    res = model.generate_content([prompt, pil_img], request_options={"timeout": 20})
                    if res and res.text and len(res.text.strip()) > 15:
                        return sanitize_ai_output(res.text), ""
                except Exception as me:
                    last_err = f"{m_name}: {str(me)[:95]}"
                    print(f"[Gemini Vision notice on {m_name}]: {me}")
                    continue
        except Exception as ke:
            last_err = f"Key notice: {str(ke)[:95]}"
            continue

    return None, f"Vision notice ({last_err})"

def ask_hybrid_text(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if client:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3000,
                timeout=15
            )
            raw = completion.choices[0].message.content
            if raw and len(raw.strip()) > 10:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Error]: {e}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}", request_options={"timeout": 14})
            if res and res.text and len(res.text.strip()) > 10:
                return sanitize_ai_output(res.text)
        except Exception:
            continue

    return "Query processed. Please ask your next question."

def ask_hybrid_json(prompt: str, system_prompt: str) -> Optional[dict]:
    client = get_groq_client()
    if client:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3500,
                response_format={"type": "json_object"},
                timeout=16
            )
            raw = completion.choices[0].message.content
            if raw:
                return json.loads(sanitize_ai_output(raw))
        except Exception as e:
            print(f"[Groq JSON Error]: {e}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            res = model.generate_content(
                f"{system_prompt}\n\nStrictly return valid JSON only.\nUser: {prompt}",
                request_options={"timeout": 14}
            )
            if res and res.text:
                clean = sanitize_ai_output(res.text)
                if "```json" in clean:
                    clean = clean.split("```json")[1].split("```")[0].strip()
                elif "```" in clean:
                    clean = clean.split("```")[1].split("```")[0].strip()
                return json.loads(clean)
        except Exception:
            continue

    return None

# -------------------------------------------------------------
# 1. PAPER PILOT FORENSIC & HISTORICAL SCANNER
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
        if filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".pptx"):
            extracted_text = extract_text_from_pptx(file_bytes)
        elif filename.endswith(".xlsx"):
            extracted_text = extract_text_from_xlsx(file_bytes)
        elif filename.endswith(".csv") or filename.endswith(".txt") or filename.endswith(".json") or filename.endswith(".md"):
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                pass
        elif filename.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
            extracted_text = extract_text_from_pdf_stream(file_bytes)

        dual_role_prompt = (
            f"You are Paper Pilot, an authentic Forensic Document Auditor and Historical Facts Examiner. "
            f"Thoroughly analyze and deconstruct this document/image in {target_language}.\n\n"
            f"PRESENTATION RULES:\n"
            f"1. Make ONLY headlines and key labels bold (e.g. **Document Type:**, **Date:**, **Issuing Authority:**). Regular text must be normal weight.\n"
            f"2. Numerical comparisons or rate sheets MUST be rendered in a clean Markdown Table.\n"
            f"3. Mark any fine print, statutory trap, or legal penalty with '🚨 **[SUSPICIOUS / RISK]:**'.\n\n"
            f"STRUCTURE:\n"
            f"• If MODERN LEGAL / ADMINISTRATIVE: Document Identity, Key Details & Rates Table, Liabilities & Traps, and Actionable Roadmap.\n"
            f"• If HISTORICAL / ARCHIVAL: Historical Era, Epigraphic Breakdown, and Historiography Fact-Check.\n\n"
            f"End your audit with a single line:\n"
            f"EXPLORE_SUGGESTIONS: [\"Suggestion 1\", \"Suggestion 2\", \"Suggestion 3\"]"
        )

        analysis_raw = None
        diagnostic_err = ""

        # Route A: Digital text in document
        if len(extracted_text.strip()) > 30:
            analysis_raw = ask_hybrid_text(
                f"DOCUMENT FILE ({filename}) CONTENT:\n{extracted_text[:14000]}\n\nAnalyze this document completely according to your directives.",
                dual_role_prompt
            )
        else:
            # Route B: Scanned PDF or Camera Photo (Direct to Gemini 2.0 Flash)
            pil_img = None
            if filename.endswith(".pdf"):
                pil_img = convert_pdf_first_page_to_pil(file_bytes)

            if pil_img is None:
                pil_img = prepare_image_pil(file_bytes)

            if pil_img:
                analysis_raw, diagnostic_err = run_vision_inspection(dual_role_prompt, pil_img)
            else:
                diagnostic_err = "Could not process this document format."

        del file_bytes
        gc.collect()

        if not analysis_raw:
            return {"status": "error", "message": diagnostic_err or "Analysis engine timed out. Please retry.", "data": None}

        suggestions = [
            "Verify issuing authority contact numbers",
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
        elif "dubai" in lower_raw:
            detected_destination = "Dubai, UAE"
        elif "mumbai" in lower_raw:
            detected_destination = "Mumbai, Maharashtra, India"

        return {
            "status": "success",
            "data": {
                "document_title": "Forensic & Historical Report",
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
                f"Include: Train Number/Name, Journey Date, Class, Stations, Booking vs Current Status, Chart Status in Grok-style tables."
            )
            ans = ask_hybrid_text(f"PNR Status for: {val}", sys_prompt)
        elif query_type == "live_train":
            sys_prompt = (
                f"You are the Indian Railways NTES officer. "
                f"Provide running status for Train: {val} in {target_language}.\n"
                f"Include: Current Location, Delay in minutes, Last Departed Station, Next Halt, and Halts Table in Grok-style."
            )
            ans = ask_hybrid_text(f"Live status of Train: {val}", sys_prompt)
        else:
            sys_prompt = (
                f"You are the Station Master for Indian Railways station: {val}. "
                f"Generate the Live Station Display Board for next 4 hours in {target_language} with Markdown columns: | Train No & Name | Expected Time | Platform | Status |."
            )
            ans = ask_hybrid_text(f"Station board for: {val}", sys_prompt)

        return {"status": "success", "answer": ans}
    except Exception as e:
        return {"status": "error", "answer": f"Transit enquiry error: {str(e)}"}

# -------------------------------------------------------------
# 3. INTERACTIVE CHAT & FOLLOW-UPS
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
        doc_awareness = f"\n[DOCUMENT CONTEXT]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Paper Pilot Companion, an authentic forensic document auditor and historical expert. "
            f"Answer the user's inquiry directly in {target_language}. "
            f"Keep bold headlines only, regular text normal weight, and clean Markdown tables for numbers.{doc_awareness}"
        )
        ans = ask_hybrid_text(clean_q, sys_prompt)
        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 4. INSTANT MULTILINGUAL TRANSLATION
# -------------------------------------------------------------
@app.post("/api/v1/translate-report")
async def translate_report(report_text: str = Form(...), target_language: str = Form("Marathi")):
    try:
        sys_prompt = f"Translate the forensic report into {target_language}. Retain bold labels, markdown tables, and red alerts."
        translated = ask_hybrid_text(report_text, sys_prompt)
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
    city: str = Form("Vasai"),
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
        f"You are the local destination officer for '{loc_clean}'.\n"
        f"Return strict JSON with authentic landmarks, local cultural facts, sightseeing rules, and emergency numbers. Do NOT include train advertisements."
    )

    user_query = f"Scan geographic directory for {loc_clean}. Provide 6-8 real landmarks and municipal emergency contacts."
    extracted_data = ask_hybrid_json(user_query, system_prompt)

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
                    "title": f"Historic Center of {city}",
                    "category": "Cultural Heritage",
                    "rating": "⭐ 4.9",
                    "dist": "Central District",
                    "description": f"The architectural and cultural heart of {city}.",
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
    city: str = Form("Vasai"),
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
        response_text = ask_hybrid_text(clean_q, sys_prompt)
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
        "service": "Omni TouristOS Cloud",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }