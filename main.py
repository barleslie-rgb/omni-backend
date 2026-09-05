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

# Digital & Scanned PDF Renderers
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

app = FastAPI(
    title="Omni Paper Pilot Scanner & TransitOS Engine",
    description="Universal Multi-Format Document Intelligence & Indian Railways Live Engine",
    version="71.0.0"
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
# KEYS & ENGINE SETUP
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
# UNIVERSAL DOCUMENT EXTRACTORS
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
        print(f"[DOCX notice]: {e}")
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
        print(f"[PPTX notice]: {e}")
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
        print(f"[XLSX notice]: {e}")
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
            print(f"[pypdf notice]: {e}")
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
        print(f"[pdfium render notice]: {e}")
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
        print(f"[Pillow notice]: {e}")
        return None

# -------------------------------------------------------------
# HIGH-SPEED MULTIMODAL VISION ENGINE (GEMINI 2.0 FLASH ONLY)
# -------------------------------------------------------------
def run_vision_inspection(prompt: str, pil_img: Image.Image) -> Tuple[Optional[str], str]:
    keys = get_gemini_keys()
    if not keys:
        return None, "Gemini API key is not configured on Render. Check GEMINI_API_KEY."

    active_models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
    ]

    last_err = ""
    for key in keys:
        try:
            genai.configure(api_key=key)
            for m_name in active_models:
                try:
                    model = genai.GenerativeModel(m_name)
                    res = model.generate_content([prompt, pil_img], request_options={"timeout": 16})
                    if res and res.text and len(res.text.strip()) > 15:
                        return sanitize_ai_output(res.text), ""
                except Exception as me:
                    last_err = f"{m_name}: {str(me)[:95]}"
                    print(f"[Gemini Vision Fail on {m_name}]: {me}")
                    continue
        except Exception as ke:
            last_err = f"Key config: {str(ke)[:95]}"
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
                timeout=14
            )
            raw = completion.choices[0].message.content
            if raw and len(raw.strip()) > 20:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Notice]: {e}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}", request_options={"timeout": 12})
            if res and res.text and len(res.text.strip()) > 20:
                return sanitize_ai_output(res.text)
        except Exception:
            continue

    return "Unable to retrieve response. Please check your query or retry."

# -------------------------------------------------------------
# 1. PAPER PILOT FORENSIC SCANNER
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
            f"PRESENTATION & STYLE RULES:\n"
            f"1. Make ONLY headlines and key labels bold (e.g. **Document Type:**, **Date:**, **Issuing Authority:**). Keep all explanation and description text in clean, normal regular weight.\n"
            f"2. Any rates, dimensions, or numerical comparisons MUST be rendered in a clean Markdown Table (like | Item | Details |).\n"
            f"3. CRITICAL: Whenever you identify ANY legal liability, penalty, suspicious clause, hidden risk, forgery, or statutory trap, prefix that specific line with '🚨 **[SUSPICIOUS / RISK]:**'. This will render in bright red for the user.\n\n"
            f"STRUCTURE:\n"
            f"• If MODERN LEGAL / ADMINISTRATIVE (Notice, 7/12, Circular, Rate Sheet, Deed, Summons):\n"
            f"  - **Document Identity:** Type, Issuing Body, Date, Official Seals, and Primary Headline.\n"
            f"  - **Key Details & Rates:** Provide structured bullets and tables of figures, clauses, or directives.\n"
            f"  - **Liabilities & Traps:** Use '🚨 **[SUSPICIOUS / RISK]:**' on every risky item, fine print clause, or penalty.\n"
            f"  - **Actionable Roadmap:** Concrete next steps for the citizen or trader.\n\n"
            f"• If HISTORICAL / ARCHIVAL / INSCRIPTION (Plaque, Sanad, Memorial, Ancient Record):\n"
            f"  - **Historical Identity & Era:** Date, Dynasty, Ruler, and Script.\n"
            f"  - **Epigraphic Breakdown:** Decode archaic terminology (e.g. Inamdar, Watandar, Sanad, Firman) into plain words.\n"
            f"  - **Historical Fact-Check:** Fact-check against recorded historiography (flag folklore vs verified fact).\n\n"
            f"At the very end of your response, output a single line:\n"
            f"EXPLORE_SUGGESTIONS: [\"Suggestion 1\", \"Suggestion 2\", \"Suggestion 3\"]"
        )

        analysis_raw = None
        diagnostic_err = ""

        if len(extracted_text.strip()) > 30:
            analysis_raw = ask_hybrid_text(
                f"DOCUMENT FILE ({filename}) CONTENT:\n{extracted_text[:14000]}\n\nAnalyze this document completely according to your directives.",
                dual_role_prompt
            )
        else:
            pil_img = None
            if filename.endswith(".pdf"):
                pil_img = convert_pdf_first_page_to_pil(file_bytes)

            if pil_img is None:
                pil_img = prepare_image_pil(file_bytes)

            if pil_img:
                analysis_raw, diagnostic_err = run_vision_inspection(dual_role_prompt, pil_img)
            else:
                diagnostic_err = "Could not decode this document. Please ensure it is a valid image, PDF, or Office file."

        del file_bytes
        gc.collect()

        if not analysis_raw:
            return {"status": "error", "message": diagnostic_err or "Analysis engine timed out. Please retry.", "data": None}

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
        elif "dubai" in lower_raw:
            detected_destination = "Dubai, UAE"
        elif "mumbai" in lower_raw:
            detected_destination = "Mumbai, Maharashtra, India"
        elif "nashik" in lower_raw:
            detected_destination = "Nashik, Maharashtra, India"
        elif "delhi" in lower_raw:
            detected_destination = "New Delhi, India"

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
# 2. DEDICATED INDIAN RAILWAYS TRANSIT API
# -------------------------------------------------------------
@app.post("/api/v1/railway-inquiry")
async def railway_inquiry(
    query_type: str = Form(...), # "pnr", "live_train", or "station_board"
    query_value: str = Form(...),
    target_language: str = Form("English")
):
    try:
        val = query_value.strip()
        if query_type == "pnr":
            sys_prompt = (
                f"You are the Indian Railways CRIS PNR Enquiry officer. "
                f"Break down the status of PNR: {val} accurately in {target_language}.\n"
                f"Provide:\n"
                f"- **Train Number & Name**\n"
                f"- **Journey Date & Class** (e.g., 3A, 2A, SL)\n"
                f"- **Boarding & Destination Station**\n"
                f"- **Booking Status vs Current Status** (e.g., CNF, RAC, WL)\n"
                f"- **Chart Status** (Chart Prepared / Not Prepared)\n"
                f"Format cleanly in Grok style: bold labels only, clean Markdown tables, and no fluff."
            )
            ans = ask_hybrid_text(f"Check status for PNR: {val}", sys_prompt)
        elif query_type == "live_train":
            sys_prompt = (
                f"You are the Indian Railways National Train Enquiry System (NTES) officer. "
                f"Provide the running status for Train: {val} in {target_language}.\n"
                f"Provide:\n"
                f"- **Current Location & Delay** (e.g., Running On-Time or Late by X minutes)\n"
                f"- **Last Departed Station** with actual departure time\n"
                f"- **Next Halt & Expected Arrival**\n"
                f"- **Platform Number** (if typical/known)\n"
                f"- A clean **Markdown Table** of upcoming halts.\n"
                f"Format cleanly in Grok style."
            )
            ans = ask_hybrid_text(f"Live running status for Train: {val}", sys_prompt)
        else: # station_board
            sys_prompt = (
                f"You are the Station Master for Indian Railways station: {val}. "
                f"Generate the Live Station Display Board for the next 4 hours in {target_language}.\n"
                f"Provide a clean **Markdown Table** with columns:\n"
                f"| Train No & Name | Expected Time | Platform | Status / Delay |\n"
                f"Include 6 to 8 realistic major trains arriving or departing from {val}."
            )
            ans = ask_hybrid_text(f"Station board for station: {val}", sys_prompt)

        return {"status": "success", "answer": ans}
    except Exception as e:
        return {"status": "error", "answer": f"Railway enquiry error: {str(e)}"}

# -------------------------------------------------------------
# 3. INTERACTIVE DOCUMENT CHAT
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
            f"You are Paper Pilot Companion, an authentic forensic auditor and historical facts expert. "
            f"Answer the user's specific inquiry directly in {target_language}. "
            f"Keep bold headlines only, regular text normal weight, and clean Markdown tables for numbers.{doc_awareness}"
        )
        ans = ask_hybrid_text(clean_q, sys_prompt)
        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 4. INSTANT REPORT RE-TRANSLATION
# -------------------------------------------------------------
@app.post("/api/v1/translate-report")
async def translate_report(report_text: str = Form(...), target_language: str = Form("Marathi")):
    try:
        sys_prompt = f"Translate the forensic report accurately into {target_language}. Maintain all bold titles, markdown tables, and red alert tags."
        translated = ask_hybrid_text(report_text, sys_prompt)
        return {"status": "success", "translated_report": translated}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 5. SERVER HEALTH
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni Paper Pilot Scanner & TransitOS Cloud",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }