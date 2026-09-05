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
    title="Omni Paper Pilot Scanner, TouristOS & Railways Transit Engine",
    description="Unified Multimodal Forensic, Travel, and Transit Intelligence Platform",
    version="72.0.0"
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
# SANITIZED CREDENTIAL HANDLING
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
# DOCUMENT TEXT EXTRACTORS
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
# ROBUST MULTIMODAL VISION CALL
# -------------------------------------------------------------
def run_vision_inspection(prompt: str, pil_img: Image.Image) -> Tuple[Optional[str], str]:
    keys = get_gemini_keys()
    if not keys:
        return None, "Gemini API key is not configured on Render. Please verify GEMINI_API_KEY environment variable."

    target_models = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash"]
    last_err = ""

    for key in keys:
        try:
            genai.configure(api_key=key)
            for m_name in target_models:
                try:
                    model = genai.GenerativeModel(m_name)
                    res = model.generate_content([prompt, pil_img], request_options={"timeout": 18})
                    if res and res.text and len(res.text.strip()) > 15:
                        return sanitize_ai_output(res.text), ""
                except Exception as me:
                    last_err = f"{m_name}: {str(me)[:95]}"
                    print(f"[Gemini Vision on {m_name}]: {me}")
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
            if raw and len(raw.strip()) > 10:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Notice]: {e}")

    for key in get_gemini_keys():
        try:
            genai.configure(api_key=key)
            for m in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(m)
                    res = model.generate_content(f"{system_prompt}\n\nUser: {prompt}", request_options={"timeout": 12})
                    if res and res.text and len(res.text.strip()) > 10:
                        return sanitize_ai_output(res.text)
                except Exception:
                    continue
        except Exception:
            continue

    return "Service is momentarily busy. Please try asking your question again."

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
            print(f"[Groq JSON Notice]: {e}")

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
# LIGHTWEIGHT NATIVE PDF & WORD COMPILERS
# -------------------------------------------------------------
def compile_pdf_document(title: str, content: str, output_path: str):
    lines = []
    lines.append(f"BT /F1 16 Tf 50 750 Td ({title[:55]}) Tj ET")
    lines.append(f"BT /F1 9 Tf 50 735 Td (Omni Dossier - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}) Tj ET")
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
  <w:body>{body_content}</w:body>
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
    query_type: str = Form(...),
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
                f"- **Journey Date & Class**\n"
                f"- **Boarding & Destination Station**\n"
                f"- **Booking Status vs Current Status** (e.g., CNF, RAC, WL)\n"
                f"- **Chart Status**\n"
                f"Format cleanly in Grok style: bold labels only, clean Markdown tables, and no fluff."
            )
            ans = ask_hybrid_text(f"Check status for PNR: {val}", sys_prompt)
        elif query_type == "live_train":
            sys_prompt = (
                f"You are the Indian Railways National Train Enquiry System (NTES) officer. "
                f"Provide the running status for Train: {val} in {target_language}.\n"
                f"Provide:\n"
                f"- **Current Location & Delay**\n"
                f"- **Last Departed Station**\n"
                f"- **Next Halt & Expected Arrival**\n"
                f"- **Platform Number**\n"
                f"- A clean **Markdown Table** of upcoming halts.\n"
                f"Format cleanly in Grok style."
            )
            ans = ask_hybrid_text(f"Live running status for Train: {val}", sys_prompt)
        else:
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
# 3. INTERACTIVE CHAT (AI STUDIO & FOLLOW-UPS)
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

        visual_triggers = ["generate image", "create image", "picture of", "photo of", "logo", "3d logo", "render", "illustration", "draw"]
        if any(t in lower_q for t in visual_triggers):
            clean_subject = clean_q
            for tr in ["generate a 3d logo of", "generate 3d logo for", "generate image of", "create image of", "generate image", "create image"]:
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

        doc_awareness = f"\n[DOCUMENT CONTEXT]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = (
            f"You are Omni AI Companion, an authentic forensic auditor and knowledgeable guide. "
            f"Answer the user's specific inquiry directly in {target_language}. "
            f"Keep bold headlines only, regular text normal weight, and clean Markdown tables for figures or dates.{doc_awareness}"
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
        sys_prompt = f"Translate the report accurately into {target_language}. Maintain all bold titles, markdown tables, and red alert tags."
        translated = ask_hybrid_text(report_text, sys_prompt)
        return {"status": "success", "translated_report": translated}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 5. TOURISTOS DESTINATION EXPLORER
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
    elif any(x in lower_loc for x in ["united states", "usa", "us", "new york", "california"]):
        curr_sym = "$"
        nat_police, nat_hosp, nat_fire, nat_pharm = "911", "911", "911", "311 / 1-800-222-1222"
    elif any(x in lower_loc for x in ["france", "italy", "germany", "spain", "europe"]):
        curr_sym = "€"
        nat_police, nat_hosp, nat_fire, nat_pharm = "112", "112", "112", "112 / 24/7 Chemist"
    else:
        curr_sym = "₹"
        nat_police, nat_hosp, nat_fire, nat_pharm = "112 / 100", "108 / 102", "101", "1800-200-1234"

    system_prompt = (
        f"You are the senior tourism and municipal officer for '{loc_clean}'.\n"
        f"Party: {adults} Adults, {children} Kids. Diet: '{dietary_preference}'. Language: {target_language}.\n"
        f"Provide real landmarks, hotels with rates in {curr_sym}, and emergency contacts in strict JSON format."
    )

    user_query = f"Scan database for {loc_clean}. Provide 8-10 real landmarks, 6 real hotels, and nearest emergency facilities."
    extracted_data = ask_hybrid_json(user_query, system_prompt)

    if extracted_data and "spots" in extracted_data and len(extracted_data["spots"]) > 0:
        spots = extracted_data["spots"]
        for sp in spots:
            t_title = sp.get("title", city)
            seed = abs(hash(t_title + city)) % 999999
            enc_t = urllib.parse.quote(f"Scenic daylight photography of {t_title} {city} {country}, photorealistic, 8k")
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
            "destination_summary": f"{city} ({country}) travel guide and directory.",
            "spots": [
                {
                    "page": 1,
                    "title": f"Historic Center of {city}",
                    "category": "Cultural Heritage",
                    "rating": "⭐ 4.9",
                    "dist": "Central District",
                    "description": f"The architectural and cultural heart of {city}.",
                    "history": f"Recorded extensively in the historical annals of {country}.",
                    "sightseeing_rules": "Respect local cultural etiquette.",
                    "culinary": f"Traditional cuisine matching {dietary_preference}.",
                    "transit": f"Accessible via {city} public transit and taxis.",
                    "best_time_and_weather": "Spring and Autumn months provide optimal weather.",
                    "shopping": f"Central bazaars and shopping precincts in {city}.",
                    "speciality": f"Historic landmarks and cultural identity in {country}.",
                    "images": [f"https://image.pollinations.ai/prompt/Scenic%20daylight%20photography%20of%20historic%20{urllib.parse.quote(city)}%20{urllib.parse.quote(country)}?width=800&height=500&nologo=true&seed=101&model=flux"]
                }
            ],
            "hotels": [
                {
                    "hotel_id": f"HTL-{city[:3].upper()}-01",
                    "name": f"Grand Central Hotel {city}",
                    "party_suitability": f"{adults} Adults & {children} Kids",
                    "price_per_night": f"{curr_sym}120",
                    "rating": "⭐ 4.7 (500+ reviews)",
                    "location_address": f"City Center, {city}",
                    "amenities": ["Free Wi-Fi", "Breakfast Included", f"{dietary_preference} Options"]
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
    question: str = Form("Can you plan a 3-day itinerary?"),
    target_language: str = Form("English")
):
    try:
        clean_q = question.strip()
        loc_label = f"{city}, {country}".strip(", ")

        sys_prompt = (
            f"You are Omni Guide, a warm, polite local concierge for '{loc_label}'. "
            f"Travelers: {party_summary}. Diet: '{dietary_preference}'. Language: {target_language}. "
            f"Break itineraries clearly by day and time of day in clean Markdown."
        )

        response_text = ask_hybrid_text(clean_q, sys_prompt)

        is_itinerary_trigger = any(
            t in clean_q.lower()
            for t in ["itinerary", "plan", "download", "pdf", "word", "docx", "schedule", "guide", "days", "tour"]
        )

        pdf_url = ""
        docx_url = ""
        pdf_name = ""
        docx_name = ""

        if is_itinerary_trigger or len(response_text) > 400:
            doc_id = uuid.uuid4().hex[:6].upper()
            title_clean = f"Omni Guide - {city} Travel Dossier"
            base_url = str(request.base_url).rstrip("/")

            pdf_file_id = f"Omni_Itinerary_{city.replace(' ', '_')}_{doc_id}.pdf"
            pdf_path = os.path.join(DOWNLOADS_DIR, pdf_file_id)
            compile_pdf_document(title_clean, response_text, pdf_path)
            pdf_url = f"{base_url}/downloads/{pdf_file_id}"
            pdf_name = pdf_file_id

            docx_file_id = f"Omni_Itinerary_{city.replace(' ', '_')}_{doc_id}.docx"
            docx_path = os.path.join(DOWNLOADS_DIR, docx_file_id)
            compile_docx_document(title_clean, response_text, docx_path)
            docx_url = f"{base_url}/downloads/{docx_file_id}"
            docx_name = docx_file_id

        return {
            "status": "success",
            "answer": response_text,
            "has_document": bool(pdf_url),
            "pdf_url": pdf_url,
            "pdf_name": pdf_name,
            "docx_url": docx_url,
            "docx_name": docx_name,
            "destination": city
        }
    except Exception as e:
        return {"status": "error", "answer": f"I am happy to guide you through {city}! Could you please repeat that question?", "has_document": False}

# -------------------------------------------------------------
# 7. EXPORTERS & STUDIO
# -------------------------------------------------------------
@app.post("/api/v1/export-pdf")
async def export_pdf(request: Request, content: str = Form(...), title: str = Form("Travel Dossier")):
    try:
        file_id = f"Omni_{uuid.uuid4().hex[:6]}.pdf"
        out_path = os.path.join(DOWNLOADS_DIR, file_id)
        compile_pdf_document(title, content, out_path)
        base_url = str(request.base_url).rstrip("/")
        return {"status": "success", "download_url": f"{base_url}/downloads/{file_id}", "file_name": file_id, "file_type": "PDF"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/export-docx")
async def export_docx(request: Request, content: str = Form(...), title: str = Form("Travel Dossier")):
    try:
        file_id = f"Omni_{uuid.uuid4().hex[:6]}.docx"
        out_path = os.path.join(DOWNLOADS_DIR, file_id)
        compile_docx_document(title, content, out_path)
        base_url = str(request.base_url).rstrip("/")
        return {"status": "success", "download_url": f"{base_url}/downloads/{file_id}", "file_name": file_id, "file_type": "DOCX"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/resize-image")
async def resize_image(
    request: Request,
    mode: str = Form("size"),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes))
        new_w, new_h = (width or 1080), (height or 1080)

        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        out_name = f"Resized_{new_w}x{new_h}_{uuid.uuid4().hex[:4]}.jpg"
        out_path = os.path.join(DOWNLOADS_DIR, out_name)
        resized.save(out_path, "JPEG", quality=92)
        base_url = str(request.base_url).rstrip("/")
        return {"status": "success", "download_url": f"{base_url}/downloads/{out_name}", "dimensions": f"{new_w} x {new_h} px"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 8. SERVER HEALTH
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni TouristOS & Transit Cloud",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }