import os
import io
import gc
import json
import re
import time
import uuid
import base64
import zipfile
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import xml.etree.ElementTree as ET

import httpx
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Query, WebSocket, WebSocketDisconnect, HTTPException
from places import router as places_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from groq import Groq
from supabase import create_client, Client

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

app = FastAPI(
    title="Omni TouristOS & Unified Intelligence Cloud",
    description="Universal Travel AI, Street Lens Vision, Dual Voice, Bargain Pal, Universal Document Auditor, Transit Cloud & Community Intelligence",
    version="93.0.0"
)

# Register the places router on the active app instance
app.include_router(places_router)

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
# 0. SUPABASE CLIENT INITIALIZATION
# -------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip().strip('"').strip("'")
supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

# -------------------------------------------------------------
# 1. LIVE BULLION BENCHMARK ENGINE
# -------------------------------------------------------------
_bullion_cache = {
    "timestamp": 0,
    "data": None
}

def clean_rate_str(val: str) -> float:
    cleaned = re.sub(r'\(.*?\)', '', val).replace('₹', '').replace(',', '').strip()
    return float(cleaned)

def fetch_domestic_bullion_mumbai():
    current_time = time.time()
    if _bullion_cache["data"] and (current_time - _bullion_cache["timestamp"] < 600):
        return _bullion_cache["data"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    gold_24k = 15430.0
    gold_22k = 14144.0
    silver_1g = 238.0

    if BeautifulSoup is not None:
        try:
            url = "https://www.goodreturns.in/gold-rates/mumbai.html"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = [td.get_text(strip=True) for td in row.find_all("td")]
                        if len(cols) >= 3 and cols[0] == "1":
                            g24_val = clean_rate_str(cols[1])
                            g22_val = clean_rate_str(cols[2])
                            if 10000 < g24_val < 30000:
                                gold_24k = g24_val
                                gold_22k = g22_val
                                break
        except Exception as e:
            print(f"[Bullion Scrape Notice]: {e}")

    result = {
        "status": "success",
        "benchmark": "IBJA / Mumbai Domestic Spot",
        "gold_24k_per_g": round(gold_24k, 2),
        "gold_22k_per_g": round(gold_22k, 2),
        "silver_per_g": round(silver_1g, 2),
        "gold_24k_10g": round(gold_24k * 10, 0),
        "gold_22k_10g": round(gold_22k * 10, 0),
        "silver_per_kg": round(silver_1g * 1000, 0),
        "unit": "INR",
        "updated_at": time.strftime("%d %b %Y, %H:%M IST")
    }

    _bullion_cache["timestamp"] = current_time
    _bullion_cache["data"] = result
    return result

@app.get("/api/v1/bullion-rates")
def get_bullion_rates(city: str = Query("mumbai")):
    return fetch_domestic_bullion_mumbai()

# -------------------------------------------------------------
# 2. CREDENTIAL MANAGEMENT & GROQ CLIENT
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    raw = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
    return Groq(api_key=raw) if raw else None

def sanitize_ai_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

ACTIVE_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

# -------------------------------------------------------------
# 3. FAST TEXT ENGINE VIA ACTIVE GROQ MODELS
# -------------------------------------------------------------
async def ask_fast_text(prompt: str, system_prompt: str) -> str:
    client = get_groq_client()
    if not client:
        raise HTTPException(status_code=500, detail="Groq API key not configured on backend.")
    
    for model_id in ACTIVE_TEXT_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048,
                timeout=60
            )
            raw = completion.choices[0].message.content
            if raw and len(raw.strip()) > 0:
                return sanitize_ai_output(raw)
        except Exception as e:
            print(f"[Groq Text Notice with {model_id}]: {e}")
            continue

    raise HTTPException(status_code=500, detail="Groq API request failed across all active models.")

# -------------------------------------------------------------
# 4. FAST JSON ENGINE VIA GROQ
# -------------------------------------------------------------
async def ask_fast_json(prompt: str, system_prompt: str) -> Optional[dict]:
    client = get_groq_client()
    if not client:
        return None
    
    for model_id in ACTIVE_TEXT_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
                timeout=45
            )
            raw = completion.choices[0].message.content
            if raw:
                return json.loads(sanitize_ai_output(raw))
        except Exception as e:
            print(f"[Groq JSON Notice with {model_id}]: {e}")
            continue
    return None

# -------------------------------------------------------------
# 5. PAPER PILOT PERSISTENT CHAT & DIRECT QUERY HANDLER
# -------------------------------------------------------------
@app.post("/api/v1/ask-question")
async def ask_question(request: Request):
    question = ""
    target_language = "English"
    active_document_context = ""

    content_type = request.headers.get("content-type", "").lower()
    try:
        if "application/json" in content_type:
            body = await request.json()
            question = body.get("question", "")
            target_language = body.get("target_language", "English")
            active_document_context = body.get("active_document_context", "")
        else:
            form = await request.form()
            question = form.get("question", "")
            target_language = form.get("target_language", "English")
            active_document_context = form.get("active_document_context", "")
    except Exception:
        pass

    clean_q = str(question).strip()
    if not clean_q:
        return {"status": "error", "answer": "How can I assist you today?"}

    if clean_q.lower() in ["hi", "hello", "hey", "greetings"]:
        return {"status": "success", "answer": "Hello!", "reply": "Hello!"}

    lang_lower = target_language.lower()
    if "marathi" in lang_lower or "मराठी" in lang_lower:
        lang_instruction = "Answer strictly in natural, professional Marathi (मराठी - Devanagari script)."
    elif "hindi" in lang_lower or "हिंदी" in lang_lower:
        lang_instruction = "Answer strictly in natural, professional Hindi (हिंदी - Devanagari script)."
    else:
        lang_instruction = f"Answer clearly and concisely in {target_language}."

    has_doc = bool(active_document_context and len(active_document_context.strip()) > 10)

    if has_doc:
        sys_prompt = f"""
You are Paper Pilot, an AI Document Auditor.
{lang_instruction}

UPLOADED DOCUMENT CONTEXT:
{active_document_context[:95000]}

RULES:
1. Answer the user's specific query strictly using the document context above.
2. Be direct, accurate, and concise. Omit unnecessary preamble.
"""
    else:
        sys_prompt = f"""
You are Paper Pilot, an AI assistant.
{lang_instruction}

RULES:
1. Answer the user's query directly and accurately.
"""

    ans = await ask_fast_text(clean_q, sys_prompt)
    return {"status": "success", "answer": ans, "reply": ans}

# -------------------------------------------------------------
# 6. GENERAL CONVERSATION ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/chat")
async def general_chat(request: Request):
    try:
        body = await request.json()
        message = body.get("message") or body.get("question") or ""
        target_language = body.get("target_language", "English")
        context = body.get("context", "")
        
        if message.strip().lower() in ["hi", "hello", "hey"]:
            return {"status": "success", "answer": "Hello!", "reply": "Hello!"}

        sys_prompt = f"You are Paper Pilot Assistant. Answer directly and concisely in {target_language}.\nContext: {context}"
        ans = await ask_fast_text(message, sys_prompt)
        return {"status": "success", "answer": ans, "reply": ans}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 7. MULTI-FORMAT DOCUMENT CONVERTER ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/convert-file")
async def convert_file(
    file: UploadFile = File(...),
    target_format: str = Form(...)
):
    try:
        file_bytes = await file.read()
        filename = file.filename or "document.bin"
        target_fmt = target_format.upper().strip()

        converted_filename = f"Converted_{uuid.uuid4().hex[:8]}.{target_fmt.lower()}"
        save_path = os.path.join(DOWNLOADS_DIR, converted_filename)

        if target_fmt in ["TXT", "TEXT", "MD"]:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text_content)
        elif target_fmt == "HTML":
            text_content = file_bytes.decode("utf-8", errors="ignore")
            html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body><pre>{text_content}</pre></body></html>"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif target_fmt in ["JPG", "JPEG", "PNG", "WEBP", "BMP"]:
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB" and target_fmt in ["JPG", "JPEG"]:
                pil_img = pil_img.convert("RGB")
            pil_img.save(save_path, format="JPEG" if target_fmt in ["JPG", "JPEG"] else target_fmt)
        else:
            with open(save_path, "wb") as f:
                f.write(file_bytes)

        host_url = str(file.headers.get("host") or "omni-backend-pk28.onrender.com")
        scheme = "https" if "onrender.com" in host_url else "http"
        download_url = f"{scheme}://{host_url}/downloads/{converted_filename}"

        return {
            "status": "success",
            "download_url": download_url,
            "filename": converted_filename,
            "format": target_fmt
        }
    except Exception as e:
        return {"status": "error", "message": f"Conversion error: {str(e)}"}

# -------------------------------------------------------------
# 8. DOSSIER EXPORT ENGINES (PDF & WORD)
# -------------------------------------------------------------
@app.post("/api/v1/export-pdf")
async def export_pdf(title: str = Form(...), content: str = Form(...)):
    try:
        pdf_filename = f"Vault_Dossier_{uuid.uuid4().hex[:8]}.pdf"
        save_path = os.path.join(DOWNLOADS_DIR, pdf_filename)
        html_source = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title></head><body><h1>{title}</h1><pre>{content}</pre></body></html>"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_source)
        return {"status": "success", "download_url": f"/downloads/{pdf_filename}", "file_name": pdf_filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/export-docx")
async def export_docx(title: str = Form(...), content: str = Form(...)):
    try:
        doc_filename = f"Vault_Dossier_{uuid.uuid4().hex[:8]}.doc"
        save_path = os.path.join(DOWNLOADS_DIR, doc_filename)
        html_source = f"""\uFEFF<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title></head><body><h1>{title}</h1><pre>{content}</pre></body></html>"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_source)
        return {"status": "success", "download_url": f"/downloads/{doc_filename}", "file_name": doc_filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 9. NATIVE IN-APP FLIGHT SEARCH & COMPARISON ENGINE
# -------------------------------------------------------------
@app.post("/api/v1/search-flights")
async def search_flights(request: Request):
    try:
        body = await request.json()
        origin = body.get("origin", "BOM").upper()
        destination = body.get("destination", "TYO").upper()
        depart_date = body.get("departure_date") or body.get("depart_date") or "2026-09-21"
        return_date = body.get("return_date") or "2026-09-28"
        adults = int(body.get("adults", 1))
        cabin_class = body.get("cabin_class", "Economy")
        is_round_trip = body.get("is_round_trip", True)

        fare_benchmarks = {
            ("BOM", "TYO"): [
                {
                    "airline": "IndiGo & Partner Carrier",
                    "code": "6E-512 / NH-860",
                    "depart_time": "08:15",
                    "arrive_time": "19:40",
                    "duration": "8h 55m",
                    "stops": "1 Stop (BKK)",
                    "price_inr": 34200,
                    "badge": "Cheapest Fare",
                    "badge_color": "0xFF16A34A",
                    "perks": "7kg Cabin • Seat Selection Selectable"
                },
                {
                    "airline": "Air India (Tata Group)",
                    "code": "AI-306",
                    "depart_time": "20:00",
                    "arrive_time": "07:55",
                    "duration": "8h 25m",
                    "stops": "Non-stop Direct",
                    "price_inr": 44500,
                    "badge": "Fastest Direct",
                    "badge_color": "0xFFDC2626",
                    "perks": "23kg Check-in • Hot Gourmet Meals Included"
                }
            ],
            ("BOM", "SIN"): [
                {
                    "airline": "Air India Express",
                    "code": "IX-245",
                    "depart_time": "11:10",
                    "arrive_time": "19:15",
                    "duration": "5h 35m",
                    "stops": "Non-stop Direct",
                    "price_inr": 14200,
                    "badge": "Cheapest Fare",
                    "badge_color": "0xFF16A34A",
                    "perks": "7kg Cabin • Paid Add-ons Available"
                }
            ]
        }

        pair = (origin, destination)
        results = fare_benchmarks.get(pair)

        if not results:
            results = [
                {
                    "airline": "Regional Value Air",
                    "code": "VA-102",
                    "depart_time": "07:30",
                    "arrive_time": "16:45",
                    "duration": "6h 45m",
                    "stops": "1 Stop",
                    "price_inr": 21500,
                    "badge": "Cheapest Fare",
                    "badge_color": "0xFF16A34A",
                    "perks": "7kg Cabin Baggage Included"
                },
                {
                    "airline": "National Full-Service Carrier",
                    "code": "FC-404",
                    "depart_time": "14:15",
                    "arrive_time": "21:30",
                    "duration": "5h 15m",
                    "stops": "Non-stop Direct",
                    "price_inr": 31200,
                    "badge": "Fastest Direct",
                    "badge_color": "0xFFDC2626",
                    "perks": "20kg Baggage + Hot Meal Included"
                }
            ]

        mult = 1.0
        if cabin_class == "Premium Economy":
            mult = 1.55
        elif cabin_class == "Business":
            mult = 2.85
        elif cabin_class == "First Class":
            mult = 4.60

        adjusted_results = []
        for f in results:
            item = dict(f)
            unit_price = item["price_inr"] * (1.85 if is_round_trip else 1.0) * mult
            item["unit_price_inr"] = int(unit_price)
            item["total_price_inr"] = int(unit_price * adults)
            item["price"] = int(unit_price)
            item["currency"] = "INR"
            item["symbol"] = "₹"
            adjusted_results.append(item)

        return {
            "status": "success",
            "origin": origin,
            "destination": destination,
            "depart_date": depart_date,
            "return_date": return_date if is_round_trip else None,
            "adults": adults,
            "cabin_class": cabin_class,
            "currency": "INR",
            "flights": adjusted_results
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "flights": []}

# -------------------------------------------------------------
# 10. CONCIERGE MULTI-TURN TEXT HELPER
# -------------------------------------------------------------
async def ask_concierge_text(prompt: str, system_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": "user" if role == "user" else "assistant", "content": content})
    messages.append({"role": "user", "content": prompt})

    client = get_groq_client()
    if client:
        for model_id in ACTIVE_TEXT_MODELS:
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=60
                )
                raw = completion.choices[0].message.content
                if raw and len(raw.strip()) > 0:
                    return sanitize_ai_output(raw)
            except Exception as e:
                print(f"[Groq Concierge Notice with {model_id}]: {e}")
                continue
    return f"Advisory for {prompt}"

# -------------------------------------------------------------
# 11. STREET VOICE TRANSLATION
# -------------------------------------------------------------
@app.post("/api/v1/street-voice-translate")
async def street_voice_translate(
    text: str = Form(...),
    source_language: str = Form("English"),
    target_language: str = Form("Marathi")
):
    clean_text = text.strip()
    if not clean_text:
        return {"status": "error", "translation": ""}

    sys_prompt = (
        f"You are a real-time conversational voice interpreter. "
        f"Translate spoken speech directly from {source_language} into natural, colloquial {target_language}. "
        f"CRITICAL: Output ONLY the direct translated phrase in the target script. "
        f"Do NOT include explanations, romanizations, quotes, or conversational preamble."
    )

    client = get_groq_client()
    if client:
        for model_id in ACTIVE_TEXT_MODELS:
            try:
                comp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": clean_text}
                    ],
                    temperature=0.1,
                    max_tokens=256,
                    timeout=15
                )
                raw_ans = comp.choices[0].message.content.strip().strip('"')
                if raw_ans:
                    return {"status": "success", "translation": sanitize_ai_output(raw_ans)}
            except Exception as e:
                print(f"[Street Voice Notice with {model_id}]: {e}")
                continue

    return {"status": "error", "translation": "Translation failed. Check connection."}

# -------------------------------------------------------------
# 12. STREET LENS
# -------------------------------------------------------------
@app.post("/api/v1/street-lens")
async def street_lens(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        img_bytes = prepare_image_bytes(file_bytes)
        if not img_bytes:
            return {"status": "error", "message": "Could not decode photo."}
        analysis = await ask_fast_text("Interpret this sign and its practical meaning for a traveler.", "You are a visual assistant.")
        return {"status": "success", "interpretation": analysis}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 13. WIKIPEDIA PHOTO RESOLVER
# -------------------------------------------------------------
def get_verified_landmark_photo(landmark_name: str, city: str) -> str:
    headers = {
        "User-Agent": "OmniTouristOS/4.0 requests/2.31"
    }
    clean_name = re.sub(r'\(.*?\)', '', landmark_name).strip()
    search_candidates = [clean_name, f"{clean_name}, {city}", landmark_name]

    for cand in search_candidates:
        if not cand or len(cand) < 2:
            continue
        try:
            slug = cand.strip().replace(" ", "_")
            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(slug)}"
            r_sum = requests.get(sum_url, headers=headers, timeout=4.0)
            if r_sum.status_code == 200:
                p_data = r_sum.json()
                if "originalimage" in p_data and "source" in p_data["originalimage"]:
                    return p_data["originalimage"]["source"]
                if "thumbnail" in p_data and "source" in p_data["thumbnail"]:
                    return re.sub(r'/\d+px-', '/1200px-', p_data["thumbnail"]["source"])
        except Exception:
            continue
    return ""

# -------------------------------------------------------------
# 14. DOCUMENT PARSERS & BINARY FORENSIC INSPECTOR
# -------------------------------------------------------------
def inspect_binary_stream(file_bytes: bytes, max_len: int = 2048) -> str:
    try:
        preview_len = min(len(file_bytes), max_len)
        printable_strings = re.findall(rb'[A-Za-z0-9/\-_:., ]{4,}', file_bytes[:preview_len * 4])
        decoded_strings = [s.decode('ascii', errors='ignore') for s in printable_strings[:100]]
        return "EXTRACTED STREAM DATA:\n" + "\n".join(f"• {s}" for s in decoded_strings)
    except Exception as e:
        return f"Stream error: {e}"

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
    except Exception:
        return ""

def extract_text_from_xlsx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            shared_strings = []
            if "xl/sharedStrings.xml" in zf.namelist():
                tree = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in tree.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    t_nodes = [node.text for node in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if node.text]
                    shared_strings.append("".join(t_nodes))
            sheets = sorted([n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")])
            table_output = []
            for s_name in sheets:
                tree = ET.fromstring(zf.read(s_name))
                for row in tree.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                    row_vals = []
                    for c in row.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                        v_node = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                        if v_node is not None and v_node.text:
                            val = v_node.text
                            if c.attrib.get('t') == 's' and val.isdigit() and int(val) < len(shared_strings):
                                val = shared_strings[int(val)]
                            row_vals.append(val)
                    if row_vals:
                        table_output.append(" | ".join(row_vals))
            return "\n".join(table_output)
    except Exception:
        return ""

def extract_massive_pdf_text(file_bytes: bytes, max_pages: int = 500) -> Tuple[str, int]:
    if PdfReader is None:
        return inspect_binary_stream(file_bytes), 1
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
        if not full_extracted.strip():
            full_extracted = inspect_binary_stream(file_bytes)
        return full_extracted.strip(), total_pages
    except Exception:
        return inspect_binary_stream(file_bytes), 1

def prepare_image_bytes(file_bytes: bytes) -> Optional[bytes]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        if max(pil_img.size) > 1600:
            pil_img.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
        out_buf = io.BytesIO()
        pil_img.save(out_buf, format="JPEG", quality=92)
        return out_buf.getvalue()
    except Exception:
        return None

# -------------------------------------------------------------
# 15. PAPER PILOT UNIVERSAL DOCUMENT AUDITOR
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        filename = (file.filename or "uploaded_document.pdf").lower()

        extracted_text = ""
        total_pages_detected = 1

        if filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            extracted_text = extract_text_from_xlsx(file_bytes)
        elif any(filename.endswith(ext) for ext in [".csv", ".txt", ".json", ".md", ".xml", ".rtf"]):
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                pass
        elif filename.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
            extracted_text, total_pages_detected = extract_massive_pdf_text(file_bytes, max_pages=500)
        else:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                extracted_text = inspect_binary_stream(file_bytes)

        if not extracted_text or len(extracted_text.strip()) < 5:
            extracted_text = inspect_binary_stream(file_bytes)

        lang_lower = target_language.lower()
        if "marathi" in lang_lower or "मराठी" in lang_lower:
            lang_instruction = "CRITICAL: Produce the entire audit summary STRICTLY IN MARATHI (मराठी - Devanagari script)."
        elif "hindi" in lang_lower or "हिंदी" in lang_lower:
            lang_instruction = "CRITICAL: Produce the entire audit summary STRICTLY IN HINDI (हिंदी - Devanagari script)."
        else:
            lang_instruction = f"Output the entire analysis clearly in {target_language}."

        audit_prompt = (
            f"You are Paper Pilot, an elite Document Analyst and Intelligence Auditor.\n"
            f"{lang_instruction}\n\n"
            f"GUIDELINES:\n"
            f"1. Provide a comprehensive, rigorous forensic audit of the document contents below.\n"
            f"2. Structure into clear sections: • Executive Summary • Key Breakdown / Clauses • Financial / Numerical Data • Actionable Recommendations.\n"
            f"3. At the very end, output: EXPLORE_SUGGESTIONS: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
        )

        analysis_raw = await ask_fast_text(
            f"DOCUMENT FILE: {filename} (Pages: {total_pages_detected})\n\nCONTENT:\n{extracted_text[:95000]}",
            audit_prompt
        )

        del file_bytes
        gc.collect()

        suggestions = [
            "What are the primary financial details here?",
            "Are there hidden liabilities or terms?",
            "How do I verify this record?"
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

        return {
            "status": "success",
            "data": {
                "document_title": f"Document Audit ({filename})",
                "actionable_advisory": clean_text,
                "detected_destination": None,
                "suggestions": suggestions
            },
            "raw_text": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit error: {str(e)}")

# -------------------------------------------------------------
# 16. REPORT TRANSLATOR
# -------------------------------------------------------------
@app.post("/api/v1/translate-report")
async def translate_report(report_text: str = Form(...), target_language: str = Form("Marathi")):
    try:
        lang_lower = target_language.lower()
        if "marathi" in lang_lower or "मराठी" in lang_lower:
            sys_prompt = "Translate this report completely into pure Marathi (Devanagari script). Keep all formatting intact."
        else:
            sys_prompt = f"Translate the report into {target_language}. Retain formatting."
        translated = await ask_fast_text(report_text, sys_prompt)
        return {"status": "success", "translated_report": translated}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 17. CONCIERGE CHAT & LIVE RADAR PIPELINE
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(request: Request):
    city = "Vasai-Virar"
    country = "India"
    question = ""
    target_language = "English"
    chat_history: List[Dict[str, str]] = []

    content_type = request.headers.get("content-type", "").lower()
    try:
        if "application/json" in content_type:
            body = await request.json()
            city = body.get("city", city)
            country = body.get("country", country)
            question = body.get("question", "")
            target_language = body.get("target_language", target_language)
            chat_history = body.get("chat_history", [])
        else:
            form = await request.form()
            city = form.get("city", city)
            country = form.get("country", country)
            question = form.get("question", "")
            target_language = form.get("target_language", target_language)
    except Exception:
        pass

    clean_q = str(question).strip()
    ans = await ask_concierge_text(clean_q, f"You are Omni Guide Assistant in {city}.", chat_history)

    return {
        "status": "success",
        "answer": ans,
        "venues": [],
        "has_document": False,
        "pdf_name": f"{city}_Itinerary.pdf",
        "docx_name": f"{city}_Itinerary.docx",
    }

# -------------------------------------------------------------
# 18. COMMUNITY INTELLIGENCE, MODERATION & 1-ON-1 SUITE
# -------------------------------------------------------------
def verify_admin_privileges(user_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not connected.")
    try:
        res = supabase.table("users").select("role, is_banned").eq("id", user_id).single().execute()
        if not res.data or res.data.get("role") not in ["admin", "supervisor"] or res.data.get("is_banned"):
            raise HTTPException(status_code=403, detail="Unauthorized: Administrator or Supervisor role required.")
    except Exception:
        raise HTTPException(status_code=403, detail="Unauthorized: Verification failed.")

def evaluate_supervisor_bot_flag(text: str) -> Optional[str]:
    lower_t = text.lower()
    if re.search(r"(http[s]?://|t\.me/|wa\.me/|bit\.ly)", lower_t):
        return "Unauthorized external link / solicitation"
    if any(k in lower_t for k in ["cheap rate gold", "crypto", "unregulated deal", "wire cash", "paytm scam", "telegram me"]):
        return "High-risk financial solicitation or manipulation"
    if any(k in lower_t for k in ["abuse", "scammer", "fraud", "kill", "threat"]):
        return "Hostile conduct or prohibited language"
    return None

@app.get("/api/v1/community/feed")
async def get_community_feed(community_id: str = Query("vasai-virar")):
    if not supabase:
        return {"status": "success", "gems": [], "pulse": []}
    try:
        gems_res = supabase.table("gems").select("*").eq("community_id", community_id).order("created_at", desc=True).limit(20).execute()
        pulse_res = supabase.table("pulse_updates").select("*").eq("community_id", community_id).order("created_at", desc=True).limit(10).execute()
        return {"status": "success", "gems": gems_res.data, "pulse": pulse_res.data}
    except Exception as e:
        return {"status": "error", "message": str(e), "gems": [], "pulse": []}

@app.post("/api/v1/gems/create")
async def create_gem(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        creator_id = body.get("creator_id")
        
        if creator_id:
            chk = supabase.table("users").select("is_banned").eq("id", creator_id).single().execute()
            if chk.data and chk.data.get("is_banned"):
                raise HTTPException(status_code=403, detail="Banned accounts cannot add gems.")

        response = supabase.table("gems").insert({
            "community_id": body.get("community_id", "vasai-virar"),
            "creator_id": creator_id,
            "title": body.get("title"),
            "category": body.get("category", "Markets"),
            "description": body.get("description"),
            "location_string": body.get("location_string"),
            "status": "New"
        }).execute()
        return {"status": "success", "gem": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/community/messages")
async def get_community_messages(community_id: str = Query("vasai-virar")):
    if not supabase:
        return {"status": "success", "messages": []}
    try:
        res = supabase.table("messages") \
            .select("*, users(display_name, avatar_url, role)") \
            .eq("community_id", community_id) \
            .neq("is_deleted", True) \
            .order("created_at", desc=False) \
            .limit(50) \
            .execute()
        return {"status": "success", "messages": res.data}
    except Exception as e:
        return {"status": "error", "message": str(e), "messages": []}

@app.post("/api/v1/community/report")
async def report_community_item(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        reporter_id = body.get("reporter_id")
        offender_id = body.get("offender_id")
        message_id = body.get("message_id")
        community_id = body.get("community_id", "vasai-virar")
        reason = body.get("reason", "Spam or Scam")
        details = body.get("details", "")

        report_res = supabase.table("community_reports").insert({
            "reporter_id": reporter_id,
            "offender_id": offender_id,
            "message_id": message_id,
            "community_id": community_id,
            "reason": reason,
            "details": details,
            "status": "pending"
        }).execute()

        if message_id:
            try:
                m = supabase.table("messages").select("flag_count").eq("id", message_id).single().execute()
                curr_count = (m.data.get("flag_count") or 0) + 1 if m.data else 1
                supabase.table("messages").update({"flag_count": curr_count}).eq("id", message_id).execute()
            except Exception:
                pass

        return {"status": "success", "message": "Report logged for admin review."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/admin/reports")
async def get_admin_reports(admin_id: str = Query(...)):
    verify_admin_privileges(admin_id)
    try:
        reports = supabase.table("community_reports") \
            .select("*, reporter:reporter_id(display_name), offender:offender_id(display_name, role, is_banned), message:message_id(text, created_at)") \
            .eq("status", "pending") \
            .order("created_at", desc=True) \
            .execute()
        return {"status": "success", "reports": reports.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/moderate")
async def moderate_community_incident(request: Request):
    try:
        body = await request.json()
        admin_id = body.get("admin_id")
        report_id = body.get("report_id")
        action = body.get("action")
        offender_id = body.get("offender_id")
        message_id = body.get("message_id")
        community_id = body.get("community_id", "vasai-virar")

        verify_admin_privileges(admin_id)

        if action == "ban" and offender_id:
            supabase.table("users").update({"is_banned": True}).eq("id", offender_id).execute()
            await manager.broadcast(community_id, {
                "type": "admin_disconnect_user",
                "user_id": offender_id,
                "reason": "Account banned for community violation."
            })

        elif action == "mute_24h" and offender_id:
            mute_until = (datetime.utcnow() + timedelta(hours=24)).isoformat()
            supabase.table("users").update({"muted_until": mute_until}).eq("id", offender_id).execute()

        if (action in ["purge_message", "ban"]) and message_id:
            supabase.table("messages").update({"is_deleted": True}).eq("id", message_id).execute()
            await manager.broadcast(community_id, {
                "type": "delete_message",
                "message_id": message_id
            })

        if report_id:
            supabase.table("community_reports").update({
                "status": "resolved" if action != "dismiss" else "dismissed",
                "action_taken": action,
                "reviewed_by": admin_id,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", report_id).execute()

        supabase.table("admin_audit_logs").insert({
            "admin_id": admin_id,
            "target_user_id": offender_id,
            "action": action,
            "reason": body.get("notes", "Administrator enforcement action")
        }).execute()

        return {"status": "success", "action_applied": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/direct/invite")
async def invite_direct_chat(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        requester_id = body.get("requester_id")
        recipient_id = body.get("recipient_id")
        initial_note = body.get("initial_note", "Hi, I would like to connect about local recommendations.")

        existing = supabase.table("direct_conversations") \
            .select("*") \
            .or_(f"and(user_a.eq.{requester_id},user_b.eq.{recipient_id}),and(user_a.eq.{recipient_id},user_b.eq.{requester_id})") \
            .execute()

        if existing.data and len(existing.data) > 0:
            return {"status": "success", "conversation": existing.data[0]}

        new_conv = supabase.table("direct_conversations").insert({
            "user_a": requester_id,
            "user_b": recipient_id,
            "status": "pending",
            "initial_note": initial_note
        }).execute()

        return {"status": "success", "conversation": new_conv.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/direct/respond")
async def respond_direct_chat(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id")
        action = body.get("action")
        status_val = "accepted" if action == "accept" else "rejected"

        upd = supabase.table("direct_conversations").update({
            "status": status_val,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()

        return {"status": "success", "status_val": status_val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/direct/delete")
async def delete_direct_conversation(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        conversation_id = body.get("conversation_id")
        supabase.table("direct_conversations").update({"status": "archived"}).eq("id", conversation_id).execute()
        return {"status": "success", "message": "Conversation purged successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------
# 19. INDIAN RAILWAYS TRANSIT & PNR ENGINE
# -------------------------------------------------------------
# Railway data layer is intentionally contained in this section only.
# The rest of main.py (Paper Pilot, Community, WebSockets, etc.) is untouched.
#
# IMPORTANT:
# - This implementation does NOT require a railway API key.
# - Timetable data is read from a public Mumbai suburban timetable reference
#   at request time and cached in memory.
# - It covers only Western, Central and Harbour lines, as requested.
# - Live delay/platform/door values are NEVER fabricated. They are returned
#   only when a real backend source supplies them.
# -------------------------------------------------------------

RAILWAY_REFERENCE_BASE = "https://www.mumbailifeline.com"
RAILWAY_REFERENCE_LABEL = "Mumbai Lifeline public timetable reference"
RAILWAY_TIMETABLE_VERSIONS = {
    "W": "May 2026",
    "C": "2024",
    "H": "May 2026",
}

# Short, stable station master used by the app.  The timetable source uses
# station slugs/names rather than the app's local codes, so we translate here.
RAILWAY_STATIONS: Dict[str, Dict[str, Any]] = {
    # ----------------------------- WESTERN -----------------------------
    "CCG": {"name": "Churchgate", "line": "W", "slug": "churchgate"},
    "MEL": {"name": "Marine Lines", "line": "W", "slug": "marine_lines"},
    "CYR": {"name": "Charni Road", "line": "W", "slug": "charni_road"},
    "GTR": {"name": "Grant Road", "line": "W", "slug": "grant_road"},
    "MMCT": {"name": "Mumbai Central", "line": "W", "slug": "mumbai_central"},
    "MX": {"name": "Mahalaxmi", "line": "W", "slug": "mahalaxmi"},
    "PL": {"name": "Lower Parel", "line": "W", "slug": "lower_parel"},
    "PBHD": {"name": "Prabhadevi", "line": "W", "slug": "prabhadevi"},
    "DDR": {"name": "Dadar", "line": "W", "slug": "dadar"},
    "MRU": {"name": "Matunga Road", "line": "W", "slug": "matunga_road"},
    "MM": {"name": "Mahim", "line": "W", "slug": "mahim"},
    "BA": {"name": "Bandra", "line": "W", "slug": "bandra"},
    "KHAR": {"name": "Khar Road", "line": "W", "slug": "khar_road"},
    "STC": {"name": "Santacruz", "line": "W", "slug": "santacruz"},
    "VLP": {"name": "Vile Parle", "line": "W", "slug": "vile_parle"},
    "ADH": {"name": "Andheri", "line": "W", "slug": "andheri"},
    "JOS": {"name": "Jogeshwari", "line": "W", "slug": "jogeshwari"},
    "RMAR": {"name": "Ram Mandir", "line": "W", "slug": "ram_mandir"},
    "GMN": {"name": "Goregaon", "line": "W", "slug": "goregaon"},
    "MDD": {"name": "Malad", "line": "W", "slug": "malad"},
    "KILE": {"name": "Kandivali", "line": "W", "slug": "kandivali"},
    "BVI": {"name": "Borivali", "line": "W", "slug": "borivali"},
    "DIC": {"name": "Dahisar", "line": "W", "slug": "dahisar"},
    "MIRA": {"name": "Mira Road", "line": "W", "slug": "mira_road"},
    "BYR": {"name": "Bhayandar", "line": "W", "slug": "bhayandar"},
    "NIG": {"name": "Naigaon", "line": "W", "slug": "naigaon"},
    "BSR": {"name": "Vasai Road", "line": "W", "slug": "vasai_road"},
    "NSP": {"name": "Nalasopara", "line": "W", "slug": "nalla_sopara"},
    "VR": {"name": "Virar", "line": "W", "slug": "virar"},
    "Kelve": {"name": "Kelva Road", "line": "W", "slug": "kelva_road"},
    "SAP": {"name": "Saphale", "line": "W", "slug": "saphale"},
    "VTN": {"name": "Vaitarna", "line": "W", "slug": "vaitarna"},
    "PAL": {"name": "Palghar", "line": "W", "slug": "palghar"},
    "BOIS": {"name": "Boisar", "line": "W", "slug": "boisar"},
    "VNG": {"name": "Vangaon", "line": "W", "slug": "vangaon"},
    "DRD": {"name": "Dahanu Road", "line": "W", "slug": "dahanu_road"},

    # ----------------------------- CENTRAL -----------------------------
    "CSMT": {"name": "Mumbai CST", "line": "C", "slug": "mumbai_cst"},
    "BY": {"name": "Byculla", "line": "C", "slug": "byculla"},
    "SDR": {"name": "Sandhurst Road", "line": "C", "slug": "sandhurst_road"},
    "CRR": {"name": "Currey Road", "line": "C", "slug": "currey_road"},
    "PAR": {"name": "Parel", "line": "C", "slug": "parel"},
    "CDR": {"name": "Dadar", "line": "C", "slug": "dadar"},
    "MAT": {"name": "Matunga", "line": "C", "slug": "matunga"},
    "SION": {"name": "Sion", "line": "C", "slug": "sion"},
    "CLA": {"name": "Kurla", "line": "C", "slug": "kurla"},
    "VID": {"name": "Vidyavihar", "line": "C", "slug": "vidyavihar"},
    "GC": {"name": "Ghatkopar", "line": "C", "slug": "ghatkopar"},
    "VSD": {"name": "Vikhroli", "line": "C", "slug": "vikhroli"},
    "KMR": {"name": "Kanjurmarg", "line": "C", "slug": "kanjurmarg"},
    "BND": {"name": "Bhandup", "line": "C", "slug": "bhandup"},
    "NAH": {"name": "Nahur", "line": "C", "slug": "nahur"},
    "MLND": {"name": "Mulund", "line": "C", "slug": "mulund"},
    "TNA": {"name": "Thane", "line": "C", "slug": "thane"},
    "KLV": {"name": "Kalva", "line": "C", "slug": "kalva"},
    "MMB": {"name": "Mumbra", "line": "C", "slug": "mumbra"},
    "DIVA": {"name": "Diva", "line": "C", "slug": "diva"},
    "KOP": {"name": "Kopar", "line": "C", "slug": "kopar"},
    "DOM": {"name": "Dombivli", "line": "C", "slug": "dombivli"},
    "THK": {"name": "Thakurli", "line": "C", "slug": "thakurli"},
    "KYN": {"name": "Kalyan", "line": "C", "slug": "kalyan"},
    "VTL": {"name": "Vitthalwadi", "line": "C", "slug": "vitthalwadi"},
    "ULH": {"name": "Ulhasnagar", "line": "C", "slug": "ulhasnagar"},
    "AMB": {"name": "Ambernath", "line": "C", "slug": "ambernath"},
    "BAD": {"name": "Badlapur", "line": "C", "slug": "badlapur"},
    "VGI": {"name": "Vangani", "line": "C", "slug": "vangani"},
    "SHE": {"name": "Shelu", "line": "C", "slug": "shelu"},
    "NER": {"name": "Neral", "line": "C", "slug": "neral"},
    "BHV": {"name": "Bhivpuri Road", "line": "C", "slug": "bhivpuri_road"},
    "KJT": {"name": "Karjat", "line": "C", "slug": "karjat"},
    "TIT": {"name": "Titwala", "line": "C", "slug": "titwala"},
    "KHD": {"name": "Khadavli", "line": "C", "slug": "khadavli"},
    "VSI": {"name": "Vasind", "line": "C", "slug": "vasind"},
    "ASN": {"name": "Asangaon", "line": "C", "slug": "asangaon"},
    "KAS": {"name": "Kasara", "line": "C", "slug": "kasara"},
    "KPH": {"name": "Khopoli", "line": "C", "slug": "khopoli"},

    # ----------------------------- HARBOUR -----------------------------
    "HCS": {"name": "Mumbai CST", "line": "H", "slug": "mumbai_cst"},
    "MSD": {"name": "Masjid", "line": "H", "slug": "masjid"},
    "SNRD": {"name": "Sandhurst Road", "line": "H", "slug": "sandhurst_road"},
    "DYR": {"name": "Dockyard Road", "line": "H", "slug": "dockyard_road"},
    "RAY": {"name": "Reay Road", "line": "H", "slug": "reay_road"},
    "CTN": {"name": "Cotton Green", "line": "H", "slug": "cotton_green"},
    "SEW": {"name": "Sewri", "line": "H", "slug": "sewri"},
    "VDR": {"name": "Vadala Road", "line": "H", "slug": "vadala_road"},
    "GTB": {"name": "GTB Nagar", "line": "H", "slug": "gtb_nagar"},
    "CHN": {"name": "Chunabhatti", "line": "H", "slug": "chunabhatti"},
    "KLS": {"name": "Kings Circle", "line": "H", "slug": "kings_circle"},
    "MAH": {"name": "Mahim", "line": "H", "slug": "mahim"},
    "BH": {"name": "Bandra", "line": "H", "slug": "bandra_hr"},
    "ADH_H": {"name": "Andheri", "line": "H", "slug": "andheri_hr"},
    "GMN_H": {"name": "Goregaon", "line": "H", "slug": "goregaon_hr"},
    "CLA_H": {"name": "Kurla", "line": "H", "slug": "kurla"},
    "CHMB": {"name": "Chembur", "line": "H", "slug": "chembur"},
    "GOV": {"name": "Govandi", "line": "H", "slug": "govandi"},
    "MNRD": {"name": "Mankhurd", "line": "H", "slug": "mankhurd"},
    "VSH": {"name": "Vashi", "line": "H", "slug": "vashi"},
    "SAN": {"name": "Sanpada", "line": "H", "slug": "sanpada"},
    "JUI": {"name": "Juinagar", "line": "H", "slug": "juinagar"},
    "NER_H": {"name": "Nerul", "line": "H", "slug": "nerul"},
    "SEW_H": {"name": "Seawoods", "line": "H", "slug": "seawoods"},
    "BEPR": {"name": "Belapur CBD", "line": "H", "slug": "belapur_cbd"},
    "KHAR_H": {"name": "Kharghar", "line": "H", "slug": "kharghar"},
    "MNS": {"name": "Mansarovar", "line": "H", "slug": "mansarovar"},
    "KLM": {"name": "Khandeshwar", "line": "H", "slug": "khandeshwar"},
    "PNVL": {"name": "Panvel", "line": "H", "slug": "panvel"},
}

# Some stations belong to more than one suburban line.  Keep these aliases
# explicit so the route planner can recognize interchanges.
RAILWAY_INTERCHANGES = {
    "DADAR": ["W", "C"],
    "KURLA": ["C", "H"],
    "MUMBAI CST": ["C", "H"],
    "ANDHERI": ["W", "H"],
    "BANDRA": ["W", "H"],
    "MAHIM": ["W", "H"],
}

RAILWAY_CACHE: Dict[str, Dict[str, Any]] = {}
RAILWAY_CACHE_TTL_SECONDS = 300
RAILWAY_HTTP_TIMEOUT_SECONDS = 10.0


def _rail_normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _rail_time_to_minutes(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip().upper().replace(".", "")
    raw = re.sub(r"\s+", " ", raw)
    match = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?$", raw)
    if not match:
        match = re.match(r"^(\d{1,2})\s*(AM|PM)$", raw)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2).isdigit() else 0
    meridiem = match.group(3)

    if meridiem:
        if hour == 12:
            hour = 0
        if meridiem == "PM":
            hour += 12

    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _rail_minutes_to_24h(total_minutes: int) -> str:
    total_minutes %= 1440
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _rail_minutes_to_12h(total_minutes: int) -> str:
    total_minutes %= 1440
    hour = total_minutes // 60
    minute = total_minutes % 60
    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {period}"


def _rail_slug_from_station(station: str, line: str = "") -> str:
    raw = str(station or "").strip()

    # First accept app station code.
    upper = raw.upper()
    if upper in RAILWAY_STATIONS:
        return str(RAILWAY_STATIONS[upper]["slug"])

    # Then accept station name.
    norm = _rail_normalize(raw)
    for code, data in RAILWAY_STATIONS.items():
        if _rail_normalize(data["name"]) == norm:
            return str(data["slug"])

    aliases = {
        "mumbaicsmt": "mumbai_cst",
        "mumbaicst": "mumbai_cst",
        "cst": "mumbai_cst",
        "csmt": "mumbai_cst",
        "vadalaroad": "vadala_road",
        "vasairoad": "vasai_road",
        "nalasopara": "nalla_sopara",
        "nallasopara": "nalla_sopara",
        "belapur": "belapur_cbd",
        "belapurcbd": "belapur_cbd",
        "kharroad": "khar_road",
    }
    if norm in aliases:
        return aliases[norm]

    # Safe generic fallback. The public reference uses underscore slugs.
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", raw.lower())).strip("_")


def _rail_station_display_name(station: str) -> str:
    upper = str(station or "").strip().upper()
    if upper in RAILWAY_STATIONS:
        return str(RAILWAY_STATIONS[upper]["name"])

    norm = _rail_normalize(station)
    for data in RAILWAY_STATIONS.values():
        if _rail_normalize(data["name"]) == norm:
            return str(data["name"])

    return str(station or "Unknown Station").strip()


def _rail_lines_for_station(station: str) -> List[str]:
    name = _rail_station_display_name(station)
    norm = name.upper()

    # Explicit interchange knowledge first.
    if norm in RAILWAY_INTERCHANGES:
        return list(RAILWAY_INTERCHANGES[norm])

    lines: List[str] = []
    upper = str(station or "").strip().upper()
    if upper in RAILWAY_STATIONS:
        lines.append(str(RAILWAY_STATIONS[upper]["line"]))

    for data in RAILWAY_STATIONS.values():
        if _rail_normalize(data["name"]) == _rail_normalize(name):
            line = str(data["line"])
            if line not in lines:
                lines.append(line)

    return lines


def _rail_line_name(line: str) -> str:
    return {
        "W": "Western Railway",
        "C": "Central Railway",
        "H": "Harbour Line",
    }.get(str(line).upper(), "Mumbai Suburban Railway")


def _rail_route_slug(line: str) -> str:
    return {
        "W": "western",
        "C": "central",
        "H": "harbour",
    }.get(str(line).upper(), "western")


def _rail_legacy_station_param(station: str, line: str = "") -> str:
    """Return the station parameter used by Mumbai Lifeline's server-rendered timetable."""
    name = _rail_station_display_name(station)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def _rail_route_url(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> str:
    """Return the primary modern Mumbai Lifeline timetable URL."""
    line_slug = _rail_route_slug(line)
    from_slug = _rail_slug_from_station(from_station, line)
    to_slug = _rail_slug_from_station(to_station, line)
    params = {
        "after": _rail_minutes_to_24h(after_minutes),
        "before": _rail_minutes_to_24h(before_minutes),
    }
    return (
        f"{RAILWAY_REFERENCE_BASE}/timetable/"
        f"{line_slug}/{from_slug}/{to_slug}?{urllib.parse.urlencode(params)}"
    )


def _rail_legacy_route_url(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> str:
    """Fallback URL for Mumbai Lifeline's server-rendered timetable.php."""
    params = {
        "Submit": "Submit",
        "sel_route": _rail_route_slug(line),
        "sfrom": _rail_legacy_station_param(from_station, line),
        "sto": _rail_legacy_station_param(to_station, line),
        "time1": _rail_minutes_to_12h(after_minutes),
        "time2": _rail_minutes_to_12h(before_minutes),
    }
    return f"{RAILWAY_REFERENCE_BASE}/timetable.php?{urllib.parse.urlencode(params)}"


def _rail_mobile_route_url(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
) -> str:
    """Last-resort lightweight timetable page used when table HTML is unavailable."""
    params = {
        "sfrom": _rail_legacy_station_param(from_station, line),
        "sto": _rail_legacy_station_param(to_station, line),
        "time1": _rail_minutes_to_12h(after_minutes),
    }
    if str(line).upper() == "W":
        path = "/m/timetable_western.php"
    else:
        path = "/m/timetable.php"
    return f"{RAILWAY_REFERENCE_BASE}{path}?{urllib.parse.urlencode(params)}"


def _rail_candidate_urls(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> List[str]:
    urls = [
        _rail_route_url(line, from_station, to_station, after_minutes, before_minutes),
        _rail_legacy_route_url(line, from_station, to_station, after_minutes, before_minutes),
        _rail_mobile_route_url(line, from_station, to_station, after_minutes),
    ]
    return list(dict.fromkeys(urls))

def _rail_cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = RAILWAY_CACHE.get(key)
    if not item:
        return None
    if time.time() - float(item.get("timestamp", 0)) > RAILWAY_CACHE_TTL_SECONDS:
        RAILWAY_CACHE.pop(key, None)
        return None
    return item.get("data")


def _rail_cache_set(key: str, data: Dict[str, Any]) -> None:
    RAILWAY_CACHE[key] = {
        "timestamp": time.time(),
        "data": data,
    }

    # Keep memory bounded.
    if len(RAILWAY_CACHE) > 80:
        oldest_key = min(
            RAILWAY_CACHE.keys(),
            key=lambda k: float(RAILWAY_CACHE[k].get("timestamp", 0)),
        )
        RAILWAY_CACHE.pop(oldest_key, None)


def _rail_clean_train_no(value: str) -> str:
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    raw = re.sub(r"\b(MON[–-]SAT|SAT[–-]SUN|DAILY|MON-SAT|SAT-SUN)\b", "", raw, flags=re.I)
    raw = re.sub(r"\s+", " ", raw).strip(" -|")
    return raw.split(" ")[0] if raw else "—"


def _rail_days_from_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Daily / per timetable"
    for token in ["Mon–Sat", "Sat–Sun", "Daily", "Mon-Sat", "Sat-Sun"]:
        if token.lower() in raw.lower():
            return token.replace("–", "-")
    return "As published"


def _rail_detect_service(train_no_cell: str, speed_cell: str) -> Tuple[str, str]:
    no_text = str(train_no_cell or "").upper()
    speed = str(speed_cell or "").strip().upper()

    if "AC" in no_text:
        return "AC", "Air-conditioned local"
    if "LADIES" in no_text or "WOMEN" in no_text:
        return "LADIES", "Ladies special"
    if "FAST" in speed:
        return "F", "Fast"
    if "SLOW" in speed:
        return "S", "Slow"
    if "MEDIUM" in speed:
        return "M", "Medium"
    return "S", speed.title() if speed and speed != "—" else "Local"


def _rail_find_header(headers: List[str], candidates: List[str]) -> Optional[int]:
    normalized_headers = [_rail_normalize(h) for h in headers]
    for candidate in candidates:
        candidate_norm = _rail_normalize(candidate)
        for idx, header_norm in enumerate(normalized_headers):
            if candidate_norm and candidate_norm in header_norm:
                return idx
    return None



def _rail_table_rows(html: str) -> List[Tuple[List[str], str]]:
    """
    Extract table rows as (cell_texts, row_html).

    Uses BeautifulSoup when available, but includes a standard-library
    HTMLParser fallback so Railway parsing does not depend on bs4.
    """
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            rows: List[Tuple[List[str], str]] = []
            for row in soup.find_all("tr"):
                cells = row.find_all(["th", "td"])
                values = [
                    re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                    for cell in cells
                ]
                if values:
                    rows.append((values, str(row)))
            return rows
        except Exception:
            pass

    try:
        from html.parser import HTMLParser

        class _RowParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.rows: List[List[str]] = []
                self.current_row: Optional[List[str]] = None
                self.in_cell = False
                self.buffer: List[str] = []

            def handle_starttag(self, tag: str, attrs) -> None:
                tag = tag.lower()
                if tag == "tr":
                    self.current_row = []
                elif tag in ("td", "th") and self.current_row is not None:
                    self.in_cell = True
                    self.buffer = []

            def handle_data(self, data: str) -> None:
                if self.in_cell:
                    self.buffer.append(data)

            def handle_endtag(self, tag: str) -> None:
                tag = tag.lower()
                if tag in ("td", "th") and self.in_cell:
                    value = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
                    if self.current_row is not None:
                        self.current_row.append(value)
                    self.buffer = []
                    self.in_cell = False
                elif tag == "tr" and self.current_row is not None:
                    if any(cell.strip() for cell in self.current_row):
                        self.rows.append(self.current_row)
                    self.current_row = None

        parser = _RowParser()
        parser.feed(html)
        parser.close()
        return [(row, "") for row in parser.rows]
    except Exception:
        return []

def _rail_parse_timetable_html(
    html: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """
    Parse the actual timetable table from Mumbai Lifeline HTML.

    The website can place navigation/summary rows before the timetable, so
    this scans all tables/rows instead of assuming row zero is the header.
    It works even when BeautifulSoup is unavailable by using the existing
    _rail_table_rows fallback.
    """
    requested_from_name = _rail_station_display_name(from_station)
    requested_to_name = _rail_station_display_name(to_station)

    rows_with_html = _rail_table_rows(html)
    if not rows_with_html:
        return []

    header_index = None
    header_cells: List[str] = []

    for i, (values, _row_html) in enumerate(rows_with_html):
        if len(values) < 3:
            continue

        normalized = [_rail_normalize(v) for v in values]
        has_train = any(
            h in {"trainno", "trainnumber", "train", "trainname"}
            or "trainno" in h
            for h in normalized
        )
        has_speed = any("speed" in h for h in normalized)
        has_requested_from = any(
            h == _rail_normalize(requested_from_name)
            or h == _rail_normalize(from_station)
            or h == _rail_normalize(str(from_station).upper())
            for h in normalized
        )
        has_requested_to = any(
            h == _rail_normalize(requested_to_name)
            or h == _rail_normalize(to_station)
            or h == _rail_normalize(str(to_station).upper())
            for h in normalized
        )

        if has_train and (has_speed or has_requested_from or has_requested_to):
            header_index = i
            header_cells = values
            break

    if header_index is None:
        return []

    train_idx = _rail_find_header(
        header_cells,
        ["Train No", "Train Number", "Train Name", "Train"],
    )
    speed_idx = _rail_find_header(header_cells, ["Speed"])
    coach_idx = _rail_find_header(header_cells, ["Coaches", "Coach"])
    origin_idx = _rail_find_header(
        header_cells,
        ["Originating From", "Origin", "Starting From"],
    )
    ending_idx = _rail_find_header(
        header_cells,
        ["Ending At", "Destination", "Ending"],
    )
    duration_idx = _rail_find_header(header_cells, ["Duration"])

    from_time_idx = _rail_find_header(
        header_cells,
        [requested_from_name, from_station, str(from_station).upper()],
    )
    to_time_idx = _rail_find_header(
        header_cells,
        [requested_to_name, to_station, str(to_station).upper()],
    )

    if train_idx is None or from_time_idx is None:
        return []

    trains: List[Dict[str, Any]] = []

    for values, _row_html in rows_with_html[header_index + 1:]:
        if len(values) <= max(train_idx, from_time_idx):
            continue

        train_cell = re.sub(
            r"\[Button:\s*(.*?)\]",
            r"\1",
            str(values[train_idx] or ""),
        )
        train_cell = re.sub(r"\s+", " ", train_cell).strip()

        dep_raw = re.sub(
            r"\[Button:\s*(.*?)\]",
            r"\1",
            str(values[from_time_idx] or ""),
        )
        dep_raw = re.sub(r"\s+", " ", dep_raw).strip()

        # Header/separator rows and UI-only rows are ignored.
        if (
            not train_cell
            or not dep_raw
            or dep_raw in {"—", "-", "–", "|"}
            or _rail_normalize(train_cell) in {
                "trainno", "trainnumber", "trainname", "train"
            }
        ):
            continue

        dep_min = _rail_time_to_minutes(dep_raw)
        if dep_min is None:
            # Some rows may have the time wrapped inside extra text. Extract it.
            m = re.search(
                r"\b(\d{1,2}:\d{2}\s*(?:AM|PM)|\d{1,2}\s*(?:AM|PM))\b",
                dep_raw,
                re.I,
            )
            if m:
                dep_raw = m.group(1)
                dep_min = _rail_time_to_minutes(dep_raw)
        if dep_min is None:
            continue

        arr_raw = ""
        arr_min = None
        if to_time_idx is not None and to_time_idx < len(values):
            arr_raw = re.sub(
                r"\[Button:\s*(.*?)\]",
                r"\1",
                str(values[to_time_idx] or ""),
            )
            arr_raw = re.sub(r"\s+", " ", arr_raw).strip()
            arr_min = _rail_time_to_minutes(arr_raw)
            if arr_min is not None and arr_min < dep_min:
                arr_min += 1440

        speed = (
            str(values[speed_idx]).strip()
            if speed_idx is not None and speed_idx < len(values)
            else ""
        )
        coaches = (
            str(values[coach_idx]).strip()
            if coach_idx is not None and coach_idx < len(values)
            else None
        )
        origin = (
            str(values[origin_idx]).strip()
            if origin_idx is not None and origin_idx < len(values)
            else None
        )
        ending = (
            str(values[ending_idx]).strip()
            if ending_idx is not None and ending_idx < len(values)
            else None
        )
        duration = (
            str(values[duration_idx]).strip()
            if duration_idx is not None and duration_idx < len(values)
            else None
        )

        service_type, service_label = _rail_detect_service(
            train_cell,
            speed,
        )

        train_no = _rail_clean_train_no(train_cell)
        if train_no == "—":
            train_no = train_cell

        trains.append({
            "time": dep_raw,
            "departure_time": dep_raw,
            "arrival_time": (
                arr_raw if arr_raw not in {"", "—", "-", "–"} else None
            ),
            "timestamp_minutes": dep_min,
            "arrival_minutes": arr_min,
            "train_no": train_no,
            "name": f"{origin or requested_from_name} → "
                    f"{ending or requested_to_name}",
            "service_type": service_type,
            "service": service_label,
            "category": service_label,
            "platform": None,
            "platform_note": "Platform not published in the timetable source.",
            "status": "Scheduled",
            "status_msg": "Scheduled timetable entry",
            "delay_minutes": None,
            "crowd": None,
            "door_side": None,
            "coaches": coaches,
            "coach_count": coaches,
            "composition": coaches,
            "source": requested_from_name,
            "destination": requested_to_name,
            "origin": origin,
            "ending_at": ending,
            "duration": duration,
            "days": _rail_days_from_label(train_cell),
            "live": False,
            "data_source": source_url,
            "data_source_type": RAILWAY_REFERENCE_LABEL,
            "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(
                line, "Unknown"
            ),
        })

    unique: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for train in trains:
        key = (
            str(train.get("train_no", "")),
            int(train.get("timestamp_minutes", -1)),
            str(train.get("origin", "")),
            str(train.get("ending_at", "")),
        )
        unique[key] = train

    return sorted(
        unique.values(),
        key=lambda item: int(item.get("timestamp_minutes", 0)),
    )


def _rail_parse_mobile_timetable_html(
    html: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse the lightweight Mumbai Lifeline timetable as a final fallback."""
    if BeautifulSoup is None:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
    except Exception:
        return []

    station = re.escape(_rail_station_display_name(from_station))
    pattern = re.compile(
        rf"^{station}\s+to\s+(.+?)\s+(\d{{1,2}}:\d{{2}}\s*[ap]m)\s+(\d{{1,2}}:\d{{2}}\s*[ap]m)$",
        re.I,
    )

    trains: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line_text = re.sub(r"\s+", " ", raw_line).strip()
        match = pattern.match(line_text)
        if not match:
            continue

        destination = match.group(1).strip()
        dep_raw = match.group(2).strip()
        arr_raw = match.group(3).strip()
        dep_min = _rail_time_to_minutes(dep_raw)
        arr_min = _rail_time_to_minutes(arr_raw)
        if dep_min is None:
            continue
        if arr_min is not None and arr_min < dep_min:
            arr_min += 1440

        trains.append({
            "time": dep_raw,
            "departure_time": dep_raw,
            "arrival_time": arr_raw,
            "timestamp_minutes": dep_min,
            "arrival_minutes": arr_min,
            "train_no": "—",
            "name": f"{_rail_station_display_name(from_station)} → {destination}",
            "service_type": "S",
            "service": "Scheduled local",
            "category": "Scheduled local",
            "platform": None,
            "status": "Scheduled",
            "status_msg": "Scheduled timetable entry",
            "delay_minutes": None,
            "coaches": None,
            "coach_count": None,
            "composition": None,
            "source": _rail_station_display_name(from_station),
            "destination": destination,
            "origin": _rail_station_display_name(from_station),
            "ending_at": destination,
            "duration": None,
            "days": "As published",
            "live": False,
            "data_source": source_url,
            "data_source_type": RAILWAY_REFERENCE_LABEL,
            "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(line, "Unknown"),
        })

    unique: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for train in trains:
        key = (str(train.get("name")), int(train.get("timestamp_minutes", -1)), str(train.get("arrival_time")))
        unique[key] = train
    return sorted(unique.values(), key=lambda item: int(item.get("timestamp_minutes", 0)))


def _rail_parse_timetable_markdown(
    text: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse the Markdown table returned by a web reader such as Jina."""
    requested_from_name = _rail_station_display_name(from_station)
    requested_to_name = _rail_station_display_name(to_station)

    clean_lines = [
        re.sub(r"\s+", " ", x).strip()
        for x in str(text or "").splitlines()
        if x.strip()
    ]

    header_index = None
    header_cells: List[str] = []

    for i, line_text in enumerate(clean_lines):
        if "|" not in line_text:
            continue

        cells = [c.strip() for c in line_text.strip("|").split("|")]
        normalized = [_rail_normalize(c) for c in cells]

        has_train = any(
            h in {"trainno", "trainnumber", "train", "trainname"}
            or "trainno" in h
            for h in normalized
        )
        has_speed = any("speed" in h for h in normalized)
        has_station = any(
            h == _rail_normalize(requested_from_name)
            or h == _rail_normalize(requested_to_name)
            or h == _rail_normalize(from_station)
            or h == _rail_normalize(to_station)
            for h in normalized
        )

        if has_train and (has_speed or has_station):
            header_index = i
            header_cells = cells
            break

    if header_index is None:
        return []

    train_idx = _rail_find_header(
        header_cells,
        ["Train No", "Train Number", "Train Name", "Train"],
    )
    speed_idx = _rail_find_header(header_cells, ["Speed"])
    coach_idx = _rail_find_header(header_cells, ["Coaches", "Coach"])
    origin_idx = _rail_find_header(
        header_cells,
        ["Originating From", "Origin", "Starting From"],
    )
    ending_idx = _rail_find_header(
        header_cells,
        ["Ending At", "Destination", "Ending"],
    )
    duration_idx = _rail_find_header(header_cells, ["Duration"])
    from_time_idx = _rail_find_header(
        header_cells,
        [requested_from_name, from_station, str(from_station).upper()],
    )
    to_time_idx = _rail_find_header(
        header_cells,
        [requested_to_name, to_station, str(to_station).upper()],
    )

    if train_idx is None or from_time_idx is None:
        return []

    trains: List[Dict[str, Any]] = []

    for line_text in clean_lines[header_index + 1:]:
        if "|" not in line_text:
            continue

        values = [c.strip() for c in line_text.strip("|").split("|")]

        if all(
            (not value)
            or re.fullmatch(r"[-: ]+", value)
            for value in values
        ):
            continue

        if any(
            idx >= len(values)
            for idx in [train_idx, from_time_idx]
        ):
            continue

        train_cell = re.sub(
            r"\[Button:\s*(.*?)\]",
            r"\1",
            values[train_idx],
        )
        train_cell = re.sub(r"\s+", " ", train_cell).strip()

        dep_raw = re.sub(
            r"\[Button:\s*(.*?)\]",
            r"\1",
            values[from_time_idx],
        )
        dep_raw = re.sub(r"\s+", " ", dep_raw).strip()

        dep_min = _rail_time_to_minutes(dep_raw)
        if dep_min is None:
            continue

        arr_raw = ""
        arr_min = None
        if to_time_idx is not None and to_time_idx < len(values):
            arr_raw = re.sub(
                r"\[Button:\s*(.*?)\]",
                r"\1",
                values[to_time_idx],
            )
            arr_raw = re.sub(r"\s+", " ", arr_raw).strip()
            arr_min = _rail_time_to_minutes(arr_raw)
            if arr_min is not None and arr_min < dep_min:
                arr_min += 1440

        speed = (
            values[speed_idx].strip()
            if speed_idx is not None and speed_idx < len(values)
            else ""
        )
        coaches = (
            values[coach_idx].strip()
            if coach_idx is not None and coach_idx < len(values)
            else None
        )
        origin = (
            values[origin_idx].strip()
            if origin_idx is not None and origin_idx < len(values)
            else None
        )
        ending = (
            values[ending_idx].strip()
            if ending_idx is not None and ending_idx < len(values)
            else None
        )
        duration = (
            values[duration_idx].strip()
            if duration_idx is not None and duration_idx < len(values)
            else None
        )

        service_type, service_label = _rail_detect_service(
            train_cell,
            speed,
        )

        trains.append({
            "time": dep_raw,
            "departure_time": dep_raw,
            "arrival_time": (
                arr_raw if arr_raw not in {"", "—", "-", "–"} else None
            ),
            "timestamp_minutes": dep_min,
            "arrival_minutes": arr_min,
            "train_no": _rail_clean_train_no(train_cell),
            "name": f"{origin or requested_from_name} → "
                    f"{ending or requested_to_name}",
            "service_type": service_type,
            "service": service_label,
            "category": service_label,
            "platform": None,
            "platform_note": "Platform not published in the timetable source.",
            "status": "Scheduled",
            "status_msg": "Scheduled timetable entry",
            "delay_minutes": None,
            "crowd": None,
            "door_side": None,
            "coaches": coaches,
            "coach_count": coaches,
            "composition": coaches,
            "source": requested_from_name,
            "destination": requested_to_name,
            "origin": origin,
            "ending_at": ending,
            "duration": duration,
            "days": _rail_days_from_label(train_cell),
            "live": False,
            "data_source": source_url,
            "data_source_type": RAILWAY_REFERENCE_LABEL,
            "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(
                line, "Unknown"
            ),
        })

    unique: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for train in trains:
        key = (
            str(train.get("train_no", "")),
            int(train.get("timestamp_minutes", -1)),
            str(train.get("origin", "")),
            str(train.get("ending_at", "")),
        )
        unique[key] = train

    return sorted(
        unique.values(),
        key=lambda item: int(item.get("timestamp_minutes", 0)),
    )

async def _rail_fetch_timetable(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> Dict[str, Any]:
    """
    Multi-layer timetable retrieval.

    1. Direct modern page.
    2. Jina Reader mirror of the exact modern page.
    3. Direct legacy timetable.php.
    4. Jina Reader mirror of legacy timetable.php.
    5. Direct mobile timetable.

    The public timetable remains the data source. Jina is only a retrieval
    adapter when the source HTML is not exposed cleanly to Render.
    """
    line = str(line).upper()
    from_name = _rail_station_display_name(from_station)
    to_name = _rail_station_display_name(to_station)

    primary_url = _rail_route_url(
        line=line,
        from_station=from_station,
        to_station=to_station,
        after_minutes=after_minutes,
        before_minutes=before_minutes,
    )

    cache_key = (
        f"{line}|{from_name}|{to_name}|"
        f"{after_minutes}|{before_minutes}"
    )

    cached = _rail_cache_get(cache_key)
    if cached is not None and cached.get("trains"):
        return cached

    result = {
        "trains": [],
        "source_url": primary_url,
        "source_label": RAILWAY_REFERENCE_LABEL,
        "line": line,
        "line_name": _rail_line_name(line),
        "from": from_name,
        "to": to_name,
        "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(line, "Unknown"),
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "error": None,
        "retrieval_path": None,
        "attempted_sources": [],
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36 "
            "OmniTouristOS-Rail/94.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,text/plain;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "en-IN,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": f"{RAILWAY_REFERENCE_BASE}/timetable",
    }

    retrieval_errors: List[str] = []

    legacy_url = _rail_legacy_route_url(
        line,
        from_station,
        to_station,
        after_minutes,
        before_minutes,
    )

    mobile_url = _rail_mobile_route_url(
        line,
        from_station,
        to_station,
        after_minutes,
    )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=RAILWAY_HTTP_TIMEOUT_SECONDS,
        headers=headers,
    ) as client:

        # ---------------- 1. Direct modern URL ----------------
        try:
            result["attempted_sources"].append(primary_url)
            response = await client.get(primary_url)

            if response.status_code == 200 and response.text:
                parsed = _rail_parse_timetable_html(
                    response.text,
                    line=line,
                    from_station=from_station,
                    to_station=to_station,
                    source_url=primary_url,
                )
                if parsed:
                    result["trains"] = parsed
                    result["retrieval_path"] = "direct_modern"
        except Exception as exc:
            retrieval_errors.append(f"Direct modern: {exc}")

        # ---------------- 2. Jina Reader modern ----------------
        if not result["trains"]:
            try:
                # Quote the complete target URL so its ? and & are retained
                # as part of Jina's target rather than becoming outer params.
                reader_target = urllib.parse.quote(
                    primary_url,
                    safe=":/",
                )
                reader_url = f"https://r.jina.ai/{reader_target}"
                result["attempted_sources"].append(reader_url)

                response = await client.get(
                    reader_url,
                    headers={
                        "User-Agent": "OmniTouristOS Railway Reader/1.0",
                        "Accept": "text/plain,text/markdown,*/*",
                    },
                )

                if response.status_code == 200 and response.text:
                    parsed = _rail_parse_timetable_markdown(
                        response.text,
                        line=line,
                        from_station=from_station,
                        to_station=to_station,
                        source_url=primary_url,
                    )

                    if not parsed:
                        parsed = _rail_parse_timetable_html(
                            response.text,
                            line=line,
                            from_station=from_station,
                            to_station=to_station,
                            source_url=primary_url,
                        )

                    if parsed:
                        result["trains"] = parsed
                        result["retrieval_path"] = "jina_modern"
            except Exception as exc:
                retrieval_errors.append(f"Jina modern: {exc}")

        # ---------------- 3. Direct legacy timetable.php ----------------
        if not result["trains"]:
            try:
                result["attempted_sources"].append(legacy_url)
                response = await client.get(legacy_url)

                if response.status_code == 200 and response.text:
                    parsed = _rail_parse_timetable_html(
                        response.text,
                        line=line,
                        from_station=from_station,
                        to_station=to_station,
                        source_url=legacy_url,
                    )
                    if parsed:
                        result["trains"] = parsed
                        result["source_url"] = legacy_url
                        result["retrieval_path"] = "direct_legacy"
            except Exception as exc:
                retrieval_errors.append(f"Direct legacy: {exc}")

        # ---------------- 4. Jina Reader legacy ----------------
        if not result["trains"]:
            try:
                reader_target = urllib.parse.quote(
                    legacy_url,
                    safe=":/",
                )
                reader_url = f"https://r.jina.ai/{reader_target}"
                result["attempted_sources"].append(reader_url)

                response = await client.get(
                    reader_url,
                    headers={
                        "User-Agent": "OmniTouristOS Railway Reader/1.0",
                        "Accept": "text/plain,text/markdown,*/*",
                    },
                )

                if response.status_code == 200 and response.text:
                    parsed = _rail_parse_timetable_markdown(
                        response.text,
                        line=line,
                        from_station=from_station,
                        to_station=to_station,
                        source_url=legacy_url,
                    )

                    if not parsed:
                        parsed = _rail_parse_timetable_html(
                            response.text,
                            line=line,
                            from_station=from_station,
                            to_station=to_station,
                            source_url=legacy_url,
                        )

                    if parsed:
                        result["trains"] = parsed
                        result["source_url"] = legacy_url
                        result["retrieval_path"] = "jina_legacy"
            except Exception as exc:
                retrieval_errors.append(f"Jina legacy: {exc}")

        # ---------------- 5. Direct mobile page ----------------
        if not result["trains"]:
            try:
                result["attempted_sources"].append(mobile_url)
                response = await client.get(mobile_url)

                if response.status_code == 200 and response.text:
                    parsed = _rail_parse_mobile_timetable_html(
                        response.text,
                        line=line,
                        from_station=from_station,
                        to_station=to_station,
                        source_url=mobile_url,
                    )
                    if parsed:
                        result["trains"] = parsed
                        result["source_url"] = mobile_url
                        result["source_label"] = (
                            RAILWAY_REFERENCE_LABEL
                            + " (mobile fallback)"
                        )
                        result["retrieval_path"] = "direct_mobile"
            except Exception as exc:
                retrieval_errors.append(f"Direct mobile: {exc}")

        if result["trains"]:
            result["error"] = None
            result["train_count"] = len(result["trains"])
            # Only cache successful schedules. An outage/parse failure should
            # never be sticky for five minutes.
            _rail_cache_set(cache_key, result)
            return result

        result["train_count"] = 0
        result["error"] = (
            "The public timetable source was reached, but no scheduled rows "
            "could be extracted by the available retrieval methods."
        )
        if retrieval_errors:
            result["retrieval_errors"] = retrieval_errors[-5:]

        # Do not cache failure responses.
        return result


def _rail_filter_trains(
    trains: List[Dict[str, Any]],
    service: str,
) -> List[Dict[str, Any]]:
    wanted = str(service or "ALL").strip().upper()
    if wanted in {"", "ALL"}:
        return list(trains)

    output: List[Dict[str, Any]] = []
    for train in trains:
        service_type = str(train.get("service_type", "")).upper()
        label = str(train.get("service", "")).upper()
        category = str(train.get("category", "")).upper()

        if wanted == "FAST" and (service_type == "F" or "FAST" in label):
            output.append(train)
        elif wanted == "SLOW" and (service_type == "S" or "SLOW" in label):
            output.append(train)
        elif wanted == "AC" and (service_type == "AC" or "AC" in label or "AC" in category):
            output.append(train)
        elif wanted == "LADIES" and (
            service_type == "LADIES"
            or "LADIES" in label
            or "WOMEN" in category
        ):
            output.append(train)

    return output


def _rail_default_board_targets(station: str, line: str) -> List[str]:
    station_name = _rail_station_display_name(station)
    norm = _rail_normalize(station_name)

    if line == "W":
        targets = ["churchgate", "virar"]
        if norm == _rail_normalize("Churchgate"):
            targets = ["virar", "borivali"]
        elif norm == _rail_normalize("Virar"):
            targets = ["churchgate", "dahanu_road"]
        elif norm in {
            _rail_normalize("Palghar"),
            _rail_normalize("Boisar"),
            _rail_normalize("Dahanu Road"),
        }:
            targets = ["churchgate", "virar"]
        return targets

    if line == "C":
        # Main-line board. At Kalyan and key junctions include the branches.
        if norm == _rail_normalize("Kalyan"):
            return ["mumbai_cst", "kasara", "khopoli"]
        if norm in {
            _rail_normalize("Karjat"),
            _rail_normalize("Neral"),
            _rail_normalize("Badlapur"),
            _rail_normalize("Ambernath"),
        }:
            return ["mumbai_cst", "kalyan"]
        if norm in {_rail_normalize("Kasara"), _rail_normalize("Asangaon"), _rail_normalize("Vasind"), _rail_normalize("Titwala")}:
            return ["mumbai_cst", "kalyan"]
        if norm == _rail_normalize("Mumbai CST"):
            return ["kalyan", "khopoli", "kasara"]
        return ["mumbai_cst", "kalyan"]

    if line == "H":
        if norm == _rail_normalize("Mumbai CST"):
            return ["panvel", "goregaon_hr", "andheri_hr"]
        if norm in {
            _rail_normalize("Vadala Road"),
            _rail_normalize("Bandra"),
            _rail_normalize("Andheri"),
            _rail_normalize("Goregaon"),
        }:
            return ["mumbai_cst", "panvel"]
        if norm == _rail_normalize("Panvel"):
            return ["mumbai_cst", "goregaon_hr"]
        return ["mumbai_cst", "panvel"]

    return []


def _rail_station_name_from_slug(slug: str) -> str:
    norm = _rail_normalize(slug)
    for data in RAILWAY_STATIONS.values():
        if _rail_normalize(data["slug"]) == norm:
            return str(data["name"])
    return slug.replace("_", " ").title()


def _rail_resolve_line_for_pair(
    from_station: str,
    to_station: str,
    requested_line: str,
) -> Tuple[Optional[str], Optional[str]]:
    requested = str(requested_line or "ALL").upper()
    from_lines = _rail_lines_for_station(from_station)
    to_lines = _rail_lines_for_station(to_station)

    if requested in {"W", "C", "H"}:
        if requested in from_lines and requested in to_lines:
            return requested, None
        return requested, "Selected line does not directly serve both stations."

    common = [line for line in ["W", "C", "H"] if line in from_lines and line in to_lines]
    if common:
        return common[0], None

    if from_lines and to_lines:
        return None, "Interchange route required."

    return None, "Could not identify a Mumbai suburban line for one or both stations."


async def _rail_cross_line_route(
    from_station: str,
    to_station: str,
    from_lines: List[str],
    to_lines: List[str],
) -> Dict[str, Any]:
    # Pick the most common/simple interchange so the backend stays lightweight.
    interchange_rules = [
        ("Dadar", "W", "C"),
        ("Andheri", "W", "H"),
        ("Kurla", "C", "H"),
        ("Mumbai CST", "C", "H"),
        ("Bandra", "W", "H"),
        ("Mahim", "W", "H"),
    ]

    choice = None
    for interchange, line_a, line_b in interchange_rules:
        if line_a in from_lines and line_b in to_lines:
            choice = (interchange, line_a, line_b)
            break
        if line_b in from_lines and line_a in to_lines:
            choice = (interchange, line_b, line_a)
            break

    if choice is None:
        return {
            "trains": [],
            "requires_interchange": True,
            "interchange_options": [],
            "message": "Stations are on different suburban lines. Choose an interchange station such as Dadar, Andheri, Kurla or Mumbai CST.",
        }

    interchange, first_line, second_line = choice
    now = datetime.now()
    start_min = now.hour * 60 + now.minute
    window_end = min(start_min + 180, 1439)

    first = await _rail_fetch_timetable(
        first_line,
        from_station,
        interchange,
        start_min,
        window_end,
    )
    second = await _rail_fetch_timetable(
        second_line,
        interchange,
        to_station,
        start_min,
        window_end,
    )

    first_trains = first.get("trains", [])
    second_trains = second.get("trains", [])

    journeys: List[Dict[str, Any]] = []

    for leg1 in first_trains:
        leg1_arrival = leg1.get("arrival_minutes")
        if leg1_arrival is None:
            continue

        for leg2 in second_trains:
            leg2_depart = leg2.get("timestamp_minutes")
            if leg2_depart is None:
                continue

            effective_arrival = int(leg1_arrival)
            if effective_arrival < int(leg1.get("timestamp_minutes", 0)):
                effective_arrival += 1440

            connection = int(leg2_depart) - effective_arrival
            if 5 <= connection <= 40:
                leg2_arrival = leg2.get("arrival_minutes")
                duration_text = "Scheduled interchange"

                if leg2_arrival is not None:
                    total_duration = int(leg2_arrival) - int(leg1.get("timestamp_minutes", 0))
                    if total_duration < 0:
                        total_duration += 1440
                    hours = total_duration // 60
                    minutes = total_duration % 60
                    duration_text = f"{hours}h {minutes}m" if hours else f"{minutes} min"

                journeys.append({
                    "time": leg1.get("departure_time"),
                    "departure_time": leg1.get("departure_time"),
                    "arrival_time": leg2.get("arrival_time"),
                    "timestamp_minutes": leg1.get("timestamp_minutes"),
                    "train_no": f"{leg1.get('train_no', '—')} + {leg2.get('train_no', '—')}",
                    "name": f"{from_station} → {interchange} → {to_station}",
                    "service_type": "CHANGE",
                    "service": "Interchange",
                    "category": "Connected route",
                    "platform": None,
                    "status": f"Scheduled • Change at {interchange}",
                    "status_msg": f"Change at {interchange} • {connection} min connection",
                    "delay_minutes": None,
                    "crowd": None,
                    "door_side": None,
                    "coaches": None,
                    "source": _rail_station_display_name(from_station),
                    "destination": _rail_station_display_name(to_station),
                    "origin": _rail_station_display_name(from_station),
                    "ending_at": _rail_station_display_name(to_station),
                    "duration": duration_text,
                    "days": "As published",
                    "live": False,
                    "interchange": interchange,
                    "connection_minutes": connection,
                    "leg_1": leg1,
                    "leg_2": leg2,
                    "data_source": [
                        first.get("source_url"),
                        second.get("source_url"),
                    ],
                    "data_source_type": RAILWAY_REFERENCE_LABEL,
                    "timetable_version": (
                        f"{RAILWAY_TIMETABLE_VERSIONS.get(first_line, 'Unknown')} / "
                        f"{RAILWAY_TIMETABLE_VERSIONS.get(second_line, 'Unknown')}"
                    ),
                })
                break

        if len(journeys) >= 12:
            break

    journeys.sort(key=lambda item: int(item.get("timestamp_minutes", 0)))

    return {
        "trains": journeys[:12],
        "requires_interchange": True,
        "interchange": interchange,
        "from": _rail_station_display_name(from_station),
        "to": _rail_station_display_name(to_station),
        "line": f"{first_line}+{second_line}",
        "line_name": f"{_rail_line_name(first_line)} + {_rail_line_name(second_line)}",
        "source_label": RAILWAY_REFERENCE_LABEL,
    }


@app.post("/api/v1/railway-inquiry")
async def railway_inquiry(request: Request):
    """
    Backward-compatible railway gateway.

    Supported query_type values:
      - station_board
      - route_search
      - live_train
      - pnr

    The Flutter client can continue using the same endpoint and payload shape.
    """

    try:
        content_type = request.headers.get("content-type", "").lower()

        query_type = "station_board"
        query_value: Any = ""
        target_language = "English"

        if "application/json" in content_type:
            body = await request.json()
            query_type = str(body.get("query_type", "station_board"))
            query_value = body.get("query_value", "")
            target_language = str(body.get("target_language", "English"))
        else:
            form = await request.form()
            query_type = str(form.get("query_type", "station_board"))
            query_value = form.get("query_value", "")
            target_language = str(form.get("target_language", "English"))

        normalized_type = query_type.strip().lower()

        # ----------------------------- ROUTE SEARCH -----------------------------
        if normalized_type == "route_search":
            try:
                route_payload = json.loads(str(query_value)) if query_value else {}
            except Exception:
                route_payload = {}

            from_station = str(
                route_payload.get("from")
                or route_payload.get("source")
                or route_payload.get("from_station")
                or "BSR"
            ).strip()

            to_station = str(
                route_payload.get("to")
                or route_payload.get("destination")
                or route_payload.get("to_station")
                or "DDR"
            ).strip()

            requested_line = str(route_payload.get("line") or "ALL").upper()
            requested_service = str(route_payload.get("service") or "ALL").upper()

            if _rail_normalize(from_station) == _rail_normalize(to_station):
                return {
                    "status": "error",
                    "type": "route_search",
                    "message": "Origin and destination cannot be the same station.",
                    "trains": [],
                }

            direct_line, line_error = _rail_resolve_line_for_pair(
                from_station,
                to_station,
                requested_line,
            )

            if direct_line and not line_error:
                now = datetime.now()
                start_min = now.hour * 60 + now.minute
                window_end = min(start_min + 180, 1439)

                result = await _rail_fetch_timetable(
                    direct_line,
                    from_station,
                    to_station,
                    start_min,
                    window_end,
                )

                trains = _rail_filter_trains(
                    list(result.get("trains", [])),
                    requested_service,
                )

                return {
                    "status": "success",
                    "type": "route_search",
                    "station_code": str(from_station).upper(),
                    "station_name": _rail_station_display_name(from_station),
                    "from": _rail_station_display_name(from_station),
                    "to": _rail_station_display_name(to_station),
                    "line": direct_line,
                    "line_name": _rail_line_name(direct_line),
                    "trains": trains,
                    "route_search": True,
                    "requires_interchange": False,
                    "service_filter": requested_service,
                    "current_time": _rail_minutes_to_12h(start_min),
                    "timetable_version": result.get(
                        "timetable_version",
                        RAILWAY_TIMETABLE_VERSIONS.get(direct_line, "Unknown"),
                    ),
                    "data_source": result.get("source_url"),
                    "data_source_label": RAILWAY_REFERENCE_LABEL,
                    "source_notice": (
                        "Timetable values are from a public timetable reference; "
                        "they are scheduled values, not a live delay feed."
                    ),
                    "source_error": result.get("error"),
                }

            # Try an interchange journey when the selected stations span lines.
            from_lines = _rail_lines_for_station(from_station)
            to_lines = _rail_lines_for_station(to_station)

            if requested_line in {"W", "C", "H"}:
                from_lines = [requested_line] if requested_line in from_lines else from_lines
                to_lines = [requested_line] if requested_line in to_lines else to_lines

            if from_lines and to_lines:
                cross_line = await _rail_cross_line_route(
                    from_station=from_station,
                    to_station=to_station,
                    from_lines=from_lines,
                    to_lines=to_lines,
                )
                cross_line["status"] = "success"
                cross_line["type"] = "route_search"
                cross_line["route_search"] = True
                cross_line["current_time"] = _rail_minutes_to_12h(
                    datetime.now().hour * 60 + datetime.now().minute
                )
                cross_line["data_source_label"] = RAILWAY_REFERENCE_LABEL
                cross_line["source_notice"] = (
                    "Connected journeys are assembled from scheduled public timetable "
                    "entries; connection times are timetable-based, not live platform data."
                )
                return cross_line

            return {
                "status": "error",
                "type": "route_search",
                "message": line_error or "No supported suburban route found.",
                "trains": [],
            }

        # ----------------------------- STATION BOARD -----------------------------
        if normalized_type == "station_board":
            station = str(query_value or "BSR").strip().upper()
            station_name = _rail_station_display_name(station)
            station_lines = _rail_lines_for_station(station)

            if not station_lines:
                return {
                    "status": "error",
                    "type": "station_board",
                    "station_code": station,
                    "station_name": station_name,
                    "message": "Station not found in the Omni Rail station directory.",
                    "trains": [],
                }

            now = datetime.now()
            current_min = now.hour * 60 + now.minute
            board_start = 3 * 60 + 30
            board_end = 23 * 60 + 59

            all_trains: List[Dict[str, Any]] = []
            source_urls: List[str] = []
            source_errors: List[str] = []

            # A station can be an interchange (e.g. Dadar / Kurla / CSMT).
            # Return both line boards rather than silently selecting one.
            unique_line_set: List[str] = []
            for line in station_lines:
                if line not in unique_line_set:
                    unique_line_set.append(line)

            for line in unique_line_set:
                targets = _rail_default_board_targets(station, line)

                for target in targets:
                    target_name = _rail_station_name_from_slug(target)
                    if _rail_normalize(target_name) == _rail_normalize(station_name):
                        continue

                    result = await _rail_fetch_timetable(
                        line=line,
                        from_station=station,
                        to_station=target_name,
                        after_minutes=board_start,
                        before_minutes=board_end,
                    )

                    source_url = result.get("source_url")
                    if source_url:
                        source_urls.append(source_url)

                    if result.get("error"):
                        source_errors.append(str(result["error"]))

                    for train in result.get("trains", []):
                        item = dict(train)
                        item["line"] = line
                        item["line_name"] = _rail_line_name(line)
                        item["direction"] = target_name
                        all_trains.append(item)

            # Apply service-independent chronological ordering.
            all_trains.sort(
                key=lambda item: (
                    int(item.get("timestamp_minutes", 0)),
                    str(item.get("train_no", "")),
                )
            )

            # De-duplicate identical rows coming from overlapping direction searches.
            seen = set()
            deduped = []
            for train in all_trains:
                key = (
                    str(train.get("train_no")),
                    int(train.get("timestamp_minutes", -1)),
                    str(train.get("destination")),
                    str(train.get("line")),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(train)

            # Keep the complete day schedule practical for mobile JSON responses,
            # while preserving enough rows for the "next trains" experience.
            deduped = deduped[:450]

            upcoming = [
                t for t in deduped
                if int(t.get("timestamp_minutes", 0)) >= current_min
            ][:12]

            return {
                "status": "success",
                "type": "station_board",
                "station_code": station,
                "station_name": station_name,
                "line": unique_line_set[0] if len(unique_line_set) == 1 else "ALL",
                "line_name": (
                    _rail_line_name(unique_line_set[0])
                    if len(unique_line_set) == 1
                    else "Mumbai Suburban Network"
                ),
                "lines": unique_line_set,
                "line_names": [_rail_line_name(line) for line in unique_line_set],
                "current_time": _rail_minutes_to_12h(current_min),
                "board_window": "03:30 AM – 11:59 PM",
                "trains": deduped,
                "next_trains": upcoming,
                "train_count": len(deduped),
                "source_urls": sorted(set(source_urls)),
                "source_errors": sorted(set(source_errors))[:5],
                "source_label": RAILWAY_REFERENCE_LABEL,
                "timetable_versions": {
                    line: RAILWAY_TIMETABLE_VERSIONS.get(line, "Unknown")
                    for line in unique_line_set
                },
                "source_notice": (
                    "Scheduled timetable data is shown from a public Mumbai suburban "
                    "timetable reference. Platform, delay, door-side and crowd values "
                    "are not invented when no live feed exists."
                ),
                "station_directory": [
                    {
                        "code": code,
                        "name": data["name"],
                        "line": data["line"],
                    }
                    for code, data in RAILWAY_STATIONS.items()
                    if _rail_normalize(data["name"]) == _rail_normalize(station_name)
                ],
            }

        # ----------------------------- LIVE TRAIN -----------------------------
        if normalized_type == "live_train":
            train_query = str(query_value or "").strip()

            if not train_query:
                return {
                    "status": "error",
                    "type": "live_train",
                    "message": "Enter a train number or timetable train code.",
                }

            # We deliberately return a truthful response here.  The timetable
            # source does not provide real-time movement/delay data.
            return {
                "status": "success",
                "type": "live_train",
                "train_no": train_query,
                "train_name": "Mumbai Suburban Local",
                "current_station": None,
                "next_station": None,
                "platform": None,
                "door_side": None,
                "delay_minutes": None,
                "crowd": None,
                "status_msg": (
                    "Live movement data is not available from the current no-key "
                    "railway source. Use the Mumbai Local timetable for scheduled stops."
                ),
                "live": False,
                "live_feed_available": False,
                "message": (
                    "Omni Rail will display live location/delay/platform information "
                    "only when an authorized or genuinely live feed is connected."
                ),
                "data_source_label": RAILWAY_REFERENCE_LABEL,
            }

        # ----------------------------- PNR -----------------------------
        if normalized_type == "pnr":
            pnr = re.sub(r"\D", "", str(query_value or ""))

            if len(pnr) != 10:
                return {
                    "status": "error",
                    "type": "pnr",
                    "message": "PNR must contain exactly 10 digits.",
                }

            return {
                "status": "success",
                "type": "pnr",
                "pnr": pnr,
                "train_no": None,
                "train_name": None,
                "from_station": None,
                "to_station": None,
                "journey_date": None,
                "booking_status": "Live PNR lookup unavailable",
                "status_msg": (
                    "This backend does not have an authorized live PNR feed. "
                    "Please verify the PNR through an official railway service."
                ),
                "live": False,
                "data_source_label": "Official railway PNR service required",
            }

        return {
            "status": "error",
            "type": normalized_type,
            "message": "Unsupported railway query type.",
            "trains": [],
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Railway inquiry error: {exc}",
            "trains": [],
        }

# -------------------------------------------------------------

# 20. BARGAIN PAL (PRICE EVALUATOR)
# -------------------------------------------------------------
@app.post("/api/v1/bargain-evaluate")
async def bargain_evaluate(request: Request):
    try:
        body = await request.json()
        item_name = body.get("item_name", "Souvenir")
        quoted_price = float(body.get("quoted_price", 100))
        currency = body.get("currency", "INR")
        city = body.get("city", "Mumbai")

        sys_prompt = f"""
You are Bargain Pal, an authentic local street market expert for {city}.
Evaluate the quoted price for '{item_name}' ({quoted_price} {currency}).
Return STRICT JSON ONLY without markdown backticks.

JSON FORMAT:
{{
  "verdict": "Fair Price / Mild Markup / Tourist Trap",
  "rating_color": "green / yellow / red",
  "estimated_fair_price": 0.0,
  "suggested_counter_offer": 0.0,
  "advice": "1 practical sentence on local bargaining etiquette for this item.",
  "polite_counter_phrase": "Polite phrase in native script to negotiate",
  "phonetic": "Pronunciation in English letters",
  "phrase_translation": "English meaning of phrase"
}}
"""
        res = await ask_fast_json(f"Evaluate street price for {item_name}", sys_prompt)
        if res:
            return {"status": "success", "data": res}
        return {"status": "error", "message": "Evaluation timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------------------
# 21. WEBSOCKET REALTIME ROUTER, BOT SUPERVISOR & SERVER HEALTH
# -------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, room_id: str, message: dict):
        if room_id in self.active_connections:
            for connection in list(self.active_connections[room_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

@app.websocket("/ws/community/{community_id}")
async def community_websocket_endpoint(websocket: WebSocket, community_id: str):
    await manager.connect(community_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            sender_id = data.get("sender_id")
            sender_name = data.get("sender_name", "Scout")
            sender_avatar_url = data.get("sender_avatar_url")
            raw_text = (data.get("text") or "").strip()

            # 1. Enforcement Check: Banned or Muted
            if supabase and sender_id:
                try:
                    user_row = supabase.table("users").select("is_banned, muted_until").eq("id", sender_id).single().execute()
                    if user_row.data:
                        if user_row.data.get("is_banned"):
                            await websocket.send_json({"type": "error", "message": "Account banned for community violations."})
                            continue
                        muted_until = user_row.data.get("muted_until")
                        if muted_until and datetime.fromisoformat(muted_until.replace("Z", "+00:00")) > datetime.utcnow().replace(tzinfo=datetime.utcnow().astimezone().tzinfo):
                            await websocket.send_json({"type": "error", "message": "Account temporarily muted."})
                            continue
                except Exception:
                    pass

            # 2. Automated Supervisor Bot Inspection
            bot_violation = evaluate_supervisor_bot_flag(raw_text) if raw_text else None
            if bot_violation:
                await websocket.send_json({
                    "type": "bot_warning",
                    "text": f"Message blocked by Supervisor Bot: {bot_violation}."
                })
                if supabase and sender_id:
                    try:
                        supabase.table("community_reports").insert({
                            "reporter_id": None,
                            "offender_id": sender_id,
                            "community_id": community_id,
                            "reason": "Supervisor Bot Auto-Flag",
                            "details": f"Flagged Content: '{raw_text}' | Issue: {bot_violation}",
                            "status": "pending"
                        }).execute()
                    except Exception:
                        pass
                continue

            # 3. Handle Message Deletion
            if data.get("type") == "delete_message":
                msg_id = data.get("message_id")
                if supabase and msg_id:
                    try:
                        supabase.table("messages").update({"is_deleted": True}).eq("id", msg_id).execute()
                    except Exception:
                        pass
                await manager.broadcast(community_id, {
                    "type": "delete_message",
                    "message_id": msg_id
                })
                continue

            # 4. Standard Message Insertion & Broadcast
            if supabase and raw_text:
                try:
                    payload_to_insert = {
                        "community_id": community_id,
                        "sender_id": sender_id,
                        "text": raw_text,
                        "type": data.get("type", "text"),
                        "gem_id": data.get("gem_id"),
                    }
                    ins_res = supabase.table("messages").insert(payload_to_insert).execute()
                    if ins_res.data and len(ins_res.data) > 0:
                        data["id"] = ins_res.data[0].get("id")
                        data["created_at"] = ins_res.data[0].get("created_at")
                except Exception as e:
                    print(f"[Supabase WS Insert Notice]: {e}")

            # Ensure sender metadata is preserved for live peers
            data["sender_name"] = sender_name
            data["sender_avatar_url"] = sender_avatar_url

            await manager.broadcast(community_id, data)
    except WebSocketDisconnect:
        manager.disconnect(community_id, websocket)
        await manager.broadcast(community_id, {"type": "system", "text": "A scout disconnected."})

@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni TouristOS & Unified Intelligence Cloud",
        "version": "93.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "supabase_connected": bool(supabase)
    }