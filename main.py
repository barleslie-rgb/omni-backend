import os
import asyncio
import io
import gc
import json
import re
import time
import uuid
import base64
import zipfile
import mimetypes
import subprocess
import tempfile
import urllib.parse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, List, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import httpx
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from places import router as places_router
from services.destination_engine import router as destination_router
from gem_scout import GemScout
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

try:
    import fitz
except ImportError:
    fitz = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

app = FastAPI(
    title="Omni TouristOS & Unified Intelligence Cloud",
    description="Universal Travel AI, Street Lens Vision, Dual Voice, Bargain Pal, Universal Document Auditor, Transit Cloud & Community Intelligence",
    version="93.0.0"
)

# Register the places router on the active app instance
app.include_router(places_router)
# Destination Engine v2 is mounted before legacy destination routes so the
# new local-catalog-first architecture owns the stable public API contract.
app.include_router(destination_router)

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
# RAILRADAR API CONFIGURATION
# -------------------------------------------------------------
RAILRADAR_API_KEY = os.environ.get("RAILRADAR_API_KEY", "").strip().strip('"').strip("'")
RAILRADAR_BASE_URL = os.environ.get(
    "RAILRADAR_BASE_URL",
    "https://api.railradar.in/v1"
).strip().rstrip("/")

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

async def ask_fast_vision(image_bytes: bytes, filename: str, target_language: str) -> str:
    """High-quality multimodal document extraction for images.

    Llama 4 Scout was deprecated by Groq in July 2026, so Paper Pilot now
    uses the current Qwen 3.8 multimodal model. The vision pass is deliberately
    extraction-heavy: the following text auditor receives the visible evidence
    instead of a vague image caption.
    """
    client = get_groq_client()
    if not client:
        return ""
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "You are Paper Pilot's visual document intelligence engine. "
            "Analyze the uploaded image as if a user handed the document to an expert human analyst.\n\n"
            "FIRST, inspect the entire image carefully. Do not say that OCR failed if the image is readable. "
            "Read visible text in every language/script you can identify, including headings, names, dates, "
            "phone numbers, addresses, prices, amounts, times, contact details, labels, tables and small print. "
            "Also describe important visual facts such as logos, photographs, seals, signatures, stamps, "
            "checkboxes, highlighted items, layout, document type, and obvious inconsistencies.\n\n"
            "RETURN A FACTUAL EVIDENCE REPORT with these headings:\n"
            "DOCUMENT TYPE / PURPOSE\n"
            "VISIBLE TEXT (transcribe faithfully; preserve important original wording)\n"
            "PEOPLE / ORGANIZATIONS / PLACES\n"
            "DATES / TIMES / NUMBERS\n"
            "FINANCIAL OR OTHER NUMERICAL DETAILS\n"
            "IMPORTANT VISUAL ELEMENTS\n"
            "POTENTIAL ISSUES OR ITEMS TO VERIFY\n"
            "CONFIDENCE / UNREADABLE AREAS\n\n"
            "Rules: Never invent missing text. If something is uncertain, mark it as uncertain. "
            "Do not replace readable content with a generic failure message. "
            f"Write the evidence report in {target_language}."
        )
        completion = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role":"system","content":"You are an expert multimodal document/OCR analyst. Accuracy is more important than brevity."},
                {"role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encoded}"}}
                ]}
            ],
            temperature=0.1,
            max_completion_tokens=5000,
            timeout=90,
        )
        return sanitize_ai_output(completion.choices[0].message.content or "")
    except Exception as exc:
        print(f"[PaperPilot vision notice]: {exc}")
        return ""

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
# 9. TRAVEL DATA PLATFORM: CURRENCY + FLIGHTS + DESTINATIONS + STAYS
# -------------------------------------------------------------
# Production travel integrations deliberately fail closed.  When a provider
# is unavailable or credentials are missing, this backend returns an explicit
# unavailable state instead of inventing flights, hotels, prices, coordinates,
# ratings, timings, or booking URLs.

# Travelpayouts / Aviasales Data API configuration.
# Keep the token on the server (Render environment), never in Flutter or source control.
TRAVELPAYOUTS_API_TOKEN = os.environ.get("TRAVELPAYOUTS_API_TOKEN", "").strip().strip('"').strip("'")
TRAVELPAYOUTS_MARKER = os.environ.get("TRAVELPAYOUTS_MARKER", "774359").strip()
TRAVELPAYOUTS_AVIASALES_BASE_URL = os.environ.get(
    "TRAVELPAYOUTS_AVIASALES_BASE_URL",
    "https://api.travelpayouts.com/aviasales/v3",
).rstrip("/")
TRAVELPAYOUTS_TIMEOUT_SECONDS = max(5.0, min(float(os.environ.get("TRAVELPAYOUTS_TIMEOUT_SECONDS", "25")), 60.0))


BOOKING_BASE_URL = os.environ.get("BOOKING_BASE_URL", "https://demandapi.booking.com/3.2").rstrip("/")
BOOKING_API_KEY = os.environ.get("BOOKING_API_KEY", "").strip()
BOOKING_AFFILIATE_ID = os.environ.get("BOOKING_AFFILIATE_ID", "").strip()
BOOKING_RADIUS_KM = max(1, min(float(os.environ.get("BOOKING_RADIUS_KM", "25")), 100))

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
GOOGLE_PLACES_BASE_URL = "https://places.googleapis.com/v1"

FX_BASE_URL = os.environ.get("FX_BASE_URL", "https://api.frankfurter.dev/v2").rstrip("/")
FX_CACHE_TTL_SECONDS = max(60, min(int(os.environ.get("FX_CACHE_TTL_SECONDS", "1800")), 86400))

# ISO-3166 country -> ISO-4217 currency.  This is only a deterministic
# destination/home-currency resolver; actual exchange rates are fetched live.
COUNTRY_TO_CURRENCY = {
    "ad": "EUR", "ae": "AED", "af": "AFN", "ag": "XCD", "ai": "XCD", "al": "ALL",
    "am": "AMD", "ao": "AOA", "ar": "ARS", "as": "USD", "at": "EUR", "au": "AUD",
    "aw": "AWG", "ax": "EUR", "az": "AZN", "ba": "BAM", "bb": "BBD", "bd": "BDT",
    "be": "EUR", "bf": "XOF", "bg": "BGN", "bh": "BHD", "bi": "BIF", "bj": "XOF",
    "bl": "EUR", "bm": "BMD", "bn": "BND", "bo": "BOB", "bq": "USD", "br": "BRL",
    "bs": "BSD", "bt": "BTN", "bv": "NOK", "bw": "BWP", "by": "BYN", "bz": "BZD",
    "ca": "CAD", "cc": "AUD", "cd": "CDF", "cf": "XAF", "cg": "XAF", "ch": "CHF",
    "ci": "XOF", "ck": "NZD", "cl": "CLP", "cm": "XAF", "cn": "CNY", "co": "COP",
    "cr": "CRC", "cu": "CUP", "cv": "CVE", "cw": "ANG", "cx": "AUD", "cy": "EUR",
    "cz": "CZK", "de": "EUR", "dj": "DJF", "dk": "DKK", "dm": "XCD", "do": "DOP",
    "dz": "DZD", "ec": "USD", "ee": "EUR", "eg": "EGP", "eh": "MAD", "er": "ERN",
    "es": "EUR", "et": "ETB", "fi": "EUR", "fj": "FJD", "fk": "FKP", "fm": "USD",
    "fo": "DKK", "fr": "EUR", "ga": "XAF", "gb": "GBP", "gd": "XCD", "ge": "GEL",
    "gf": "EUR", "gg": "GBP", "gh": "GHS", "gi": "GIP", "gl": "DKK", "gm": "GMD",
    "gn": "GNF", "gp": "EUR", "gq": "XAF", "gr": "EUR", "gs": "GBP", "gt": "GTQ",
    "gu": "USD", "gw": "XOF", "gy": "GYD", "hk": "HKD", "hm": "AUD", "hn": "HNL",
    "hr": "EUR", "ht": "HTG", "hu": "HUF", "id": "IDR", "ie": "EUR", "il": "ILS",
    "im": "GBP", "in": "INR", "io": "USD", "iq": "IQD", "ir": "IRR", "is": "ISK",
    "it": "EUR", "je": "GBP", "jm": "JMD", "jo": "JOD", "jp": "JPY", "ke": "KES",
    "kg": "KGS", "kh": "KHR", "ki": "AUD", "km": "KMF", "kn": "XCD", "kp": "KPW",
    "kr": "KRW", "kw": "KWD", "ky": "KYD", "kz": "KZT", "la": "LAK", "lb": "LBP",
    "lc": "XCD", "li": "CHF", "lk": "LKR", "lr": "LRD", "ls": "LSL", "lt": "EUR",
    "lu": "EUR", "lv": "EUR", "ly": "LYD", "ma": "MAD", "mc": "EUR", "md": "MDL",
    "me": "EUR", "mf": "EUR", "mg": "MGA", "mh": "USD", "mk": "MKD", "ml": "XOF",
    "mm": "MMK", "mn": "MNT", "mo": "MOP", "mp": "USD", "mq": "EUR", "mr": "MRU",
    "ms": "XCD", "mt": "EUR", "mu": "MUR", "mv": "MVR", "mw": "MWK", "mx": "MXN",
    "my": "MYR", "mz": "MZN", "na": "NAD", "nc": "XPF", "ne": "XOF", "nf": "AUD",
    "ng": "NGN", "ni": "NIO", "nl": "EUR", "no": "NOK", "np": "NPR", "nr": "AUD",
    "nu": "NZD", "nz": "NZD", "om": "OMR", "pa": "PAB", "pe": "PEN", "pf": "XPF",
    "pg": "PGK", "ph": "PHP", "pk": "PKR", "pl": "PLN", "pm": "EUR", "pn": "NZD",
    "pr": "USD", "ps": "ILS", "pt": "EUR", "pw": "USD", "py": "PYG", "qa": "QAR",
    "re": "EUR", "ro": "RON", "rs": "RSD", "ru": "RUB", "rw": "RWF", "sa": "SAR",
    "sb": "SBD", "sc": "SCR", "sd": "SDG", "se": "SEK", "sg": "SGD", "sh": "SHP",
    "si": "EUR", "sj": "NOK", "sk": "EUR", "sl": "SLE", "sm": "EUR", "sn": "XOF",
    "so": "SOS", "sr": "SRD", "ss": "SSP", "st": "STN", "sv": "USD", "sx": "ANG",
    "sy": "SYP", "sz": "SZL", "tc": "USD", "td": "XAF", "tf": "EUR", "tg": "XOF",
    "th": "THB", "tj": "TJS", "tk": "NZD", "tl": "USD", "tm": "TMT", "tn": "TND",
    "to": "TOP", "tr": "TRY", "tt": "TTD", "tv": "AUD", "tw": "TWD", "tz": "TZS",
    "ua": "UAH", "ug": "UGX", "um": "USD", "us": "USD", "uy": "UYU", "uz": "UZS",
    "va": "EUR", "vc": "XCD", "ve": "VES", "vg": "USD", "vi": "USD", "vn": "VND",
    "vu": "VUV", "wf": "XPF", "ws": "WST", "ye": "YER", "yt": "EUR", "za": "ZAR",
    "zm": "ZMW", "zw": "ZWG"
}

COUNTRY_NAME_TO_ISO2 = {
    "india": "in", "united states": "us", "usa": "us", "united states of america": "us",
    "united kingdom": "gb", "uk": "gb", "england": "gb", "uae": "ae",
    "united arab emirates": "ae", "singapore": "sg", "japan": "jp", "thailand": "th",
    "france": "fr", "germany": "de", "italy": "it", "spain": "es", "switzerland": "ch",
    "australia": "au", "canada": "ca", "new zealand": "nz", "china": "cn", "hong kong": "hk",
    "south korea": "kr", "korea": "kr", "indonesia": "id", "malaysia": "my", "philippines": "ph",
    "vietnam": "vn", "nepal": "np", "sri lanka": "lk", "bangladesh": "bd", "pakistan": "pk",
    "saudi arabia": "sa", "qatar": "qa", "bahrain": "bh", "oman": "om", "kuwait": "kw",
    "turkey": "tr", "türkiye": "tr", "netherlands": "nl", "belgium": "be", "austria": "at",
    "portugal": "pt", "ireland": "ie", "greece": "gr", "czech republic": "cz", "czechia": "cz",
    "poland": "pl", "hungary": "hu", "sweden": "se", "norway": "no", "denmark": "dk",
    "finland": "fi", "iceland": "is", "mexico": "mx", "brazil": "br", "argentina": "ar",
    "chile": "cl", "colombia": "co", "peru": "pe", "south africa": "za", "egypt": "eg",
    "morocco": "ma", "kenya": "ke", "tanzania": "tz", "nigeria": "ng", "ghana": "gh",
    "israel": "il", "russia": "ru", "ukraine": "ua"
}

CURRENCY_SYMBOLS = {
    "AED": "د.إ", "AUD": "A$", "BRL": "R$", "CAD": "C$", "CHF": "CHF", "CNY": "¥",
    "CZK": "Kč", "DKK": "kr", "EUR": "€", "GBP": "£", "HKD": "HK$", "HUF": "Ft",
    "IDR": "Rp", "INR": "₹", "ILS": "₪", "JPY": "¥", "KRW": "₩", "KWD": "د.ك",
    "MAD": "MAD", "MXN": "MX$", "MYR": "RM", "NOK": "kr", "NZD": "NZ$", "PHP": "₱",
    "PLN": "zł", "QAR": "ر.ق", "RON": "lei", "RUB": "₽", "SAR": "﷼", "SEK": "kr",
    "SGD": "S$", "THB": "฿", "TRY": "₺", "TWD": "NT$", "USD": "$", "VND": "₫", "ZAR": "R"
}

_fx_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_google_destination_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_open_destination_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_open_places_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
_wikimedia_image_cache: Dict[str, Dict[str, Any]] = {}
_wikimedia_semaphore = asyncio.Semaphore(3)
_open_geo_semaphore = asyncio.Semaphore(1)
_overpass_semaphore = asyncio.Semaphore(1)
_google_places_disabled = False
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_USER_AGENT = "OmniTouristOS/1.0 (destination image service)"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
OPEN_DATA_USER_AGENT = "OmniTouristOS/1.0 (destination explorer; contact via app)"
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"


def _normalize_country_iso2(country: str) -> str:
    raw = str(country or "").strip().lower()
    if len(raw) == 2 and raw.isalpha():
        return raw
    return COUNTRY_NAME_TO_ISO2.get(raw, "in")


def _currency_for_country(country: str) -> str:
    return COUNTRY_TO_CURRENCY.get(_normalize_country_iso2(country), "USD")


def _currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get(str(code or "USD").upper(), str(code or "USD").upper())


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        if value is None or value == "":
            return default
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _format_decimal(value: Decimal) -> str:
    normalized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(normalized, "f")


def _parse_iso_duration(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.startswith("P"):
        return text
    match = re.fullmatch(r"P(?:([0-9]+)D)?(?:T(?:([0-9]+)H)?(?:([0-9]+)M)?(?:([0-9]+)S)?)?", text)
    if not match:
        return text
    days, hours, minutes, seconds = match.groups()
    parts = []
    if days and int(days):
        parts.append(f"{int(days)}d")
    if hours and int(hours):
        parts.append(f"{int(hours)}h")
    if minutes and int(minutes):
        parts.append(f"{int(minutes)}m")
    if seconds and int(seconds) and not parts:
        parts.append(f"{int(seconds)}s")
    return " ".join(parts) or "0m"


def _format_clock(iso_value: Any) -> str:
    text = str(iso_value or "")
    if "T" in text:
        text = text.split("T", 1)[1]
    text = text.split("+", 1)[0].split("Z", 1)[0]
    if len(text) >= 5:
        return text[:5]
    return text


def _format_date(iso_value: Any) -> str:
    text = str(iso_value or "")
    return text[:10] if len(text) >= 10 else text

def _add_minutes_iso(iso_value: Any, minutes: Any) -> Optional[str]:
    text = str(iso_value or "").strip()
    try:
        mins = int(minutes or 0)
    except (TypeError, ValueError):
        return None
    if not text or mins <= 0:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return (dt + timedelta(minutes=mins)).isoformat()
    except Exception:
        return None


async def _fx_rate(base: str, quote: str) -> Optional[Dict[str, Any]]:
    base = str(base or "").upper()
    quote = str(quote or "").upper()
    if not base or not quote:
        return None
    if base == quote:
        return {"base": base, "quote": quote, "rate": Decimal("1"), "date": None, "provider": "identity"}

    key = (base, quote)
    cached = _fx_cache.get(key)
    if cached and (time.time() - cached.get("timestamp", 0) < FX_CACHE_TTL_SECONDS):
        return cached.get("data")

    url = f"{FX_BASE_URL}/rate/{base.lower()}/{quote.lower()}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
            response = await client.get(url, headers={"Accept": "application/json"})
        if response.status_code != 200:
            print(f"[FX Notice] {response.status_code}: {response.text[:200]}")
            return None
        data = response.json()
        rate = _safe_decimal(data.get("rate"), Decimal("0"))
        if rate <= 0:
            return None
        result = {
            "base": base,
            "quote": quote,
            "rate": rate,
            "date": data.get("date"),
            "provider": "Frankfurter"
        }
        _fx_cache[key] = {"timestamp": time.time(), "data": result}
        return result
    except Exception as e:
        print(f"[FX Notice]: {e}")
        return None


async def _convert_amount(value: Any, base: str, quote: str) -> Optional[Decimal]:
    amount = _safe_decimal(value)
    rate_data = await _fx_rate(base, quote)
    if not rate_data:
        return None
    return amount * rate_data["rate"]


def _travelpayouts_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "X-Access-Token": TRAVELPAYOUTS_API_TOKEN,
        "User-Agent": "Omni-TouristOS/1.0",
    }


def _travelpayouts_cabin_class(value: str) -> str:
    raw = str(value or "Economy").strip().lower()
    mapping = {
        "economy": "economy",
        "premium economy": "premium_economy",
        "premium_economy": "premium_economy",
        "business": "business",
        "first": "first",
    }
    return mapping.get(raw, "economy")


def _minutes_to_duration(minutes: Any) -> str:
    try:
        total = max(0, int(minutes))
    except (TypeError, ValueError):
        return ""
    hours, mins = divmod(total, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def _encode_travelpayouts_offer(item: Dict[str, Any]) -> str:
    raw = json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_travelpayouts_offer(value: str) -> Dict[str, Any]:
    padding = "=" * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
    data = json.loads(decoded.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _booking_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {BOOKING_API_KEY}",
        "X-Affiliate-Id": BOOKING_AFFILIATE_ID,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _wikimedia_image_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Find openly hosted destination images on Wikimedia Commons.

    This is deliberately independent of Google Places billing/quota. Commons
    exposes public MediaWiki APIs and the returned image URLs are direct HTTPS
    URLs suitable for Flutter Image.network().
    """
    clean_query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not clean_query:
        return {"images": [], "credits": []}

    cache_key = clean_query.lower()
    cached = _wikimedia_image_cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": clean_query,
        "gsrnamespace": "6",
        "gsrlimit": str(max(1, min(int(limit), 6))),
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "1200",
        "origin": "*",
    }

    result: Dict[str, Any] = {"images": [], "credits": []}
    try:
        async with _wikimedia_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
                response = await client.get(
                    WIKIMEDIA_API_URL,
                    params=params,
                    headers={"User-Agent": WIKIMEDIA_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Wikimedia Image Notice] {response.status_code}: {response.text[:300]}")
            _wikimedia_image_cache[cache_key] = result
            return result

        payload = response.json() or {}
        pages = ((payload.get("query") or {}).get("pages") or {})
        seen: set = set()
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0] or {}
            mime = str(info.get("mime") or "").lower()
            image_url = str(info.get("thumburl") or info.get("url") or "").strip()
            if not image_url.startswith(("https://", "http://")):
                continue
            if mime and not mime.startswith("image/"):
                continue
            if image_url in seen:
                continue
            seen.add(image_url)

            metadata = info.get("extmetadata") or {}
            artist = str((metadata.get("Artist") or {}).get("value") or "").strip()
            license_name = str((metadata.get("LicenseShortName") or {}).get("value") or "").strip()
            page_title = str(page.get("title") or "").strip()
            page_id = page.get("pageid")
            source_url = (
                f"https://commons.wikimedia.org/wiki/Special:Redirect/file/"
                f"{urllib.parse.quote(page_title.removeprefix('File:'), safe='') }"
            ) if page_title else "https://commons.wikimedia.org/"

            result["images"].append(image_url)
            result["credits"].append({
                "title": page_title,
                "source_url": source_url,
                "artist": re.sub(r"<[^>]+>", "", artist),
                "license": re.sub(r"<[^>]+>", "", license_name),
                "page_id": page_id,
            })
            if len(result["images"]) >= int(limit):
                break
    except Exception as e:
        print(f"[Wikimedia Image Notice]: {e}")

    _wikimedia_image_cache[cache_key] = result
    return result


async def _attach_wikimedia_images(
    places: List[Dict[str, Any]],
    city: str,
    country: str,
    limit_places: int = 15,
) -> None:
    """Attach open destination photos without changing provider place facts."""
    targets = places[:max(0, int(limit_places))]
    tasks = []
    for item in targets:
        name = str(item.get("name") or "").strip()
        if not name:
            tasks.append(asyncio.sleep(0, result={"images": [], "credits": []}))
            continue
        query = f"{name} {city} {country}".strip()
        tasks.append(_wikimedia_image_search(query, 5))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(targets, results):
        if isinstance(result, Exception) or not isinstance(result, dict):
            continue
        images = result.get("images") or []
        credits = result.get("credits") or []
        if images:
            item["images"] = images
            item["image_provider"] = "Wikimedia Commons"
            item["image_credits"] = credits
            item["google_photo_names"] = []
        else:
            item["images"] = []
            item["image_provider"] = "NONE"
            item["image_credits"] = []
            item["google_photo_names"] = []


async def _google_text_search(query: str, max_result_count: int = 10, latitude: Optional[float] = None, longitude: Optional[float] = None, radius_meters: float = 15000) -> List[Dict[str, Any]]:
    # Never permanently disable Google after one provider error.
    if not GOOGLE_PLACES_API_KEY:
        print('[Google Places Notice] GOOGLE_PLACES_API_KEY is not configured on the backend.')
        return []

    endpoint = f"{GOOGLE_PLACES_BASE_URL}/places:searchText"
    field_mask = ",".join([
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.googleMapsUri",
        "places.websiteUri",
        "places.primaryType",
        "places.primaryTypeDisplayName",
        "places.types",
        "places.addressComponents",
        "places.rating",
        "places.userRatingCount",
        "places.regularOpeningHours",
        "places.photos",
        "places.nationalPhoneNumber",
        "places.internationalPhoneNumber",
    ])
    payload = {
        "textQuery": query,
        "pageSize": max(1, min(int(max_result_count), 20)),
        "languageCode": "en",
    }
    if latitude is not None and longitude is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": float(latitude), "longitude": float(longitude)},
                "radius": max(500.0, min(float(radius_meters), 50000.0)),
            }
        }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0)) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                    "X-Goog-FieldMask": field_mask,
                },
                json=payload,
            )
        if response.status_code != 200:
            detail = response.text[:800]
            print(f"[Google Places Notice] HTTP {response.status_code}: {detail}")
            return []
        data = response.json()
        return data.get("places", []) or []
    except Exception as e:
        print(f"[Google Places Notice]: {e}")
        return []


async def _resolve_destination_with_google(city: str, state: str, country: str) -> Optional[Dict[str, Any]]:
    key = (city.strip().lower(), state.strip().lower(), country.strip().lower())
    if key in _google_destination_cache:
        return _google_destination_cache[key]
    if not GOOGLE_PLACES_API_KEY:
        return None

    query_parts = [city.strip()]
    if state.strip():
        query_parts.append(state.strip())
    if country.strip():
        query_parts.append(country.strip())

    results = await _google_text_search(", ".join(query_parts), max_result_count=8)
    if not results:
        return None

    def candidate_score(place: Dict[str, Any]) -> int:
        types = {str(x).lower() for x in (place.get("types") or [])}
        primary = str(place.get("primaryType") or "").lower()
        display = str((place.get("displayName") or {}).get("text") or "").strip().lower()
        score = 0
        if "locality" in types or "postal_town" in types:
            score += 100
        if "administrative_area_level_2" in types:
            score += 80
        if "administrative_area_level_1" in types:
            score += 40
        if primary in {"locality", "administrative_area_level_2", "administrative_area_level_1"}:
            score += 30
        if display == city.strip().lower():
            score += 25
        elif city.strip().lower() in display:
            score += 10
        return score

    chosen = max(results, key=candidate_score)
    loc = chosen.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    if lat is None or lng is None:
        return None

    display_name = (chosen.get("displayName") or {}).get("text") or city
    resolved = {
        "city": city,
        "state": state,
        "country": country,
        "display_name": display_name,
        "place_id": chosen.get("id"),
        "latitude": float(lat),
        "longitude": float(lng),
        "formatted_address": chosen.get("formattedAddress"),
        "google_maps_url": chosen.get("googleMapsUri"),
        "website_url": chosen.get("websiteUri"),
        "types": chosen.get("types") or [],
        "address_components": chosen.get("addressComponents") or [],
        "source": "Google Places",
        "data_state": "VERIFIED",
    }
    _google_destination_cache[key] = resolved
    return resolved


async def _resolve_destination_with_open_data(city: str, state: str, country: str) -> Optional[Dict[str, Any]]:
    """Resolve a destination without Google using OpenStreetMap Nominatim.

    This is a real geocoder lookup, not generated/synthetic destination data.
    Requests are serialized to respect the public Nominatim service policy.
    """
    key = (city.strip().lower(), state.strip().lower(), country.strip().lower())
    cached = _open_destination_cache.get(key)
    if cached is not None:
        return cached

    query_parts = [city.strip()]
    if state.strip():
        query_parts.append(state.strip())
    if country.strip():
        query_parts.append(country.strip())

    params = {
        "q": ", ".join(x for x in query_parts if x),
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": "5",
        "accept-language": "en",
    }
    try:
        async with _open_geo_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0)) as client:
                response = await client.get(
                    NOMINATIM_API_URL,
                    params=params,
                    headers={"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Open Geocoder Notice] {response.status_code}: {response.text[:300]}")
            return None
        results = response.json() or []
        if not results:
            return None

        city_norm = re.sub(r"\\s+", " ", city.strip().lower())
        country_norm = country.strip().lower()

        def score(item: Dict[str, Any]) -> int:
            display = str(item.get("display_name") or "").lower()
            item_name = str(item.get("name") or "").lower()
            item_type = str(item.get("type") or "").lower()
            address = item.get("address") or {}
            score_value = 0
            if item_name == city_norm:
                score_value += 100
            elif city_norm and city_norm in display:
                score_value += 40
            if item_type in {"city", "town", "municipality", "village", "locality"}:
                score_value += 40
            if country_norm and country_norm in display:
                score_value += 10
            if address.get("city", "").lower() == city_norm:
                score_value += 25
            return score_value

        chosen = max(results, key=score)
        lat = chosen.get("lat")
        lon = chosen.get("lon")
        if lat is None or lon is None:
            return None

        resolved = {
            "city": city,
            "state": state,
            "country": country,
            "display_name": chosen.get("name") or city,
            "place_id": f"osm:{chosen.get('osm_type','')}/{chosen.get('osm_id','')}",
            "latitude": float(lat),
            "longitude": float(lon),
            "formatted_address": chosen.get("display_name"),
            "google_maps_url": None,
            "website_url": None,
            "types": [str(chosen.get("type") or "")],
            "address_components": chosen.get("address") or {},
            "source": "OpenStreetMap Nominatim",
            "data_state": "VERIFIED_OPEN_DATA",
            "open_data": True,
        }
        _open_destination_cache[key] = resolved
        return resolved
    except Exception as e:
        print(f"[Open Geocoder Notice]: {e}")
        return None


def _open_place_category(title: str, categories: List[str]) -> str:
    text = f"{title} {' '.join(categories)}".lower()
    if any(k in text for k in ["temple", "church", "mosque", "shrine", "fort", "castle", "monument", "historic", "heritage", "palace"]):
        return "Heritage & Forts"
    if any(k in text for k in ["beach", "park", "garden", "waterfall", "lake", "wildlife", "island", "mountain"]):
        return "Nature & Wildlife"
    if any(k in text for k in ["museum", "gallery", "theatre", "theater"]):
        return "Arts & Culture"
    if any(k in text for k in ["zoo", "aquarium", "amusement", "theme park", "tourist"]):
        return "Family & Attractions"
    return "Sights & Landmarks"


async def _load_open_destination_places(city: str, state: str, country: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load real attraction records from OpenStreetMap when Google is unavailable.

    This deliberately avoids Wikipedia/Wikimedia search APIs because those public
    endpoints can reject server-side traffic under their robot policy. Attraction
    facts come from OSM objects themselves; photos are used only when the OSM object
    already carries an image or wikimedia_commons tag.
    """
    key = (city.strip().lower(), state.strip().lower(), country.strip().lower())
    if key in _open_places_cache:
        destination = _open_destination_cache.get(key)
        return destination, _open_places_cache[key]

    destination = await _resolve_destination_with_open_data(city, state, country)
    if not destination:
        return None, []

    lat = float(destination["latitude"])
    lng = float(destination["longitude"])
    radius_m = 20000
    query = f"""
[out:json][timeout:30];
(
  nwr(around:{radius_m},{lat},{lng})["tourism"~"^(attraction|museum|viewpoint|theme_park|zoo|aquarium|gallery|artwork|information)$"]["name"];
  nwr(around:{radius_m},{lat},{lng})["historic"]["name"];
  nwr(around:{radius_m},{lat},{lng})["natural"~"^(beach|waterfall|peak|cave)$"]["name"];
  nwr(around:{radius_m},{lat},{lng})["leisure"~"^(park|garden|nature_reserve)$"]["name"];
);
out center tags;
"""

    try:
        async with _overpass_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(35.0, connect=8.0)) as client:
                response = await client.post(
                    OVERPASS_API_URL,
                    data={"data": query},
                    headers={"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Overpass Notice] {response.status_code}: {response.text[:300]}")
            return destination, []

        elements = (response.json() or {}).get("elements") or []
        normalized: List[Dict[str, Any]] = []
        seen_names: set = set()
        city_norm = city.strip().lower()

        def tag_value(tags: Dict[str, Any], key_name: str) -> str:
            return str(tags.get(key_name) or "").strip()

        def image_from_tags(tags: Dict[str, Any]) -> Optional[str]:
            direct = tag_value(tags, "image")
            if direct.startswith(("https://", "http://")):
                return direct
            commons = tag_value(tags, "wikimedia_commons")
            if commons.startswith("File:"):
                filename = commons[5:].strip()
                if filename:
                    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename, safe="")
            return None

        for element in elements:
            tags = element.get("tags") or {}
            name = tag_value(tags, "name")
            if not name:
                continue
            name_norm = name.lower()
            if name_norm == city_norm or name_norm in seen_names:
                continue

            if element.get("lat") is not None and element.get("lon") is not None:
                item_lat = float(element["lat"])
                item_lng = float(element["lon"])
            else:
                center = element.get("center") or {}
                if center.get("lat") is None or center.get("lon") is None:
                    continue
                item_lat = float(center["lat"])
                item_lng = float(center["lon"])

            distance = _place_distance_km(item_lat, item_lng, lat, lng)
            if distance > 25:
                continue

            tourism = tag_value(tags, "tourism")
            historic = tag_value(tags, "historic")
            natural = tag_value(tags, "natural")
            leisure = tag_value(tags, "leisure")
            categories = [x for x in [tourism, historic, natural, leisure, tag_value(tags, "attraction")] if x]
            category = _open_place_category(name, categories)

            website = tag_value(tags, "website") or tag_value(tags, "contact:website")
            wikidata = tag_value(tags, "wikidata")
            wikipedia = tag_value(tags, "wikipedia")
            source_url = website if website.startswith(("https://", "http://")) else f"https://www.openstreetmap.org/?mlat={item_lat}&mlon={item_lng}#map=17/{item_lat}/{item_lng}"
            image_url = image_from_tags(tags)

            item = {
                "name": name,
                "category": category,
                "distance": f"{distance:.1f} km from Center",
                "distance_km": round(distance, 2),
                "distance_type": "straight_line",
                "timing": tag_value(tags, "opening_hours") or "See official source",
                "entry": "Free / see official source" if tag_value(tags, "fee").lower() in {"no", "0"} else "See official source",
                "lat": item_lat,
                "lng": item_lng,
                "history": "Verified OpenStreetMap feature record.",
                "best_food": "",
                "things_to_do": "",
                "best_time": "",
                "warnings": "Check the official venue/source for current access, hours and local conditions.",
                "rating": None,
                "reviews": None,
                "images": [image_url] if image_url else [],
                "google_photo_names": [],
                "image_provider": "OpenStreetMap-linked image" if image_url else "NONE",
                "image_credits": [{"title": name, "source_url": source_url, "artist": "", "license": "See source page"}] if image_url else [],
                "maps_url": f"https://www.openstreetmap.org/?mlat={item_lat}&mlon={item_lng}#map=17/{item_lat}/{item_lng}",
                "website_url": source_url,
                "place_id": f"osm:{element.get('type','')}/{element.get('id','')}",
                "source": "OpenStreetMap Open Data",
                "data_state": "VERIFIED_OPEN_DATA",
                "provider_verified": True,
                "attribution_required": "OpenStreetMap" + (" + Wikimedia Commons" if image_url and "commons.wikimedia.org" in image_url else ""),
                "osm_tags": {
                    "tourism": tourism,
                    "historic": historic,
                    "natural": natural,
                    "leisure": leisure,
                    "wikidata": wikidata,
                    "wikipedia": wikipedia,
                },
            }
            normalized.append(item)
            seen_names.add(name_norm)

        normalized.sort(key=lambda x: (-(1 if x.get("images") else 0), float(x.get("distance_km") or 999), str(x.get("name") or "")))
        normalized = normalized[:30]
        _open_places_cache[key] = normalized
        return destination, normalized
    except Exception as e:
        print(f"[Overpass Notice]: {e}")
        return destination, []


def _place_category(place: Dict[str, Any]) -> str:
    primary = str(place.get("primaryType") or "").lower()
    label = str((place.get("primaryTypeDisplayName") or {}).get("text") or "").lower()
    combined = f"{primary} {label}"
    if any(k in combined for k in ["museum", "church", "temple", "mosque", "shrine", "historical", "castle", "monument"]):
        return "Heritage & Forts"
    if any(k in combined for k in ["beach", "park", "garden", "national_park", "nature", "waterfall"]):
        return "Nature & Wildlife"
    if any(k in combined for k in ["restaurant", "cafe", "market", "food", "bakery"]):
        return "Culinary & Bazaars"
    if any(k in combined for k in ["zoo", "aquarium", "amusement", "tourist"]):
        return "Family & Attractions"
    return "Sights & Landmarks"


def _place_distance_km(lat: float, lng: float, origin_lat: float, origin_lng: float) -> float:
    from math import asin, cos, radians, sin, sqrt
    earth_radius_km = 6371.0088
    lat1, lon1, lat2, lon2 = map(radians, [origin_lat, origin_lng, lat, lng])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return earth_radius_km * 2 * asin(min(1.0, sqrt(a)))


def _compact_hours(place: Dict[str, Any]) -> Optional[str]:
    hours = place.get("regularOpeningHours") or {}
    descriptions = hours.get("weekdayDescriptions") or []
    if not descriptions:
        return None
    return " • ".join(str(x) for x in descriptions[:2])



async def _google_place_details(place_id: str) -> Optional[Dict[str, Any]]:
    if not GOOGLE_PLACES_API_KEY or not place_id:
        return None
    clean_id = str(place_id).strip()
    if not clean_id:
        return None
    field_mask = ",".join([
        "id",
        "displayName",
        "formattedAddress",
        "location",
        "googleMapsUri",
        "websiteUri",
        "primaryType",
        "primaryTypeDisplayName",
        "types",
        "rating",
        "userRatingCount",
        "regularOpeningHours",
        "currentOpeningHours",
        "photos",
        "editorialSummary",
        "internationalPhoneNumber",
        "nationalPhoneNumber",
        "priceLevel",
        "parkingOptions",
        "paymentOptions",
        "accessibilityOptions",
        "servesVegetarianFood",
        "outdoorSeating",
        "reservable",
    ])
    resource = clean_id if clean_id.startswith("places/") else f"places/{clean_id}"
    endpoint = f"{GOOGLE_PLACES_BASE_URL}/{urllib.parse.quote(resource, safe='/')}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
            response = await client.get(
                endpoint,
                headers={
                    "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                    "X-Goog-FieldMask": field_mask,
                },
            )
        if response.status_code != 200:
            print(f"[Google Place Details Notice] {response.status_code}: {response.text[:300]}")
            return None
        return response.json()
    except Exception as e:
        print(f"[Google Place Details Notice]: {e}")
        return None


def _google_photo_proxy_urls(photo_names: List[str], max_width: int = 1200) -> List[str]:
    urls: List[str] = []
    for raw_name in photo_names[:10]:
        name = str(raw_name or "").strip()
        if not name:
            continue
        urls.append(
            f"/api/v1/place-photo?name={urllib.parse.quote(name, safe='')}&max_width={int(max_width)}"
        )
    return urls


@app.get("/api/v1/place-photo")
async def google_place_photo(
    name: str = Query(...),
    max_width: int = Query(1200, ge=320, le=4800),
):
    """
    Proxy a Google Places (New) photo to the Flutter client.

    Google can return the actual image via redirect or, with
    skipHttpRedirect=true, a short-lived photoUri. We support both paths so
    the client never needs the Google API key.
    """
    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(status_code=503, detail="Google Places photo service is not configured.")

    clean_name = str(name or "").strip()
    if not clean_name.startswith("places/") or "/photos/" not in clean_name:
        raise HTTPException(status_code=400, detail="Invalid Google photo resource name.")

    encoded_name = urllib.parse.quote(clean_name, safe="/")
    endpoint = f"{GOOGLE_PLACES_BASE_URL}/{encoded_name}/media"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=True,
        ) as client:
            # First request asks Google for a stable photoUri. This avoids
            # depending on redirect handling in every HTTP client/device.
            metadata_response = await client.get(
                endpoint,
                params={
                    "maxWidthPx": int(max_width),
                    "skipHttpRedirect": "true",
                    "key": GOOGLE_PLACES_API_KEY,
                },
                headers={"Accept": "application/json"},
            )

            photo_uri = ""
            if metadata_response.status_code == 200:
                try:
                    metadata = metadata_response.json()
                    photo_uri = str(metadata.get("photoUri") or "").strip()
                except Exception:
                    photo_uri = ""

            if photo_uri:
                image_response = await client.get(photo_uri)
            else:
                # Fallback to Google's normal image redirect response.
                image_response = await client.get(
                    endpoint,
                    params={
                        "maxWidthPx": int(max_width),
                        "key": GOOGLE_PLACES_API_KEY,
                    },
                )

        if image_response.status_code != 200:
            detail = image_response.text[:300] if image_response.content else "empty response"
            print(f"[Google Photo Notice] {image_response.status_code}: {detail}")
            raise HTTPException(status_code=502, detail="Google photo could not be retrieved.")

        media_type = image_response.headers.get("content-type", "image/jpeg").split(";")[0]
        if not media_type.startswith("image/"):
            raise HTTPException(status_code=502, detail="Google returned an invalid photo response.")

        return Response(
            content=image_response.content,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=900",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Google Photo Notice]: {e}")
        raise HTTPException(status_code=502, detail="Google photo service temporarily unavailable.")


@app.get("/api/v1/place-photo-by-id")
async def google_place_photo_by_id(
    place_id: str = Query(...),
    max_width: int = Query(1200, ge=320, le=4800),
):
    """Resolve the first current Google Places photo for a place ID and proxy it.

    This is a second-level fallback for clients that receive a verified place ID
    but do not receive photo resource names in the original explore response.
    The Google API key remains server-side.
    """
    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(status_code=503, detail="Google Places photo service is not configured.")

    clean_id = str(place_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=400, detail="Place ID is required.")

    place = await _google_place_details(clean_id)
    if not place:
        raise HTTPException(status_code=404, detail="Google place could not be resolved.")

    photos = place.get("photos") or []
    photo_names = [str(photo.get("name") or "").strip() for photo in photos if photo.get("name")]
    if not photo_names:
        raise HTTPException(status_code=404, detail="No Google photo is available for this place.")

    photo_name = photo_names[0]
    encoded_name = urllib.parse.quote(photo_name, safe="/")
    endpoint = f"{GOOGLE_PLACES_BASE_URL}/{encoded_name}/media"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=True,
        ) as client:
            metadata_response = await client.get(
                endpoint,
                params={
                    "maxWidthPx": int(max_width),
                    "skipHttpRedirect": "true",
                    "key": GOOGLE_PLACES_API_KEY,
                },
                headers={"Accept": "application/json"},
            )
            photo_uri = ""
            if metadata_response.status_code == 200:
                try:
                    metadata = metadata_response.json()
                    photo_uri = str(metadata.get("photoUri") or "").strip()
                except Exception:
                    photo_uri = ""

            if photo_uri:
                image_response = await client.get(photo_uri)
            else:
                image_response = await client.get(
                    endpoint,
                    params={"maxWidthPx": int(max_width), "key": GOOGLE_PLACES_API_KEY},
                )

        if image_response.status_code != 200:
            detail = image_response.text[:300] if image_response.content else "empty response"
            print(f"[Google Photo By ID Notice] {image_response.status_code}: {detail}")
            raise HTTPException(status_code=502, detail="Google photo could not be retrieved.")

        media_type = image_response.headers.get("content-type", "image/jpeg").split(";")[0]
        if not media_type.startswith("image/"):
            raise HTTPException(status_code=502, detail="Google returned an invalid photo response.")

        return Response(
            content=image_response.content,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=900",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Google Photo By ID Notice]: {e}")
        raise HTTPException(status_code=502, detail="Google photo service temporarily unavailable.")


def _normalize_google_place(place: Dict[str, Any], origin_lat: float, origin_lng: float) -> Optional[Dict[str, Any]]:
    loc = place.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    name = (place.get("displayName") or {}).get("text")
    if lat is None or lng is None or not name:
        return None

    distance = _place_distance_km(float(lat), float(lng), origin_lat, origin_lng)
    photos = place.get("photos") or []
    photo_names = [str(photo.get("name")) for photo in photos if photo.get("name")]
    rating = place.get("rating")
    rating_count = place.get("userRatingCount")
    hours = _compact_hours(place)

    result = {
        "name": name,
        "category": _place_category(place),
        "distance": f"{distance:.1f} km from Center",
        "distance_km": round(distance, 2),
        "distance_type": "straight_line",
        "timing": hours or "",
        "entry": "See provider",
        "lat": float(lat),
        "lng": float(lng),
        "history": str((place.get("primaryTypeDisplayName") or {}).get("text") or "Verified place listing"),
        "best_food": "",
        "things_to_do": "",
        "best_time": "",
        "warnings": "",
        "rating": rating,
        "reviews": rating_count,
        "images": [],
        "google_photo_names": [],
        "image_provider": "Wikimedia Commons",
        "image_credits": [],
        "maps_url": place.get("googleMapsUri"),
        "website_url": place.get("websiteUri"),
        "place_id": place.get("id"),
        "source": "Google Places",
        "data_state": "VERIFIED",
        "provider_verified": True,
        "attribution_required": "Google Maps",
    }
    return result



def _normalize_google_hotel(place: Dict[str, Any], center_lat: float, center_lng: float, city: str) -> Optional[Dict[str, Any]]:
    loc = place.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    name = str((place.get("displayName") or {}).get("text") or "").strip()
    if lat is None or lng is None or not name:
        return None
    distance = _place_distance_km(float(lat), float(lng), center_lat, center_lng)
    photos = place.get("photos") or []
    photo_names = [str(photo.get("name")) for photo in photos if photo.get("name")]
    return {
        "name": name,
        "tier": "Hotel",
        "rating": place.get("rating"),
        "reviews": place.get("userRatingCount"),
        "price": None,
        "price_total": None,
        "currency": None,
        "booker_currency": None,
        "price_display_currency": None,
        "price_display_symbol": None,
        "phone": place.get("internationalPhoneNumber") or place.get("nationalPhoneNumber"),
        "distance": f"{distance:.1f} km from selected place",
        "distance_km": round(distance, 2),
        "amenities": "Verified hotel/place listing",
        "lat": float(lat),
        "lng": float(lng),
        "images": [],
        "google_photo_names": [],
        "image_provider": "NONE",
        "image_credits": [],
        "address": place.get("formattedAddress") or "",
        "city": city,
        "accommodation_id": place.get("id"),
        "booking_url": None,
        "website_url": place.get("websiteUri"),
        "maps_url": place.get("googleMapsUri"),
        "cancellation_type": None,
        "free_cancellation_until": None,
        "meal_plan": None,
        "payment_timings": [],
        "inventory_type": "Google Places",
        "third_party_inventory": False,
        "charges": None,
        "product_id": None,
        "source": "Google Places",
        "data_state": "VERIFIED",
        "provider_verified": True,
        "live_price": False,
    }


async def _load_google_hotels(query: str, center_lat: float, center_lng: float, city: str) -> List[Dict[str, Any]]:
    batches = await asyncio.gather(
        _google_text_search(f"hotels near {query}", 10),
        _google_text_search(f"best hotels near {query}", 10),
        return_exceptions=True,
    )
    merged: Dict[str, Dict[str, Any]] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        for place in batch:
            pid = str(place.get("id") or "")
            types = {str(x).lower() for x in (place.get("types") or [])}
            label = str((place.get("primaryTypeDisplayName") or {}).get("text") or "").lower()
            if not pid or ("lodging" not in types and "hotel" not in types and "resort" not in label and "hotel" not in label):
                continue
            merged.setdefault(pid, place)
    hotels = []
    for place in merged.values():
        item = _normalize_google_hotel(place, center_lat, center_lng, city)
        if item:
            hotels.append(item)
    hotels.sort(key=lambda x: (-(float(x.get("rating") or 0)), -(int(x.get("reviews") or 0)), float(x.get("distance_km") or 999)))
    return hotels[:10]


async def _load_google_destination_places(city: str, state: str, country: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    destination = await _resolve_destination_with_google(city, state, country)
    if not destination:
        return None, []

    queries = [
        f"top attractions and landmarks in {city}, {country}",
        f"things to do and tourist attractions in {city}, {country}",
    ]
    batches = await asyncio.gather(*[_google_text_search(q, 10) for q in queries], return_exceptions=True)
    merged: Dict[str, Dict[str, Any]] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        for place in batch:
            place_id = str(place.get("id") or "")
            if place_id and place_id not in merged:
                merged[place_id] = place

    # Do not hydrate every card through Google Place Details. That endpoint
    # consumes the daily GetPlaceRequest quota and was the source of the 429
    # RESOURCE_EXHAUSTED errors seen in Render. Destination photos are now
    # supplied independently from Wikimedia Commons.
    place_values = list(merged.values())

    normalized: List[Dict[str, Any]] = []
    for place in place_values:
        item = _normalize_google_place(place, destination["latitude"], destination["longitude"])
        if item:
            normalized.append(item)

    normalized.sort(key=lambda x: (-(float(x.get("rating") or 0)), -(int(x.get("reviews") or 0))))
    normalized = normalized[:20]
    await _attach_wikimedia_images(normalized, city, country, limit_places=15)
    return destination, normalized


async def _booking_search_stays(
    destination: Dict[str, Any],
    checkin: str,
    checkout: str,
    adults: int,
    rooms: int,
    child_ages: List[int],
    traveler_country: str,
    display_currency: str,
) -> Dict[str, Any]:
    if not BOOKING_API_KEY or not BOOKING_AFFILIATE_ID:
        return {
            "status": "unavailable",
            "provider": "Booking.com Demand API",
            "reason": "BOOKING_API_KEY and/or BOOKING_AFFILIATE_ID is not configured.",
            "hotels": [],
        }

    guests: Dict[str, Any] = {
        "number_of_rooms": rooms,
        "number_of_adults": adults,
    }
    if child_ages:
        guests["children"] = child_ages
        if rooms == 1:
            guests["allocation"] = [{"number_of_adults": adults, "children": child_ages}]

    body: Dict[str, Any] = {
        "coordinates": {
            "latitude": destination["latitude"],
            "longitude": destination["longitude"],
            "radius": BOOKING_RADIUS_KM,
        },
        "booker": {
            "platform": "mobile",
            "country": _normalize_country_iso2(traveler_country),
        },
        "currency": display_currency,
        "checkin": checkin,
        "checkout": checkout,
        "guests": guests,
        "rows": 20,
        "extras": ["extra_charges", "products"],
        "sort": {
            "by": "distance",
            "direction": "ascending",
        },
    }

    endpoint = f"{BOOKING_BASE_URL}/accommodations/search"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0)) as client:
            response = await client.post(endpoint, headers=_booking_headers(), json=body)
        if response.status_code != 200:
            print(f"[Booking Search Notice] {response.status_code}: {response.text[:500]}")
            return {
                "status": "unavailable",
                "provider": "Booking.com Demand API",
                "reason": f"Provider returned HTTP {response.status_code}.",
                "hotels": [],
            }
        payload = response.json()
    except Exception as e:
        print(f"[Booking Search Notice]: {e}")
        return {
            "status": "unavailable",
            "provider": "Booking.com Demand API",
            "reason": "Provider request failed or timed out.",
            "hotels": [],
        }

    hotels: List[Dict[str, Any]] = []
    for raw in payload.get("data") or []:
        item = _normalize_booking_hotel(
            raw,
            city=destination.get("city", ""),
            traveler_currency=display_currency,
            center_lat=destination.get("latitude"),
            center_lng=destination.get("longitude"),
        )
        if item:
            hotels.append(item)

    return {
        "status": "success" if hotels else "empty",
        "provider": "Booking.com Demand API",
        "request_id": payload.get("request_id"),
        "next_page": payload.get("metadata", {}).get("next_page_token") or payload.get("next_page"),
        "hotels": hotels,
    }


def _normalize_booking_hotel(
    raw: Dict[str, Any],
    city: str,
    traveler_currency: str,
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    name = raw.get("name") or raw.get("property_name")
    if not name:
        return None

    location = raw.get("location") or {}
    coords = location.get("coordinates") or {}
    price = raw.get("price") or {}
    currency = raw.get("currency") or {}
    if isinstance(currency, str):
        accommodation_currency = currency
        booker_currency = traveler_currency
    else:
        accommodation_currency = currency.get("accommodation") or traveler_currency
        booker_currency = currency.get("booker") or traveler_currency

    display_price = price.get("display") if isinstance(price, dict) else None
    total = None
    if isinstance(display_price, dict):
        total = display_price.get("total") or display_price.get("base")
    total = total if total is not None else price.get("total") if isinstance(price, dict) else None
    if total is None and isinstance(price, dict):
        total = price.get("book") or price.get("display")

    charges = price.get("charges") if isinstance(price, dict) else None
    products = raw.get("products") or []
    primary_product = products[0] if products else {}
    policy = primary_product.get("policies") or {}
    cancellation = policy.get("cancellation") or {}
    meal_plan = policy.get("meal_plan") or {}
    payment = policy.get("payment") or {}
    inventory = primary_product.get("inventory") or {}
    urls = raw.get("url") or {}
    booking_url = urls.get("web") if isinstance(urls, dict) else urls if isinstance(urls, str) else None

    review_score = raw.get("review_score")
    if review_score is None:
        rating_obj = raw.get("rating")
        if isinstance(rating_obj, dict):
            review_score = rating_obj.get("score") or rating_obj.get("value")
        else:
            review_score = rating_obj

    images = []
    for key in ["photos", "images"]:
        candidate = raw.get(key)
        if isinstance(candidate, list):
            images.extend([str(x.get("url") or x.get("source") or x) for x in candidate if x])

    hotel_lat = coords.get("latitude") if isinstance(coords, dict) else None
    hotel_lng = coords.get("longitude") if isinstance(coords, dict) else None
    distance_km = None
    distance_text = ""
    if hotel_lat is not None and hotel_lng is not None and center_lat is not None and center_lng is not None:
        try:
            distance_km = round(_place_distance_km(float(hotel_lat), float(hotel_lng), float(center_lat), float(center_lng)), 2)
            distance_text = f"{distance_km:.1f} km from Center"
        except (TypeError, ValueError):
            distance_km = None
            distance_text = ""

    result = {
        "name": name,
        "tier": "Stay",
        "rating": review_score,
        "reviews": raw.get("review_count") or raw.get("number_of_reviews"),
        "price": total,
        "price_total": total,
        "currency": accommodation_currency,
        "booker_currency": booker_currency,
        "price_display_currency": booker_currency,
        "price_display_symbol": _currency_symbol(booker_currency),
        "phone": location.get("phone") or raw.get("phone"),
        "distance": distance_text,
        "distance_km": distance_km,
        "amenities": "",
        "lat": hotel_lat,
        "lng": hotel_lng,
        "images": images[:5],
        "address": location.get("address"),
        "city": city,
        "accommodation_id": raw.get("id"),
        "booking_url": booking_url,
        "cancellation_type": cancellation.get("type"),
        "free_cancellation_until": cancellation.get("free_cancellation_until"),
        "meal_plan": meal_plan.get("plan") or meal_plan.get("meals"),
        "payment_timings": payment.get("timings") or [],
        "inventory_type": inventory.get("type"),
        "third_party_inventory": inventory.get("third_party"),
        "charges": charges,
        "product_id": primary_product.get("id"),
        "source": "Booking.com Demand API",
        "data_state": "PROVIDER",
        "provider_verified": True,
    }

    return result


def _duffel_cabin(value: str) -> str:
    raw = str(value or "Economy").strip().lower()
    mapping = {
        "economy": "economy",
        "premium economy": "premium_economy",
        "premium_economy": "premium_economy",
        "business": "business",
        "first class": "first",
        "first": "first",
    }
    return mapping.get(raw, "economy")


def _build_aviasales_search_url(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    children: int,
    infants: int,
    cabin_class: str,
) -> str:
    """Build an Aviasales route-search URL as a booking fallback.

    This is intentionally a search/booking handoff, not a fabricated agency
    ticket URL. The user completes the purchase on Aviasales/its agencies.
    Travelpayouts documents the /search/PARAMS format and the required adult
    passenger count.
    """
    def ddmm(value: Optional[str]) -> str:
        if not value:
            return ""
        try:
            return datetime.fromisoformat(str(value)[:10]).strftime("%d%m")
        except Exception:
            return ""

    cabin = {
        "business": "c",
        "premium_economy": "w",
        "premium economy": "w",
        "first": "f",
        "first class": "f",
    }.get(str(cabin_class or "economy").strip().lower(), "")
    pax = f"{cabin}{max(1, int(adults or 1))}{max(0, int(children or 0))}{max(0, int(infants or 0))}"
    params = f"{origin.upper()}{ddmm(departure_date)}{destination.upper()}"
    if return_date:
        params += f"{ddmm(return_date)}"
    params += pax
    return f"https://www.aviasales.com/search/{params}"


def _normalize_travelpayouts_offer(
    offer: Dict[str, Any],
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str],
    traveler_currency: str,
    cabin_class: str,
) -> Optional[Dict[str, Any]]:
    try:
        price = _safe_decimal(offer.get("price"))
        if price <= 0:
            return None

        source_currency = str(offer.get("currency") or traveler_currency or "USD").upper()
        departure_at = offer.get("departure_at")
        return_at = offer.get("return_at")
        transfers = offer.get("transfers")
        return_transfers = offer.get("return_transfers")

        try:
            stop_count = int(transfers or 0)
        except (TypeError, ValueError):
            stop_count = 0

        try:
            return_stop_count = int(return_transfers or 0)
        except (TypeError, ValueError):
            return_stop_count = 0

        if return_date and return_at:
            stops = f"{stop_count + return_stop_count} stop(s) total"
        else:
            stops = "Direct" if stop_count == 0 else f"{stop_count} stop(s)"

        duration_minutes = offer.get("duration")
        duration_to = offer.get("duration_to")
        duration_back = offer.get("duration_back")
        duration = _minutes_to_duration(duration_minutes)
        if not duration:
            duration = _minutes_to_duration(duration_to)
        if return_date and duration_back:
            back_duration = _minutes_to_duration(duration_back)
            if back_duration:
                duration = f"{duration} outbound • {back_duration} return" if duration else back_duration

        outbound_duration_minutes = duration_to or duration_minutes
        outbound_arrival_at = _add_minutes_iso(departure_at, outbound_duration_minutes)
        return_arrival_at = _add_minutes_iso(return_at, duration_back)

        flight_number = str(offer.get("flight_number") or "").strip()
        airline = str(offer.get("airline") or "").strip()
        # Aviasales Data API can return a relative ticket/search link.
        # Flutter's URL launcher requires an absolute URL, so normalize it.
        # If no provider link is available, the app will use the safe route
        # search fallback generated below.
        booking_link = str(offer.get("link") or "").strip()
        if booking_link.startswith("/"):
            booking_link = "https://www.aviasales.com" + booking_link
        elif booking_link and not booking_link.startswith(("http://", "https://")):
            booking_link = "https://www.aviasales.com/" + booking_link.lstrip("/")

        outbound_slice = {
            "origin": origin,
            "destination": destination,
            "departure_datetime": departure_at,
            "arrival_datetime": outbound_arrival_at,
            "departure_time": _format_clock(departure_at),
            "arrival_time": _format_clock(outbound_arrival_at),
            "duration": _minutes_to_duration(outbound_duration_minutes),
            "stop_count": stop_count,
            "flight_number": flight_number,
            "airline": airline,
        }
        flight_slices = [outbound_slice]

        if return_date and return_at:
            flight_slices.append({
                "origin": destination,
                "destination": origin,
                "departure_datetime": return_at,
                "arrival_datetime": return_arrival_at,
                "departure_time": _format_clock(return_at),
                "arrival_time": _format_clock(return_arrival_at),
                "duration": _minutes_to_duration(duration_back),
                "stop_count": return_stop_count,
                "flight_number": flight_number,
                "airline": airline,
            })

        # Data API is cached/provider data, not a live multi-passenger quote.
        # Preserve the provider fare as the displayed base fare and let the
        # existing FX layer convert it for the traveler.
        normalized = {
            "id": _encode_travelpayouts_offer({
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date,
                "traveler_currency": traveler_currency,
                "cabin_class": cabin_class,
                "offer": offer,
            }),
            "origin": origin,
            "destination": destination,
            "depart_datetime": departure_at,
            "arrive_datetime": outbound_arrival_at,
            "depart_date": _format_date(departure_at) or departure_date,
            "arrive_date": _format_date(outbound_arrival_at) or departure_date,
            "departure_time": _format_clock(departure_at),
            "arrival_time": _format_clock(outbound_arrival_at),
            "return_depart_datetime": return_at,
            "return_date": _format_date(return_at) if return_at else return_date,
             "return_arrive_datetime": return_arrival_at,
            "duration": duration,
            "stops": stops,
            "stop_count": stop_count,
            "return_stop_count": return_stop_count,
            "price": float(price),
            "total_price": float(price),
            "price_per_passenger": float(price),
            "currency": source_currency,
            "symbol": _currency_symbol(source_currency),
            "expires_at": None,
            "source": "Aviasales Data API",
            "data_state": "CACHED_PROVIDER",
            "provider_verified": True,
            "live_mode": False,
            "booking_capable": bool(booking_link),
            "booking_reference": booking_link or None,
            "ticket_link": booking_link or None,
            "cabin_class": _travelpayouts_cabin_class(cabin_class),
            "airline": airline,
            "operating_carrier": airline,
            "flight_number": flight_number,
            "flight_numbers": [flight_number] if flight_number else [],
            "fare_conditions": {},
            "slices": flight_slices,
            "baggage": [],
            "provider_found_at": offer.get("found_at"),
            "passenger_pricing_note": (
                "Provider fare from Aviasales cached search data. "
                "Final multi-passenger price and availability must be confirmed by the booking provider."
            ),
        }
        return normalized
    except Exception as e:
        print(f"[Travelpayouts Normalize Notice]: {e}")
        return None


async def _search_travelpayouts_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    child_ages: List[int],
    cabin_class: str,
    max_connections: int,
    traveler_country: str,
    traveler_currency: str,
) -> Dict[str, Any]:
    if not TRAVELPAYOUTS_API_TOKEN:
        return {
            "status": "unavailable",
            "provider": "Aviasales Data API",
            "reason": "TRAVELPAYOUTS_API_TOKEN is not configured.",
            "live_mode": False,
            "flights": [],
        }

    endpoint = f"{TRAVELPAYOUTS_AVIASALES_BASE_URL}/prices_for_dates"
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "departure_at": departure_date,
        "one_way": "true" if not return_date else "false",
        "unique": "false",
        "sorting": "price",
        "currency": traveler_currency.lower(),
        "limit": 100,
        "page": 1,
        "market": _normalize_country_iso2(traveler_country),
    }

    if return_date:
        params["return_at"] = return_date
    if max_connections == 0:
        params["direct"] = "true"

    async def _fetch(params_to_use: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(TRAVELPAYOUTS_TIMEOUT_SECONDS, connect=6.0)
            ) as client:
                response = await client.get(
                    endpoint,
                    headers=_travelpayouts_headers(),
                    params=params_to_use,
                )
            if response.status_code != 200:
                print(
                    f"[Travelpayouts Search Notice] {response.status_code}: "
                    f"{response.text[:500]}"
                )
                return None, f"Provider returned HTTP {response.status_code}."
            payload = response.json()
            return (payload if isinstance(payload, dict) else None), None
        except Exception as e:
            print(f"[Travelpayouts Search Notice]: {e}")
            return None, "Provider request failed or timed out."

    payload, fetch_error = await _fetch(params)
    if fetch_error:
        return {
            "status": "unavailable",
            "provider": "Aviasales Data API",
            "reason": fetch_error,
            "live_mode": False,
            "flights": [],
        }

    raw_prices = (payload or {}).get("data") or []
    if not isinstance(raw_prices, list):
        raw_prices = []

    # The Data API is cached data, so an exact future round-trip can legitimately
    # have no record even when the route itself has fares. In that case make a
    # second month-level request and surface the nearest cached dates instead of
    # making the user think the provider is broken.
    fallback_mode = "EXACT_DATE"
    requested_departure = departure_date
    requested_return = return_date
    if not raw_prices and len(departure_date) == 10:
        month_params = dict(params)
        month_params["departure_at"] = departure_date[:7]
        if return_date and len(return_date) == 10:
            month_params["return_at"] = return_date[:7]
        month_payload, month_error = await _fetch(month_params)
        if month_payload is not None:
            month_prices = month_payload.get("data") or []
            if isinstance(month_prices, list) and month_prices:
                raw_prices = month_prices
                fallback_mode = "NEAREST_CACHED_DATE"
        elif month_error:
            print(f"[Travelpayouts Fallback Notice]: {month_error}")
    normalized: List[Dict[str, Any]] = []
    seen: set = set()

    for raw in raw_prices:
        if not isinstance(raw, dict):
            continue

        # The API can return cached records whose transfer count is higher
        # than the UI's requested maximum. Apply the filter locally as well.
        try:
            outbound_stops = int(raw.get("transfers") or 0)
        except (TypeError, ValueError):
            outbound_stops = 0
        try:
            return_stops = int(raw.get("return_transfers") or 0)
        except (TypeError, ValueError):
            return_stops = 0

        if max_connections < 3 and max(outbound_stops, return_stops) > max_connections:
            continue

        item = _normalize_travelpayouts_offer(
            raw,
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            traveler_currency=traveler_currency,
            cabin_class=cabin_class,
        )
        if not item:
            continue

        key = (
            item.get("flight_number"),
            item.get("depart_datetime"),
            item.get("return_depart_datetime"),
            item.get("total_price"),
            item.get("currency"),
        )
        if key in seen:
            continue
        seen.add(key)
        # Data API results may not contain a directly usable ticket link.
        # Provide a provider search handoff so every displayed result still has
        # a real booking path instead of a dead/invalid button.
        if not item.get("ticket_link"):
            item["ticket_link"] = _build_aviasales_search_url(
                origin=origin,
                destination=destination,
                departure_date=str(item.get("depart_datetime") or departure_date)[:10],
                return_date=(str(item.get("return_depart_datetime") or return_date)[:10] if (item.get("return_depart_datetime") or return_date) else None),
                adults=adults,
                children=len(child_ages),
                infants=0,
                cabin_class=cabin_class,
            )
            item["booking_reference"] = item["ticket_link"]
            item["booking_capable"] = True
            item["booking_mode"] = "AVIASALES_ROUTE_SEARCH"
        elif item.get("ticket_link"):
            item["booking_mode"] = "PROVIDER_OR_AVIASALES_LINK"

        normalized.append(item)

    normalized.sort(key=lambda x: float(x.get("total_price") or 0))
    normalized = normalized[:50]

    if normalized:
        reason = (
            None
            if fallback_mode == "EXACT_DATE"
            else "No exact cached fare was found; showing the closest cached fares available for the selected month."
        )
    else:
        reason = "No cached provider fares were found for this route/date or its selected month."

    for item in normalized:
        item["requested_departure_date"] = requested_departure
        item["requested_return_date"] = requested_return
        item["date_match"] = fallback_mode

    return {
        "status": "success" if normalized else "empty",
        "provider": "Aviasales Data API",
        "offer_request_id": None,
        "live_mode": False,
        "reason": reason,
        "data_freshness": "cached_up_to_48h",
        "date_match": fallback_mode,
        "requested_departure_date": requested_departure,
        "requested_return_date": requested_return,
        "flights": normalized,
    }


@app.post("/api/v1/search-flights")
async def search_flights(request: Request):
    try:
        body = await request.json()
        origin = str(body.get("origin") or "").strip().upper()
        destination = str(body.get("destination") or "").strip().upper()
        departure_date = str(body.get("departure_date") or body.get("depart_date") or "").strip()
        return_date_raw = body.get("return_date")
        return_date = str(return_date_raw).strip() if return_date_raw else None
        adults = int(body.get("adults", 1))
        kids = int(body.get("kids", 0))
        child_ages = [int(x) for x in (body.get("child_ages") or [])]
        cabin_class = str(body.get("cabin_class") or "Economy")
        max_connections = int(body.get("max_connections", 2))
        traveler_country = str(
            body.get("traveler_country") or body.get("booker_country") or "IN"
        )
        traveler_currency = str(
            body.get("traveler_currency") or _currency_for_country(traveler_country)
        ).upper()

        if not re.fullmatch(r"[A-Z]{3}", origin) or not re.fullmatch(r"[A-Z]{3}", destination):
            return {
                "status": "error",
                "message": "Valid 3-letter IATA origin and destination codes are required.",
                "flights": [],
            }
        if origin == destination:
            return {
                "status": "error",
                "message": "Origin and destination cannot be the same airport.",
                "flights": [],
            }
        if not departure_date:
            return {"status": "error", "message": "departure_date is required.", "flights": []}
        if adults < 1 or adults > 9:
            return {
                "status": "error",
                "message": "adults must be between 1 and 9.",
                "flights": [],
            }
        if kids < 0 or kids > 8:
            return {"status": "error", "message": "kids must be between 0 and 8.", "flights": []}
        if kids != len(child_ages):
            return {
                "status": "error",
                "message": "Child ages are required; send one age for each child.",
                "required": ["child_ages"],
                "flights": [],
            }
        if any(age < 2 or age > 11 for age in child_ages):
            return {
                "status": "error",
                "message": "child_ages must contain values from 2 through 11.",
                "flights": [],
            }
        if return_date and return_date <= departure_date:
            return {
                "status": "error",
                "message": "return_date must be later than departure_date.",
                "flights": [],
            }
        if max_connections < 0 or max_connections > 3:
            return {
                "status": "error",
                "message": "max_connections must be between 0 and 3.",
                "flights": [],
            }

        result = await _search_travelpayouts_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            adults=adults,
            child_ages=child_ages,
            cabin_class=cabin_class,
            max_connections=max_connections,
            traveler_country=traveler_country,
            traveler_currency=traveler_currency,
        )

        flights = result.get("flights") or []
        source_currencies = sorted(
            set(str(item.get("currency") or traveler_currency).upper() for item in flights)
        )

        fx_results = await asyncio.gather(
            *[_fx_rate(cur, traveler_currency) for cur in source_currencies],
            return_exceptions=True,
        )
        fx_map: Dict[str, Dict[str, Any]] = {}
        for cur, fx in zip(source_currencies, fx_results):
            if isinstance(fx, dict):
                fx_map[cur] = fx

        for item in flights:
            source_currency = str(
                item.get("currency") or traveler_currency
            ).upper()
            rate_data = fx_map.get(source_currency)
            if rate_data:
                rate = rate_data["rate"]
                source_total = _safe_decimal(item.get("total_price"))
                source_per_passenger = _safe_decimal(item.get("price_per_passenger"))
                item["display_currency"] = traveler_currency
                item["display_symbol"] = _currency_symbol(traveler_currency)
                item["display_total_price"] = float(source_total * rate)
                item["display_price_per_passenger"] = float(source_per_passenger * rate)
                item["fx_rate"] = float(rate)
                item["fx_date"] = rate_data.get("date")
            else:
                item["display_currency"] = source_currency
                item["display_symbol"] = _currency_symbol(source_currency)
                item["display_total_price"] = item.get("total_price")
                item["display_price_per_passenger"] = item.get("price_per_passenger")
                item["fx_rate"] = 1.0
                item["fx_date"] = None

        return {
            "status": result.get("status", "empty"),
            "provider": result.get("provider", "Aviasales Data API"),
            "provider_state": "PROVIDER_CACHE" if flights else "UNAVAILABLE",
            "offer_request_id": None,
            "live_mode": False,
            "reason": result.get("reason"),
            "origin": origin,
            "destination": destination,
            "depart_date": departure_date,
            "return_date": return_date,
            "adults": adults,
            "kids": kids,
            "child_ages": child_ages,
            "cabin_class": cabin_class,
            "currency": traveler_currency,
            "symbol": _currency_symbol(traveler_currency),
            "source_currencies": source_currencies,
            "data_freshness": result.get("data_freshness", "cached_up_to_48h"),
            "flights": flights,
        }
    except Exception as e:
        print(f"[Flight Search Error]: {e}")
        return {"status": "error", "message": str(e), "flights": []}


@app.get("/api/v1/flight-offers/{offer_id}")
async def get_live_flight_offer(
    offer_id: str,
    traveler_currency: str = Query("INR"),
):
    if not TRAVELPAYOUTS_API_TOKEN:
        return {
            "status": "unavailable",
            "provider": "Aviasales Data API",
            "message": "TRAVELPAYOUTS_API_TOKEN is not configured.",
        }

    try:
        decoded = _decode_travelpayouts_offer(offer_id)
        raw_offer = decoded.get("offer") or {}
        if not isinstance(raw_offer, dict):
            return {
                "status": "empty",
                "provider": "Aviasales Data API",
                "offer": None,
            }

        normalized = _normalize_travelpayouts_offer(
            raw_offer,
            origin=str(decoded.get("origin") or raw_offer.get("origin") or ""),
            destination=str(decoded.get("destination") or raw_offer.get("destination") or ""),
            departure_date=str(decoded.get("departure_date") or ""),
            return_date=decoded.get("return_date"),
            traveler_currency=str(
                decoded.get("traveler_currency") or traveler_currency or "INR"
            ).upper(),
            cabin_class=str(decoded.get("cabin_class") or "Economy"),
        )
        if not normalized:
            return {
                "status": "empty",
                "provider": "Aviasales Data API",
                "offer": None,
            }

        source_currency = str(normalized.get("currency") or "USD").upper()
        quote_currency = str(traveler_currency or "INR").upper()
        fx = await _fx_rate(source_currency, quote_currency)
        if fx:
            rate = fx["rate"]
            normalized["display_currency"] = quote_currency
            normalized["display_symbol"] = _currency_symbol(quote_currency)
            normalized["display_total_price"] = float(
                _safe_decimal(normalized["total_price"]) * rate
            )
            normalized["display_price_per_passenger"] = float(
                _safe_decimal(normalized["price_per_passenger"]) * rate
            )
            normalized["fx_rate"] = float(rate)
            normalized["fx_date"] = fx.get("date")
        else:
            normalized["display_currency"] = source_currency
            normalized["display_symbol"] = _currency_symbol(source_currency)
            normalized["display_total_price"] = normalized.get("total_price")
            normalized["display_price_per_passenger"] = normalized.get("price_per_passenger")
            normalized["fx_rate"] = 1.0
            normalized["fx_date"] = None

        return {
            "status": "success",
            "provider": "Aviasales Data API",
            "offer": normalized,
            "data_freshness": "cached_up_to_48h",
        }
    except Exception as e:
        print(f"[Flight Offer Error]: {e}")
        return {
            "status": "unavailable",
            "provider": "Aviasales Data API",
            "message": "The provider offer reference could not be decoded.",
        }


# -------------------------------------------------------------
# NEWS, MARKET INDICES & CRYPTO DATA
# Provider credentials stay on the server; never ship them in Flutter.
# GNews: set GNEWS_API_KEY. CoinGecko: set COINGECKO_DEMO_API_KEY.
# Market indices: optional MARKET_FALLBACK_URL must point to a provider whose terms permit app display/redistribution.
# Groww live mode: set GROWW_ACCESS_TOKEN or GROWW_API_KEY + GROWW_API_SECRET in Render.
# Never put provider credentials in Flutter or return them from API responses.
# -------------------------------------------------------------
NEWS_REFRESH_SECONDS = 15 * 60
_news_cache: Dict[str, Dict[str, Any]] = {}
_groww_token_cache: Dict[str, Any] = {"token": None, "expires_at": 0.0}


def _news_cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = _news_cache.get(key)
    if item and time.time() - item.get("cached_at", 0) < NEWS_REFRESH_SECONDS:
        return item.get("data")
    return None


def _news_cache_put(key: str, data: Dict[str, Any]) -> None:
    _news_cache[key] = {"cached_at": time.time(), "data": data}


@app.get("/api/v1/news/feed")
async def get_news_feed(
    category: str = Query("world"),
    town: str = Query(""),
    city: str = Query(""),
    country: str = Query("India"),
    language: str = Query("en"),
    limit: int = Query(10, ge=1, le=10),
):
    api_key = os.environ.get("GNEWS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="News provider is not configured. Set GNEWS_API_KEY on the backend.")
    category = category.lower().strip()
    allowed = {"local", "city", "national", "world", "financial", "business", "sports", "technology"}
    if category not in allowed:
        raise HTTPException(status_code=400, detail=f"category must be one of: {', '.join(sorted(allowed))}")
    place = ", ".join(part.strip() for part in (town, city, country) if part and part.strip())
    query_by_category = {
        "local": place or "local news",
        "city": (city.strip() or place or "city news"),
        "national": (country.strip() or "India") + " news",
        "world": "world news",
        "financial": "financial markets stock market economy",
        "business": "business startups companies",
        "sports": "sports",
        "technology": "technology AI innovation",
    }
    query = query_by_category[category]
    cache_key = "|".join([category, query, language, str(limit)])
    cached = _news_cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}
    params = {"q": query, "lang": language[:2], "max": limit, "apikey": api_key, "sortby": "publishedAt"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://gnews.io/api/v4/search", params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"News provider returned HTTP {response.status_code}.")
        payload = response.json()
        articles = []
        for article in payload.get("articles", []):
            articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "image": article.get("image"),
                "published_at": article.get("publishedAt"),
                "source": (article.get("source") or {}).get("name", "News source"),
            })
        result = {"status": "success", "provider": "GNews", "category": category,
                  "query": query, "articles": articles, "updated_at": datetime.now(timezone.utc).isoformat(), "cached": False}
        _news_cache_put(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        stale = _news_cache.get(cache_key, {}).get("data")
        if stale:
            return {**stale, "cached": True, "warning": "Provider temporarily unavailable; showing cached results."}
        raise HTTPException(status_code=502, detail="News provider is temporarily unavailable.") from exc


@app.get("/api/v1/news/breaking")
async def get_breaking_news(country: str = Query("in"), language: str = Query("en"), limit: int = Query(5, ge=1, le=10)):
    api_key = os.environ.get("GNEWS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="News provider is not configured. Set GNEWS_API_KEY on the backend.")
    cache_key = f"breaking|{country}|{language}|{limit}"
    cached = _news_cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://gnews.io/api/v4/top-headlines", params={
                "country": country[:2].lower(), "lang": language[:2], "max": limit, "apikey": api_key})
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"News provider returned HTTP {response.status_code}.")
        payload = response.json()
        result = {"status": "success", "provider": "GNews", "articles": [{
            "title": a.get("title"), "description": a.get("description"), "url": a.get("url"),
            "image": a.get("image"), "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name", "News source")
        } for a in payload.get("articles", [])], "updated_at": datetime.now(timezone.utc).isoformat(), "cached": False}
        _news_cache_put(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Breaking-news provider is temporarily unavailable.") from exc


@app.get("/api/v1/markets/crypto")
async def get_crypto_markets(vs_currency: str = Query("usd"), ids: str = Query("bitcoin,ethereum,solana,ripple")):
    api_key = os.environ.get("COINGECKO_DEMO_API_KEY", "").strip()
    cache_key = f"crypto|{vs_currency.lower()}|{ids}"
    cached = _news_cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}
    headers = {"accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://api.coingecko.com/api/v3/coins/markets", params={
                "vs_currency": vs_currency.lower(), "ids": ids, "order": "market_cap_desc", "per_page": 10,
                "page": 1, "sparkline": "false", "price_change_percentage": "24h"}, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Crypto provider returned HTTP {response.status_code}; configure COINGECKO_DEMO_API_KEY if needed.")
        rows = [{"id": x.get("id"), "symbol": x.get("symbol", "").upper(), "name": x.get("name"),
                 "price": x.get("current_price"), "change_24h": x.get("price_change_percentage_24h"),
                 "market_cap": x.get("market_cap"), "last_updated": x.get("last_updated"), "image": x.get("image")}
                for x in response.json()]
        result = {"status": "success", "provider": "CoinGecko", "currency": vs_currency.lower(),
                  "assets": rows, "updated_at": datetime.now(timezone.utc).isoformat(), "cached": False,
                  "attribution": "Market data provided by CoinGecko"}
        _news_cache_put(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Crypto market data is temporarily unavailable.") from exc


async def _get_groww_access_token(client: httpx.AsyncClient) -> str:
    """Use a dashboard token when configured; otherwise generate a token via API key/secret."""
    now = time.time()
    dashboard_token = os.environ.get("GROWW_ACCESS_TOKEN", "").strip().strip('"').strip("'")
    if dashboard_token:
        # Dashboard-issued tokens expire daily. Cache it only until a conservative expiry.
        cached = _groww_token_cache.get("dashboard_token")
        if cached == dashboard_token and now < float(_groww_token_cache.get("dashboard_expires_at", 0)):
            return dashboard_token
        _groww_token_cache["dashboard_token"] = dashboard_token
        _groww_token_cache["dashboard_expires_at"] = now + (20 * 60 * 60)
        return dashboard_token

    cached_token = _groww_token_cache.get("token")
    if cached_token and now < float(_groww_token_cache.get("expires_at", 0)):
        return str(cached_token)

    api_key = os.environ.get("GROWW_API_KEY", "").strip().strip('"').strip("'")
    api_secret = os.environ.get("GROWW_API_SECRET", "").strip().strip('"').strip("'")
    if not api_key or not api_secret:
        raise HTTPException(status_code=503, detail="Groww credentials are not configured.")

    timestamp = str(int(now))
    checksum = __import__("hashlib").sha256(f"{api_secret}{timestamp}".encode("utf-8")).hexdigest()
    try:
        response = await client.post(
            "https://api.groww.in/v1/token/api/access",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            json={"key_type": "approval", "checksum": checksum, "timestamp": timestamp},
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Groww authentication service.") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groww authentication failed (HTTP {response.status_code}).")
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Groww authentication returned an invalid response.") from exc
    token = payload.get("token") if isinstance(payload, dict) else None
    if not token:
        raise HTTPException(status_code=502, detail="Groww authentication response did not contain an access token.")

    expires_at = now + (20 * 60 * 60)
    expiry_text = payload.get("expiry")
    if expiry_text:
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_text).replace("Z", "+00:00"))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            expires_at = min(expires_at, expiry_dt.timestamp() - 300)
        except (TypeError, ValueError, OverflowError):
            pass
    _groww_token_cache["token"] = str(token)
    _groww_token_cache["expires_at"] = max(now + 60, expires_at)
    return str(token)


def _normalise_market_indices(payload: Any, provider: str, data_mode: str) -> Optional[Dict[str, Any]]:
    """Validate the provider-neutral index response before it reaches Flutter."""
    if not isinstance(payload, dict) or not isinstance(payload.get("indices"), list):
        return None
    wanted = {"NSE_NIFTY": ("Nifty 50", "NSE"), "BSE_SENSEX": ("Sensex", "BSE")}
    by_symbol = {row.get("instrument_key"): row for row in payload["indices"] if isinstance(row, dict)}
    rows = []
    for symbol, (name, exchange) in wanted.items():
        row = by_symbol.get(symbol)
        if not row:
            continue
        try:
            price = float(row.get("last_price"))
            if price <= 0:
                continue
        except (TypeError, ValueError):
            continue
        def _optional_float(value):
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None
        rows.append({
            "instrument_key": symbol,
            "name": name,
            "exchange": exchange,
            "last_price": price,
            "net_change": _optional_float(row.get("net_change")),
            "change_percent": _optional_float(row.get("change_percent")),
            "ohlc": row.get("ohlc") if isinstance(row.get("ohlc"), dict) else None,
            "timestamp": row.get("timestamp") or payload.get("data_as_of") or payload.get("updated_at"),
        })
    if not rows:
        return None
    return {
        "status": "success",
        "provider": provider,
        "data_mode": data_mode,
        "indices": rows,
        "updated_at": payload.get("updated_at") or datetime.now(timezone.utc).isoformat(),
        "data_as_of": payload.get("data_as_of") or max((str(r.get("timestamp") or "") for r in rows), default=None),
        "cached": False,
        "is_stale": bool(payload.get("is_stale", False)),
    }


async def _get_market_fallback(client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fetch an operator-configured, licensed fallback source; no public scrape is assumed."""
    fallback_url = os.environ.get("MARKET_FALLBACK_URL", "").strip()
    if not fallback_url:
        return None
    try:
        response = await client.get(fallback_url, headers={"Accept": "application/json"})
        if response.status_code != 200:
            print(f"[Market fallback notice] configured provider returned HTTP {response.status_code}")
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"[Market fallback notice] provider unavailable: {type(exc).__name__}")
        return None
    mode = str(payload.get("data_mode", "last_close")) if isinstance(payload, dict) else "last_close"
    if mode not in {"delayed", "last_close"}:
        # Keep this fallback deliberately non-live until a live provider is licensed/configured.
        mode = "last_close"
    provider = str(payload.get("provider") or "Configured fallback provider") if isinstance(payload, dict) else "Configured fallback provider"
    return _normalise_market_indices(payload, provider, mode)


@app.get("/api/v1/markets/indices")
async def get_market_indices():
    """Provider-independent Nifty/Sensex endpoint with live attempt and compliant fallback."""
    cache_key = "indices|provider-neutral|NSE_NIFTY|BSE_SENSEX"
    cached = _news_cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    live_error = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0)) as client:
            # Groww is retained as a future live provider; a free-trial 403 falls through safely.
            try:
                token = await _get_groww_access_token(client)
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "X-API-VERSION": "1.0"}
                ltp_response = await client.get(
                    "https://api.groww.in/v1/live-data/ltp",
                    params={"segment": "CASH", "exchange_trading_symbols": "NSE_NIFTY,BSE_SENSEX"},
                    headers=headers,
                )
                if ltp_response.status_code != 200:
                    live_error = f"Groww returned HTTP {ltp_response.status_code}"
                else:
                    ltp_json = ltp_response.json()
                    ltp_payload = ltp_json.get("payload", ltp_json) if isinstance(ltp_json, dict) else {}
                    ohlc_payload = {}
                    ohlc_response = await client.get(
                        "https://api.groww.in/v1/live-data/ohlc",
                        params={"segment": "CASH", "exchange_trading_symbols": "NSE_NIFTY,BSE_SENSEX"},
                        headers=headers,
                    )
                    if ohlc_response.status_code == 200:
                        try:
                            ohlc_json = ohlc_response.json()
                            ohlc_payload = ohlc_json.get("payload", ohlc_json) if isinstance(ohlc_json, dict) else {}
                        except ValueError:
                            ohlc_payload = {}
                    rows = []
                    for symbol, name, exchange in [("NSE_NIFTY", "Nifty 50", "NSE"), ("BSE_SENSEX", "Sensex", "BSE")]:
                        raw = ltp_payload.get(symbol) if isinstance(ltp_payload, dict) else None
                        try:
                            price = float(raw) if raw is not None else None
                        except (TypeError, ValueError):
                            price = None
                        if price is None or price <= 0:
                            continue
                        candle = ohlc_payload.get(symbol, {}) if isinstance(ohlc_payload, dict) else {}
                        try:
                            close = float(candle.get("close")) if isinstance(candle, dict) and candle.get("close") is not None else None
                        except (TypeError, ValueError):
                            close = None
                        change = round(price - close, 2) if close else None
                        rows.append({"instrument_key": symbol, "name": name, "exchange": exchange,
                                     "last_price": price, "net_change": change,
                                     "change_percent": round(change / close * 100, 2) if change is not None and close else None,
                                     "ohlc": candle if isinstance(candle, dict) else None,
                                     "timestamp": datetime.now(timezone.utc).isoformat()})
                    if rows:
                        result = {"status": "success", "provider": "Groww", "data_mode": "live", "indices": rows,
                                  "updated_at": datetime.now(timezone.utc).isoformat(), "data_as_of": datetime.now(timezone.utc).isoformat(),
                                  "cached": False, "is_stale": False}
                        _news_cache_put(cache_key, result)
                        return result
                    live_error = "Groww returned no valid index prices"
            except HTTPException as exc:
                live_error = exc.detail
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                live_error = type(exc).__name__

            fallback = await _get_market_fallback(client)
            if fallback:
                _news_cache_put(cache_key, fallback)
                return fallback

    except (httpx.HTTPError, ValueError, TypeError) as exc:
        live_error = type(exc).__name__

    # Serve the last successful snapshot with its original data timestamp; never make it look fresh.
    old_item = _news_cache.get(cache_key)
    if old_item and isinstance(old_item.get("data"), dict):
        stale = dict(old_item["data"])
        stale["cached"] = True
        stale["is_stale"] = True
        stale["warning"] = "Providers unavailable; showing the last successfully received snapshot. Check data_as_of before use."
        return stale

    # A valid unavailable response is safer for the Flutter UI than an unhandled 502.
    return {
        "status": "unavailable",
        "provider": None,
        "data_mode": "unavailable",
        "indices": [],
        "updated_at": None,
        "data_as_of": None,
        "cached": False,
        "is_stale": True,
        "message": "No permitted market-data provider is currently configured or available.",
        "diagnostic": "Groww live data was unavailable; configure MARKET_FALLBACK_URL to a source whose terms permit display/redistribution.",
    }


@app.get("/api/v1/currency-rate")
async def get_currency_rate(base: str = Query(...), quote: str = Query(...)):
    base_code = str(base or "").strip().upper()
    quote_code = str(quote or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", base_code) or not re.fullmatch(r"[A-Z]{3}", quote_code):
        raise HTTPException(status_code=400, detail="base and quote must be valid 3-letter ISO currency codes.")
    rate_data = await _fx_rate(base_code, quote_code)
    if not rate_data:
        raise HTTPException(status_code=503, detail="Currency rate temporarily unavailable.")
    return {
        "status": "success",
        "base": rate_data["base"],
        "quote": rate_data["quote"],
        "rate": float(rate_data["rate"]),
        "date": rate_data.get("date"),
        "provider": rate_data.get("provider"),
        "symbol": _currency_symbol(quote_code),
    }



@app.post("/api/v1/place-details")
async def place_details(request: Request):
    try:
        body = await request.json()
        place_id = str(body.get("place_id") or "").strip()
        name = str(body.get("name") or "Verified place").strip()
        city = str(body.get("city") or "").strip()
        state = str(body.get("state") or "").strip()
        country = str(body.get("country") or "").strip()
        category = str(body.get("category") or "Sights & Landmarks").strip()
        traveler_country = str(body.get("traveler_country") or "IN").strip()
        traveler_currency = str(body.get("traveler_currency") or _currency_for_country(traveler_country)).upper()
        start_date = str(body.get("start_date") or "").strip()
        return_date = str(body.get("return_date") or "").strip()
        adults = max(1, int(body.get("adults", 2)))
        rooms = max(1, int(body.get("rooms", 1)))
        child_ages = [int(x) for x in (body.get("child_ages") or [])]

        if not city or not country:
            return {"status": "error", "message": "Destination city and country are required."}
        if not start_date:
            start_date = (datetime.utcnow() + timedelta(days=3)).date().isoformat()
        if not return_date:
            return_date = (datetime.fromisoformat(start_date) + timedelta(days=4)).date().isoformat()

        # Do not call Google Place Details here. It consumes the daily
        # GetPlaceRequest quota and caused HTTP 429 RESOURCE_EXHAUSTED in the
        # deployed service. The destination card already contains the verified
        # provider facts needed for this screen, so details are built from the
        # card payload plus Omni travel intelligence.
        place = {
            "displayName": {"text": name},
            "primaryTypeDisplayName": {"text": category},
            "formattedAddress": str(body.get("address") or ""),
            "location": {
                "latitude": body.get("lat"),
                "longitude": body.get("lng"),
            },
            "rating": body.get("rating"),
            "userRatingCount": body.get("reviews"),
            "regularOpeningHours": {"weekdayDescriptions": [str(body.get("timing"))]} if body.get("timing") else {},
            "googleMapsUri": body.get("maps_url"),
            "websiteUri": body.get("website_url"),
            "photos": [],
            "types": [],
        }

        display_name = str((place.get("displayName") or {}).get("text") or name)
        primary_label = str((place.get("primaryTypeDisplayName") or {}).get("text") or category)
        formatted_address = str(place.get("formattedAddress") or body.get("address") or "")
        location = place.get("location") or {}
        lat = location.get("latitude")
        lng = location.get("longitude")
        photos = place.get("photos") or []
        photo_names = [str(x.get("name")) for x in photos if x.get("name")]
        hours = _compact_hours(place) or str(body.get("timing") or "")
        rating = place.get("rating") if place.get("rating") is not None else body.get("rating")
        reviews = place.get("userRatingCount") if place.get("userRatingCount") is not None else body.get("reviews")

        open_images = [str(x).strip() for x in (body.get("images") or []) if str(x).strip().startswith(("https://", "http://"))]
        image_credits = body.get("image_credits") if isinstance(body.get("image_credits"), list) else []

        try:
            ai = await ask_fast_json(
                prompt=json.dumps({
                    "place": display_name,
                    "type": primary_label,
                    "category": category,
                    "destination": f"{city}, {state}, {country}" if state else f"{city}, {country}",
                    "address": formatted_address,
                    "opening_hours": hours,
                    "verified_rating": rating,
                    "verified_review_count": reviews,
                }, ensure_ascii=False),
                system_prompt=(
                    "You are Omni TouristOS Destination Intelligence. Return JSON only. "
                    "Create useful, concise travel guidance for the named place and its immediate area. "
                    "Separate provider facts from general travel guidance: never invent exact prices, "
                    "official opening hours, crime incidents, closures, phone numbers, ticket rules, or "
                    "claims that a named business or person is a scam. Safety items must be practical "
                    "general advisories, not allegations. Local food recommendations should be recognizable "
                    "regional specialties and should be phrased cautiously when the place itself has no food service. "
                    "Plans must be practical sequences, not guaranteed schedules. Keep each item concise. "
                    "Required keys: overview (string), best_time (string), things_to_do (array of strings), "
                    "best_food (array of strings), family (string), group (string), solo (string), couple (string), "
                    "safety_alerts (array of strings), local_tips (array of strings), "
                    "visit_plans (object). visit_plans must contain family, group, solo, couple; each must contain "
                    "short_2h, half_day, full_day, each as an array of 3-6 concise step strings. "
                    "Use empty arrays rather than fabricated specifics when confidence is low."
                ),
            ) or {}
        except Exception as ai_error:
            print(f"[Place Intelligence Notice]: {ai_error}")
            ai = {}

        def _ai_list(value: Any, limit: int = 8) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(x).strip() for x in value if str(x).strip()][:limit]

        visit_plans = ai.get("visit_plans") if isinstance(ai.get("visit_plans"), dict) else {}
        normalized_plans: Dict[str, Dict[str, List[str]]] = {}
        for audience in ("family", "group", "solo", "couple"):
            raw_audience = visit_plans.get(audience) if isinstance(visit_plans, dict) else {}
            if not isinstance(raw_audience, dict):
                raw_audience = {}
            normalized_plans[audience] = {
                "short_2h": _ai_list(raw_audience.get("short_2h"), 6),
                "half_day": _ai_list(raw_audience.get("half_day"), 6),
                "full_day": _ai_list(raw_audience.get("full_day"), 6),
            }

        destination_for_hotels = {
            "city": city,
            "latitude": float(lat) if lat is not None else body.get("destination_latitude"),
            "longitude": float(lng) if lng is not None else body.get("destination_longitude"),
        }
        nearby_stays = {"status": "unavailable", "provider": "Booking.com Demand API", "reason": "Place coordinates unavailable.", "hotels": []}
        if destination_for_hotels.get("latitude") is not None and destination_for_hotels.get("longitude") is not None:
            try:
                nearby_stays = await _booking_search_stays(
                    destination=destination_for_hotels,
                    checkin=start_date,
                    checkout=return_date,
                    adults=adults,
                    rooms=rooms,
                    child_ages=child_ages,
                    traveler_country=traveler_country,
                    display_currency=traveler_currency,
                )
            except Exception as stay_error:
                print(f"[Place Hotel Notice]: {stay_error}")
                nearby_stays = {
                    "status": "unavailable",
                    "provider": "Booking.com Demand API",
                    "reason": "Booking provider unavailable; showing verified nearby properties where possible.",
                    "hotels": [],
                }

            if not nearby_stays.get("hotels"):
                try:
                    google_hotels = await _load_google_hotels(
                        f"hotels near {display_name}, {city}, {country}",
                        float(destination_for_hotels["latitude"]),
                        float(destination_for_hotels["longitude"]),
                        city,
                    )
                except Exception as hotel_error:
                    print(f"[Google Hotel Notice]: {hotel_error}")
                    google_hotels = []

                if google_hotels:
                    nearby_stays = {
                        "status": "success",
                        "provider": "Google Places",
                        "reason": "Verified nearby hotel listings are shown. Live room prices and availability require a booking provider.",
                        "hotels": google_hotels,
                    }

        places_provider = str(body.get("places_provider") or body.get("source") or "").strip()

        return {
            "status": "success",
            "place": {
                "name": display_name,
                "type": primary_label,
                "category": category,
                "address": formatted_address,
                "lat": lat,
                "lng": lng,
                "rating": rating,
                "reviews": reviews,
                "timing": hours,
                "maps_url": place.get("googleMapsUri") or body.get("maps_url"),
                "website_url": place.get("websiteUri") or body.get("website_url"),
                "phone": place.get("internationalPhoneNumber") or place.get("nationalPhoneNumber"),
                "overview": ai.get("overview") or str((place.get("editorialSummary") or {}).get("text") or body.get("history") or "Verified place information."),
                "best_time": ai.get("best_time") or "Check current opening hours and local conditions before visiting.",
                "things_to_do": _ai_list(ai.get("things_to_do"), 8) or ["Explore the main visitor areas and follow local site rules."],
                "best_food": _ai_list(ai.get("best_food"), 8) or ["Ask for well-reviewed local specialties nearby."],
                "family": ai.get("family") or "Suitable activities depend on mobility, opening hours and crowd levels.",
                "group": ai.get("group") or "Groups should allow extra time for queues and meeting points.",
                "solo": ai.get("solo") or "Keep valuables secure and use well-lit routes after dark.",
                "couple": ai.get("couple") or "Consider quieter visiting hours for a more relaxed experience.",
                "safety_alerts": _ai_list(ai.get("safety_alerts"), 8),
                "local_tips": _ai_list(ai.get("local_tips"), 8),
                "visit_plans": normalized_plans,
                "intelligence_state": "AI-GUIDED",
                "images": open_images,
                "google_photo_names": [],
                "image_provider": "Wikimedia Commons" if open_images else "NONE",
                "image_credits": image_credits,
                "source": "Provider facts + OpenStreetMap-linked imagery + Omni travel intelligence",
                "data_state": "VERIFIED_PROVIDER_FACTS_PLUS_AI_GUIDANCE",
                "details_state": "AI_GUIDED_WITH_VERIFIED_PROVIDER_FACTS",
            },
            "hotels": nearby_stays.get("hotels", []),
            "hotel_state": nearby_stays.get("status"),
            "hotel_provider": nearby_stays.get("provider"),
            "hotel_reason": nearby_stays.get("reason"),
            "attribution_required": (["Google Maps", "Booking.com"]
                                     if places_provider == "Google Places"
                                     else ["OpenStreetMap", "Wikimedia Commons", "Booking.com"]),
        }
    except Exception as e:
        print(f"[Place Details Error]: {e}")
        return {"status": "error", "message": "Could not load verified place details."}


@app.post("/api/v1/explore-city")
async def explore_city(request: Request):
    try:
        body = await request.json()
        city = str(body.get("city") or "").strip()
        state = str(body.get("state") or "").strip()
        country = str(body.get("country") or "").strip()
        traveler_country = str(body.get("traveler_country") or body.get("booker_country") or "IN").strip()
        traveler_currency = str(body.get("traveler_currency") or _currency_for_country(traveler_country)).upper()
        start_date = str(body.get("start_date") or "").strip()
        return_date = str(body.get("return_date") or "").strip()
        adults = int(body.get("adults", 2))
        rooms = max(1, int(body.get("rooms", 1)))
        child_ages = [int(x) for x in (body.get("child_ages") or [])]
        kids = int(body.get("kids", len(child_ages)))

        if not city:
            return {"status": "error", "message": "city is required.", "landmarks": [], "places": [], "hotels": []}
        if not start_date:
            start_date = (datetime.utcnow() + timedelta(days=3)).date().isoformat()
        if not return_date:
            return_date = (datetime.fromisoformat(start_date) + timedelta(days=4)).date().isoformat()
        if return_date <= start_date:
            return {"status": "error", "message": "return_date must be later than start_date.", "landmarks": [], "places": [], "hotels": []}
        if kids != len(child_ages):
            return {
                "status": "error",
                "message": "Child ages are required for provider-backed stay pricing.",
                "required": ["child_ages"],
                "landmarks": [],
                "places": [],
                "hotels": [],
            }

        destination, places = await _load_google_destination_places(city, state, country)
        places_provider = "Google Places"
        if not destination:
            # Google Places is optional. If its key/quota/permissions are unavailable,
            # continue with real open-data destination records rather than returning
            # the misleading "no verified attractions" state.
            destination, places = await _load_open_destination_places(city, state, country)
            places_provider = "Wikipedia/Wikimedia + OpenStreetMap"
        elif _google_places_disabled:
            # Google resolved the destination before the circuit breaker tripped.
            # Keep the already-resolved destination but use open records for attractions.
            open_destination, open_places = await _load_open_destination_places(city, state, country)
            if open_destination:
                destination, places = open_destination, open_places
                places_provider = "Wikipedia/Wikimedia + OpenStreetMap"

        if not destination:
            return {
                "status": "unavailable",
                "message": "Destination resolution is unavailable from Google Places and the open-data destination providers.",
                "destination": None,
                "landmarks": [],
                "places": [],
                "hotels": [],
                "currency": traveler_currency,
                "destination_currency": _currency_for_country(country),
            }

        destination_country_currency = _currency_for_country(country)
        stays = await _booking_search_stays(
            destination=destination,
            checkin=start_date,
            checkout=return_date,
            adults=adults,
            rooms=rooms,
            child_ages=child_ages,
            traveler_country=traveler_country,
            display_currency=traveler_currency,
        )
        if not stays.get("hotels") and not _google_places_disabled:
            google_hotels = await _load_google_hotels(
                f"{city}, {country}",
                float(destination["latitude"]),
                float(destination["longitude"]),
                city,
            )
            if google_hotels:
                stays = {
                    "status": "success",
                    "provider": "Google Places",
                    "reason": "Verified nearby hotel listings are shown. Live room prices and availability require a booking provider.",
                    "hotels": google_hotels,
                }

        return {
            "status": "success",
            "destination": {
                **destination,
                "currency": destination_country_currency,
                "currency_symbol": _currency_symbol(destination_country_currency),
                "display_currency": traveler_currency,
                "display_currency_symbol": _currency_symbol(traveler_currency),
            },
            "traveler_currency": traveler_currency,
            "traveler_currency_symbol": _currency_symbol(traveler_currency),
            "destination_currency": destination_country_currency,
            "destination_currency_symbol": _currency_symbol(destination_country_currency),
            "start_date": start_date,
            "return_date": return_date,
            "adults": adults,
            "kids": kids,
            "child_ages": child_ages,
            "places_provider": places_provider,
            "places_state": "VERIFIED" if places else "EMPTY",
            "landmarks": places,
            "places": places,
            "hotels": stays.get("hotels", []),
            "hotel_provider": stays.get("provider"),
            "hotel_state": stays.get("status"),
            "hotel_reason": stays.get("reason"),
            "hotel_request_id": stays.get("request_id"),
            "attribution_required": ["OpenStreetMap", "Google Maps", "Booking.com"],
        }
    except Exception as e:
        print(f"[Destination Explore Error]: {e}")
        return {"status": "error", "message": str(e), "landmarks": [], "places": [], "hotels": []}


# -------------------------------------------------------------
# 10. CONCIERGE MULTI-TURN TEXT HELPER
# -------------------------------------------------------------

# -------------------------------------------------------------
async def ask_concierge_text(prompt: str, system_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history[-24:]:
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
        if max(pil_img.size) > 2200:
            pil_img.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
        out_buf = io.BytesIO()
        pil_img.save(out_buf, format="JPEG", quality=95, optimize=True)
        return out_buf.getvalue()
    except Exception:
        return None

# -------------------------------------------------------------
# 15. PAPER PILOT UNIVERSAL DOCUMENT AUDITOR
# -------------------------------------------------------------

async def _extract_scanned_pdf_with_vision(file_bytes: bytes, target_language: str, max_pages: int = 10) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        chunks = []
        count = min(len(doc), max_pages)
        for idx in range(count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            jpg = pix.tobytes("jpeg", jpg_quality=82)
            text = await ask_fast_vision(jpg, f"page-{idx+1}.jpg", target_language)
            if text:
                chunks.append(f"--- [SCANNED PAGE {idx+1} OF {len(doc)}] ---\n{text}")
        doc.close()
        return "\n\n".join(chunks).strip()
    except Exception as exc:
        print(f"[PaperPilot scanned PDF notice]: {exc}")
        return ""


def _file_kind(filename: str, content_type: str = "") -> str:
    ext = os.path.splitext(filename.lower())[1]
    groups = {
        "pdf": {".pdf"},
        "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
        "word": {".doc", ".docx", ".odt", ".rtf"},
        "spreadsheet": {".xls", ".xlsx", ".xlsm", ".csv", ".tsv"},
        "presentation": {".ppt", ".pptx", ".odp"},
        "ebook": {".epub"},
        "text": {".txt", ".md", ".json", ".xml", ".html", ".htm", ".yaml", ".yml"},
        "audio": {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"},
        "video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    }
    for kind, exts in groups.items():
        if ext in exts:
            return kind
    if "pdf" in content_type.lower():
        return "pdf"
    if content_type.lower().startswith("image/"):
        return "image"
    if content_type.lower().startswith("audio/"):
        return "audio"
    if content_type.lower().startswith("video/"):
        return "video"
    return "binary"


def extract_text_from_pptx(file_bytes: bytes) -> Tuple[str, int]:
    """Read PPTX text/tables using the OOXML zip structure; no extra package required."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            slide_names = sorted(
                [n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)],
                key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)),
            )
            ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
            chunks = []
            for idx, name in enumerate(slide_names, 1):
                root = ET.fromstring(zf.read(name))
                texts = [node.text.strip() for node in root.iter(ns + "t") if node.text and node.text.strip()]
                if texts:
                    chunks.append(f"--- [SLIDE {idx} OF {len(slide_names)}] ---\n" + "\n".join(texts))
            return "\n\n".join(chunks).strip(), len(slide_names)
    except Exception:
        return inspect_binary_stream(file_bytes), 0


def extract_text_from_epub(file_bytes: bytes) -> Tuple[str, int]:
    """Extract EPUB XHTML/HTML chapters using only zip + HTML stripping."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
            chunks = []
            for idx, name in enumerate(names, 1):
                raw = zf.read(name).decode("utf-8", errors="ignore")
                raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
                raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
                text = re.sub(r"&amp;", "&", text, flags=re.IGNORECASE)
                text = re.sub(r"\\s+", " ", text).strip()
                if text:
                    chunks.append(f"--- [CHAPTER/DOCUMENT {idx}] ---\n{text}")
            return "\n\n".join(chunks).strip(), len(chunks)
    except Exception:
        return inspect_binary_stream(file_bytes), 0


def _media_probe(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    result = {"duration_seconds": None, "width": None, "height": None, "codec": None, "bitrate": None}
    suffix = os.path.splitext(filename)[1] or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", tmp_path],
                capture_output=True, text=True, timeout=12,
            )
            if proc.returncode == 0 and proc.stdout:
                data = json.loads(proc.stdout)
                fmt = data.get("format", {})
                result["duration_seconds"] = float(fmt.get("duration")) if fmt.get("duration") else None
                result["bitrate"] = int(float(fmt.get("bit_rate"))) if fmt.get("bit_rate") else None
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        result["width"] = stream.get("width")
                        result["height"] = stream.get("height")
                        result["codec"] = stream.get("codec_name")
                        break
                    if stream.get("codec_type") == "audio" and result["codec"] is None:
                        result["codec"] = stream.get("codec_name")
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass
    except Exception:
        pass
    return result


async def _transcribe_media(file_bytes: bytes, filename: str) -> str:
    client = get_groq_client()
    if client is None:
        return ""
    try:
        suffix = os.path.splitext(filename)[1] or ".mp3"
        # Groq's transcription endpoint accepts audio files. For video, extract
        # the audio track first when ffmpeg is available.
        audio_bytes = file_bytes
        audio_name = filename
        if _file_kind(filename) == "video":
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
                src.write(file_bytes)
                src_path = src.name
            out_path = src_path + ".mp3"
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg is not None else "ffmpeg"
                subprocess.run([ffmpeg_exe, "-y", "-i", src_path, "-vn", "-acodec", "libmp3lame", "-b:a", "96k", out_path], capture_output=True, timeout=90, check=True)
                audio_bytes = open(out_path, "rb").read()
                audio_name = os.path.basename(out_path)
            finally:
                for path in (src_path, out_path):
                    try: os.unlink(path)
                    except Exception: pass
        bio = io.BytesIO(audio_bytes)
        bio.name = audio_name
        transcript = client.audio.transcriptions.create(
            file=bio,
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        return str(getattr(transcript, "text", transcript) or "").strip()
    except Exception as exc:
        print(f"[PaperPilot transcription notice]: {exc}")
        return ""


@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    """Universal Paper Pilot ingestion pipeline.

    The endpoint always returns a structured document envelope so the Flutter
    client can render a file preview even when a specialized parser is not
    installed. Media files receive metadata and, when configured, a speech
    transcript. Images are passed through the existing normalized image path.
    """
    try:
        file_bytes = await file.read()
        filename = file.filename or "uploaded_file"
        kind = _file_kind(filename, file.content_type or "")
        ext = os.path.splitext(filename.lower())[1]
        mime = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        extracted_text = ""
        total_pages_detected = 0
        scanned = False
        media = {}
        preview_type = kind

        if kind == "pdf":
            extracted_text, total_pages_detected = extract_massive_pdf_text(file_bytes, max_pages=500)
            scanned = not bool(extracted_text.strip()) or extracted_text.startswith("EXTRACTED STREAM DATA:")
            if scanned:
                visual_text = await _extract_scanned_pdf_with_vision(file_bytes, target_language, max_pages=10)
                if visual_text:
                    extracted_text = visual_text
        elif kind == "image":
            normalized = prepare_image_bytes(file_bytes)
            visual_text = await ask_fast_vision(normalized or file_bytes, filename, target_language)
            extracted_text = visual_text or "Image document received. Visual/OCR inspection could not be completed by the configured vision model."
            if normalized and not visual_text:
                extracted_text += f"\nNormalized image size: {len(normalized)} bytes."
        elif kind == "word":
            if ext == ".docx":
                extracted_text = extract_text_from_docx(file_bytes)
            else:
                extracted_text = inspect_binary_stream(file_bytes)
        elif kind == "spreadsheet":
            if ext in {".xlsx", ".xlsm", ".xls"}:
                extracted_text = extract_text_from_xlsx(file_bytes)
            else:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif kind == "presentation":
            if ext == ".pptx":
                extracted_text, total_pages_detected = extract_text_from_pptx(file_bytes)
            else:
                extracted_text = inspect_binary_stream(file_bytes)
        elif kind == "ebook":
            extracted_text, total_pages_detected = extract_text_from_epub(file_bytes)
        elif kind == "text":
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif kind in {"audio", "video"}:
            media = _media_probe(file_bytes, filename)
            extracted_text = await _transcribe_media(file_bytes, filename)
            if not extracted_text:
                extracted_text = f"{kind.title()} file received. No speech transcript was produced by the configured transcription service."
        else:
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

        type_guidance = {
            "pdf":"Treat this as a document/legal/financial record.",
            "image":"Treat this as a photographed or scanned document/image. Identify visible text, entities, tables and risks.",
            "word":"Preserve headings, clauses, parties, dates and numerical details.",
            "spreadsheet":"Focus on tables, totals, anomalies, formulas represented in the extracted data and important numerical patterns.",
            "presentation":"Treat each slide as a separate information unit and identify key conclusions.",
            "ebook":"Treat chapters as a continuous book and identify themes, entities and important passages.",
            "audio":"Treat the transcript as a conversation/recording. Extract speakers if possible, decisions, commitments, dates and risks.",
            "video":"Treat the transcript as a recording. Extract decisions, actions, dates and important observations; mention that visual scene analysis may require a dedicated vision pass.",
            "text":"Treat this as a text record and preserve its structure.",
        }.get(kind, "Treat this as a general uploaded file and explain what can be reliably inferred.")

        audit_prompt = (
            f"You are Paper Pilot, an expert document analyst and intelligent auditor.\n"
            f"{lang_instruction}\n{type_guidance}\n\n"
            "The goal is to make the user understand exactly what this file is about, in simple language, "
            "while still being precise enough for serious document review.\n\n"
            "GUIDELINES:\n"
            "1. Start with a plain-English 'What this file is about' explanation in 2-4 sentences.\n"
            "2. Then provide: Executive Summary; Key Information; Important Details / Clauses; "
            "People, Organizations & Places; Dates & Deadlines; Financial / Numerical Data; "
            "Risks / Things to Verify; What the User Should Do Next.\n"
            "3. For images, treat the visual evidence report as primary source material and synthesize it carefully.\n"
            "4. Preserve exact names, numbers, dates, phone numbers, addresses and wording when they are readable.\n"
            "5. Explain technical/legal/financial terms in simple words immediately after using them.\n"
            "6. Never invent facts. If a value is unreadable or uncertain, explicitly say so.\n"
            "7. Distinguish clearly between facts visible in the file and reasonable observations/inferences.\n"
            "8. Do not say 'OCR failed', 'content unavailable', or similar when usable evidence is present.\n"
            "9. For posters, notices, invitations, receipts, IDs, forms and photographs, explain the purpose and "
            "the practical meaning rather than forcing them into a financial/legal template.\n"
            "10. End with 3-4 useful follow-up questions under EXPLORE_SUGGESTIONS.\n"
            "At the very end, output exactly: EXPLORE_SUGGESTIONS: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
        )

        analysis_raw = await ask_fast_text(
            f"DOCUMENT FILE: {filename}\nTYPE: {kind}\nMIME: {mime}\nPAGES/SLIDES/CHAPTERS: {total_pages_detected}\n\nSOURCE EVIDENCE / EXTRACTED CONTENT:\n{extracted_text[:120000]}",
            audit_prompt
        )

        suggestions = [
            "What are the primary financial details here?",
            "Are there hidden liabilities or important risks?",
            "What should I verify before relying on this file?"
        ]
        clean_text = analysis_raw
        if "EXPLORE_SUGGESTIONS:" in analysis_raw:
            parts = analysis_raw.split("EXPLORE_SUGGESTIONS:", 1)
            clean_text = parts[0].strip()
            try:
                parsed_sugg = json.loads(parts[1].strip())
                if isinstance(parsed_sugg, list) and parsed_sugg:
                    suggestions = [str(s) for s in parsed_sugg[:4]]
            except Exception:
                pass

        return {
            "status": "success",
            "data": {
                "document_title": filename,
                "file_name": filename,
                "file_type": kind,
                "mime_type": mime,
                "extension": ext,
                "size_bytes": len(file_bytes),
                "pages": total_pages_detected,
                "scanned": scanned,
                "preview": {
                    "type": preview_type,
                    "available": True,
                    "text": extracted_text[:12000],
                    "page_count": total_pages_detected,
                },
                "media": media,
                "actionable_advisory": clean_text,
                "detected_destination": None,
                "suggestions": suggestions,
                "extracted_text": extracted_text[:120000],
            },
            "raw_text": clean_text
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PaperPilot universal audit error]: {e}")
        raise HTTPException(status_code=500, detail=f"Audit error at universal ingestion stage: {str(e)}")

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
# 16A. SARATHI IDENTITY & PERSISTENT CONVERSATIONS
# -------------------------------------------------------------
# Storage policy for the 25K-user design. Raw messages are deliberately kept
# short-lived; older turns are compressed into one conversation memory row.
SARATHI_RAW_MESSAGE_LIMIT = 12
SARATHI_COMPACTION_TRIGGER = 20
SARATHI_COMPACTION_BATCH = 8
SARATHI_MAX_PERSISTED_CHARS = 6000
SARATHI_MAX_MEMORY_CHARS = 7000


async def _sarathi_compact_conversation(conversation_id: str, user_id: str) -> None:
    """Compress older Sarathi turns so database size does not grow linearly forever."""
    if not supabase:
        return
    try:
        rows = (
            supabase.table("sarathi_messages")
            .select("id, role, content, created_at")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(200)
            .execute()
        ).data or []
        if len(rows) <= SARATHI_RAW_MESSAGE_LIMIT + SARATHI_COMPACTION_BATCH:
            return

        older = rows[:-SARATHI_RAW_MESSAGE_LIMIT]
        batch = older[-SARATHI_COMPACTION_BATCH:]
        if not batch:
            return

        previous = (
            supabase.table("sarathi_conversation_memory")
            .select("summary, summarized_message_count")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        ).data or {}
        previous_summary = str(previous.get("summary") or "").strip()

        transcript = "\n".join(
            f"{str(item.get('role') or 'assistant').upper()}: {str(item.get('content') or '')[:3500]}"
            for item in batch
        )
        summary_prompt = f"""Update the compact memory for a travel conversation.
Preserve only durable facts, decisions, preferences, trip details, unresolved questions,
and important context needed to continue naturally. Do not invent anything.
Keep the result under 6500 characters.

Existing memory:
{previous_summary or '(none)'}

New older turns:
{transcript}

Return only the updated memory summary."""
        summary = (await ask_fast_text(
            summary_prompt,
            "You compress conversation history accurately. Preserve facts and uncertainty; never invent details."
        )).strip()
        if not summary:
            return
        summary = summary[:SARATHI_MAX_MEMORY_CHARS]

        total_summarized = int(previous.get("summarized_message_count") or 0) + len(batch)
        supabase.table("sarathi_conversation_memory").upsert({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "summary": summary,
            "summarized_message_count": total_summarized,
            "updated_at": datetime.utcnow().isoformat(),
        }).execute()

        ids = [item.get("id") for item in batch if item.get("id")]
        if ids:
            supabase.table("sarathi_messages").delete().in_("id", ids).eq("user_id", user_id).execute()
    except Exception:
        # Compaction must never break a successful AI response.
        return



# Sarathi never trusts a user_id supplied by the client. The authenticated
# Supabase access token is the source of identity; the server resolves the
# account before reading or writing conversation data.

def _extract_bearer_token(request: Request) -> Optional[str]:
    value = request.headers.get("authorization", "").strip()
    if not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


def _require_sarathi_user(request: Request) -> Dict[str, Any]:
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required for Sarathi.")
    try:
        auth_response = supabase.auth.get_user(token)
        auth_user = getattr(auth_response, "user", None)
        if auth_user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired authentication session.")

        user_id = str(auth_user.id)
        metadata = getattr(auth_user, "user_metadata", {}) or {}
        profile = {}
        try:
            profile_res = (
                supabase.table("users")
                .select("id, display_name, avatar_url, role, preferred_language")
                .eq("id", user_id)
                .maybe_single()
                .execute()
            )
            profile = profile_res.data or {}
        except Exception:
            profile = {}

        display_name = (
            profile.get("display_name")
            or metadata.get("display_name")
            or metadata.get("full_name")
            or metadata.get("name")
            or (getattr(auth_user, "email", "") or "").split("@")[0]
            or "Traveler"
        )
        display_name = str(display_name).strip() or "Traveler"
        first_name = display_name.split()[0]

        return {
            "id": user_id,
            "email": getattr(auth_user, "email", None),
            "display_name": display_name,
            "first_name": first_name,
            "avatar_url": profile.get("avatar_url") or metadata.get("avatar_url"),
            "role": profile.get("role"),
            "preferred_language": profile.get("preferred_language") or metadata.get("preferred_language"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Authentication verification failed: {exc}")


def _sarathi_conversation_title(text: str, fallback: str = "New conversation") -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return fallback
    value = value.replace("\n", " ")
    return value[:72].rstrip() + ("…" if len(value) > 72 else "")


@app.get("/api/v1/sarathi/me")
async def sarathi_me(request: Request):
    user = _require_sarathi_user(request)
    return {"status": "success", "user": user}


@app.get("/api/v1/sarathi/conversations")
async def sarathi_list_conversations(request: Request, include_archived: bool = Query(False)):
    user = _require_sarathi_user(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        query = (
            supabase.table("sarathi_conversations")
            .select("id, title, city, language, archived, created_at, updated_at")
            .eq("user_id", user["id"])
            .order("updated_at", desc=True)
            .limit(100)
        )
        if not include_archived:
            query = query.eq("archived", False)
        result = query.execute()
        return {"status": "success", "conversations": result.data or [], "user": user}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load Sarathi conversations: {exc}")


@app.post("/api/v1/sarathi/conversations")
async def sarathi_create_conversation(request: Request):
    user = _require_sarathi_user(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    city = str(body.get("city") or "Vasai-Virar").strip()
    language = str(body.get("language") or user.get("preferred_language") or "English").strip()
    title = _sarathi_conversation_title(body.get("title"), "New conversation")
    try:
        result = supabase.table("sarathi_conversations").insert({
            "user_id": user["id"],
            "title": title,
            "city": city,
            "language": language,
            "archived": False,
        }).execute()
        conversation = (result.data or [None])[0]
        if not conversation:
            raise HTTPException(status_code=500, detail="Conversation was not created.")
        return {"status": "success", "conversation": conversation, "user": user}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to create Sarathi conversation: {exc}")


@app.get("/api/v1/sarathi/conversations/{conversation_id}")
async def sarathi_get_conversation(conversation_id: str, request: Request):
    user = _require_sarathi_user(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        conv = (
            supabase.table("sarathi_conversations")
            .select("id, title, city, language, archived, created_at, updated_at")
            .eq("id", conversation_id)
            .eq("user_id", user["id"])
            .maybe_single()
            .execute()
        )
        if not conv.data:
            raise HTTPException(status_code=404, detail="Sarathi conversation not found.")
        messages = (
            supabase.table("sarathi_messages")
            .select("id, role, content, created_at")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user["id"])
            .order("created_at", desc=False)
            .limit(SARATHI_RAW_MESSAGE_LIMIT)
            .execute()
        )
        # Conversation memory is optional. A missing table/column, RLS issue,
        # or temporary Supabase problem must not make an otherwise valid
        # conversation return HTTP 500. The raw messages remain usable.
        memory = {}
        try:
            memory = (
                supabase.table("sarathi_conversation_memory")
                .select("summary, summarized_message_count, updated_at")
                .eq("conversation_id", conversation_id)
                .eq("user_id", user["id"])
                .maybe_single()
                .execute()
            ).data or {}
        except Exception as memory_exc:
            print(f"[Sarathi memory warning]: {memory_exc}")

        return {
            "status": "success",
            "conversation": conv.data,
            "messages": messages.data or [],
            "memory_summary": memory.get("summary", ""),
            "summarized_message_count": memory.get("summarized_message_count", 0),
            "storage_policy": {
                "raw_messages_kept": SARATHI_RAW_MESSAGE_LIMIT,
                "older_messages_compacted": True,
            },
            "user": user,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load Sarathi conversation: {exc}")


@app.patch("/api/v1/sarathi/conversations/{conversation_id}")
async def sarathi_update_conversation(conversation_id: str, request: Request):
    user = _require_sarathi_user(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
    except Exception:
        body = {}
    updates: Dict[str, Any] = {}
    if "title" in body:
        updates["title"] = _sarathi_conversation_title(body.get("title"))
    if "archived" in body:
        updates["archived"] = bool(body.get("archived"))
    if not updates:
        return {"status": "success", "message": "No changes requested."}
    try:
        result = (
            supabase.table("sarathi_conversations")
            .update(updates)
            .eq("id", conversation_id)
            .eq("user_id", user["id"])
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Sarathi conversation not found.")
        return {"status": "success", "conversation": result.data[0]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to update Sarathi conversation: {exc}")


@app.delete("/api/v1/sarathi/conversations/{conversation_id}")
async def sarathi_delete_conversation(conversation_id: str, request: Request):
    user = _require_sarathi_user(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        result = (
            supabase.table("sarathi_conversations")
            .delete()
            .eq("id", conversation_id)
            .eq("user_id", user["id"])
            .execute()
        )
        return {"status": "success", "deleted": bool(result.data)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to delete Sarathi conversation: {exc}")

# -------------------------------------------------------------
# 17. CONCIERGE CHAT & LIVE RADAR PIPELINE
# -------------------------------------------------------------
@app.post("/api/v1/explore-chat")
async def explore_chat(request: Request):
    # Sarathi is authenticated for persistent conversations. If the endpoint is
    # called by an older/guest client without a token, keep the legacy response
    # path working instead of breaking unrelated app builds.
    authenticated_user: Optional[Dict[str, Any]] = None
    try:
        authenticated_user = _require_sarathi_user(request)
    except HTTPException as auth_error:
        if auth_error.status_code != 401:
            raise

    city = "Vasai-Virar"
    country = "India"
    question = ""
    target_language = "English"
    chat_history: List[Dict[str, str]] = []
    conversation_id: Optional[str] = None
    saved_home_base = ""
    current_gps = ""
    persist_user_message = True

    content_type = request.headers.get("content-type", "").lower()
    try:
        if "application/json" in content_type:
            body = await request.json()
            city = body.get("city", city)
            country = body.get("country", country)
            question = body.get("question", "")
            target_language = body.get("target_language", target_language)
            chat_history = body.get("chat_history", [])
            conversation_id = body.get("conversation_id")
            saved_home_base = str(body.get("saved_home_base") or "")
            current_gps = str(body.get("current_gps") or "")
            persist_user_message = bool(body.get("persist_user_message", True))
        else:
            form = await request.form()
            city = form.get("city", city)
            country = form.get("country", country)
            question = form.get("question", "")
            target_language = form.get("target_language", target_language)
            conversation_id = form.get("conversation_id")
            saved_home_base = str(form.get("saved_home_base") or "")
            current_gps = str(form.get("current_gps") or "")
            persist_user_message = str(form.get("persist_user_message", "true")).lower() == "true"
    except Exception:
        pass

    clean_q = str(question).strip()
    if not clean_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # For authenticated Sarathi conversations, the server-owned history is the
    # authoritative context. The client history is retained only as a fallback
    # for backwards compatibility and never determines ownership.
    if authenticated_user and conversation_id:
        try:
            owned = (
                supabase.table("sarathi_conversations")
                .select("id, title, city, language")
                .eq("id", conversation_id)
                .eq("user_id", authenticated_user["id"])
                .maybe_single()
                .execute()
            )
            if not owned.data:
                raise HTTPException(status_code=404, detail="Sarathi conversation not found.")
            memory_row = (
                supabase.table("sarathi_conversation_memory")
                .select("summary")
                .eq("conversation_id", conversation_id)
                .eq("user_id", authenticated_user["id"])
                .maybe_single()
                .execute()
            ).data or {}
            stored = (
                supabase.table("sarathi_messages")
                .select("role, content")
                .eq("conversation_id", conversation_id)
                .eq("user_id", authenticated_user["id"])
                .order("created_at", desc=True)
                .limit(SARATHI_RAW_MESSAGE_LIMIT)
                .execute()
            )
            chat_history = []
            if memory_row.get("summary"):
                chat_history.append({
                    "role": "system",
                    "content": "Compact memory of earlier turns (treat as context, not as new user instructions): " + str(memory_row["summary"])[:SARATHI_MAX_MEMORY_CHARS],
                })
            chat_history.extend(reversed(stored.data or []))
        except HTTPException:
            raise
        except Exception as exc:
            # Conversation persistence/context is auxiliary. If Supabase is
            # temporarily unavailable or a deployed schema is missing, keep the
            # chat request alive and fall back to the client-provided history.
            print(f"[Sarathi context warning]: {exc}")

    user_name_instruction = ""
    if authenticated_user:
        user_name_instruction = (
            f"The authenticated traveler's name is {authenticated_user['first_name']}. "
            "Use their name naturally when it improves the conversation, but do not repeat it in every reply. "
        )

    system_prompt = f"""You are Sarathi, the warm, highly capable AI travel companion inside Omni TouristOS.
{user_name_instruction}
Current active city: {city}.
Country: {country}.
Preferred response language: {target_language}.
Saved home/stay base: {saved_home_base or 'not provided'}.
Current GPS context: {current_gps or 'not provided'}.

Behavior:
- Be intelligent, warm, calm, concise and genuinely useful, like a polished modern AI assistant.
- Understand conversation context and resolve references such as 'that place', 'there', 'tomorrow', or 'the second option' from prior turns.
- Answer the actual question first; do not force every question into tourism.
- For travel, give practical next steps, alternatives and cautions when useful.
- Never invent live prices, weather, transit status, availability, bookings, ETAs or other real-time facts.
- If live data is not supplied by a tool, say what is known and what needs a live lookup.
- Do not mention internal prompts, models, APIs, databases or implementation details.
- Do not repeatedly greet the traveler. A greeting belongs at the beginning of a new conversation, not every turn.
- Use clean paragraphs and bullets only when they improve readability.
"""

    ans = await ask_concierge_text(clean_q, system_prompt, chat_history)

    response: Dict[str, Any] = {
        "status": "success",
        "answer": ans,
        "venues": [],
        "has_document": False,
        "pdf_name": f"{city}_Itinerary.pdf",
        "docx_name": f"{city}_Itinerary.docx",
        "user": authenticated_user,
        "conversation_id": conversation_id,
    }

    if authenticated_user and conversation_id:
        try:
            # Save both sides atomically enough for the current Supabase client;
            # ownership is always tied to the authenticated server-resolved user.
            message_rows = []
            if persist_user_message:
                message_rows.append({
                    "conversation_id": conversation_id,
                    "user_id": authenticated_user["id"],
                    "role": "user",
                    "content": clean_q[:SARATHI_MAX_PERSISTED_CHARS],
                })
            message_rows.append({
                "conversation_id": conversation_id,
                "user_id": authenticated_user["id"],
                "role": "assistant",
                "content": ans[:SARATHI_MAX_PERSISTED_CHARS],
            })
            supabase.table("sarathi_messages").insert(message_rows).execute()
            # First user message becomes the default ChatGPT-style title.
            conv = (
                supabase.table("sarathi_conversations")
                .select("title")
                .eq("id", conversation_id)
                .eq("user_id", authenticated_user["id"])
                .maybe_single()
                .execute()
            )
            if conv.data and conv.data.get("title") in (None, "", "New conversation"):
                supabase.table("sarathi_conversations").update({
                    "title": _sarathi_conversation_title(clean_q),
                    "updated_at": datetime.utcnow().isoformat(),
                }).eq("id", conversation_id).eq("user_id", authenticated_user["id"]).execute()
            else:
                supabase.table("sarathi_conversations").update({
                    "updated_at": datetime.utcnow().isoformat(),
                }).eq("id", conversation_id).eq("user_id", authenticated_user["id"]).execute()

            # Compact only at a safe threshold so normal messages do not incur an
            # extra AI call. The raw database footprint therefore stays bounded.
            message_count = (
                supabase.table("sarathi_messages")
                .select("id", count="exact", head=True)
                .eq("conversation_id", conversation_id)
                .eq("user_id", authenticated_user["id"])
                .execute()
            ).count or 0
            if message_count >= SARATHI_COMPACTION_TRIGGER:
                await _sarathi_compact_conversation(conversation_id, authenticated_user["id"])
        except Exception as exc:
            # The AI answer is still valid, but tell the client persistence failed
            # so the UI can avoid pretending the message was safely stored.
            response["persistence_warning"] = str(exc)

    return response

# -------------------------------------------------------------
# 17B. GEM SCOUT — COMMUNITY DISCOVERY CANDIDATES
# -------------------------------------------------------------
@app.post("/api/v1/scout/community/run")
async def run_community_gem_scout(
    request: Request,
):
    """Run one Community Gem Scout job and persist candidates in Supabase.

    Security: set GEM_SCOUT_KEY in Render and send it as X-Gem-Scout-Key.
    The Scout stores open-data-derived candidate fields and only the Google
    place ID from live Google verification.
    """
    configured_key = os.environ.get("GEM_SCOUT_KEY", "").strip()
    supplied_key = request.headers.get("X-Gem-Scout-Key", "").strip()
    if not configured_key or supplied_key != configured_key:
        raise HTTPException(status_code=403, detail="Gem Scout authorization failed.")
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase is not configured.")

    try:
        body = await request.json()
        city = str(body.get("city") or "").strip()
        category = str(body.get("category") or "Food").strip()
        quantity = int(body.get("quantity") or 10)
        verify_google = bool(body.get("verify_google", True))
        if not city:
            raise HTTPException(status_code=422, detail="city is required")
        if quantity < 1 or quantity > 50:
            raise HTTPException(status_code=422, detail="quantity must be between 1 and 50")

        result = await GemScout(supabase).run_community_scout(
            city=city,
            category=category,
            quantity=quantity,
            verify_google=verify_google,
        )
        return {
            "status": "success",
            "city": result.city,
            "requested_category": result.requested_category,
            "canonical_category": result.canonical_category,
            "requested_quantity": result.requested_quantity,
            "scanned": result.scanned,
            "inserted": result.inserted,
            "updated": result.updated,
            "source_counts": result.source_counts,
            "candidates": [c.__dict__ for c in result.candidates],
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[Gem Scout Error] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/scout/community/candidates")
async def get_community_gem_scout_candidates(
    city: str = Query(""),
    category: str = Query(""),
    status: str = Query(""),
    limit: int = Query(50, ge=1, le=100),
):
    """Read Scout candidates for the future Admin Console."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase is not configured.")
    try:
        query = supabase.table("gem_scout_candidates").select("*")
        if city.strip():
            query = query.ilike("city", f"%{city.strip()}%")
        if category.strip():
            query = query.eq("category", category.strip())
        if status.strip():
            query = query.eq("status", status.strip())
        response = query.order("confidence", desc=True).limit(limit).execute()
        return {"status": "success", "candidates": response.data or []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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

COMMUNITY_GOOGLE_CATEGORY_QUERIES = {
    "Pharmacy / Chemist": "pharmacy chemist",
    "Barber & Salon": "barber salon",
    "Kirana & Essentials": "grocery store",
    "General Store / Supermarket": "general store supermarket",
    "Fruits & Vegetables": "fruit vegetable market",
    "Meat / Fish": "meat fish market",
    "Ice Cream & Dairy": "ice cream dairy",
    "Cold Storage & Meat": "meat cold storage",
    "Bakery & Sweets": "bakery sweets",
    "Indo-Chinese & Snacks": "indo chinese snacks",
    "Chai & Quick Bites": "cafe tea snacks",
    "Bar & Restaurant": "restaurant",
    "Diner & Seafood": "seafood restaurant diner",
    "Market, Bazaar & Mall": "market bazaar mall",
    "Movie Cinema & Theater": "cinema theater",
    "Picnic Spot & Landscape": "park picnic spot",
    "Resort & Farmhouse": "resort farmhouse",
    "Heritage & Sight": "tourist attraction heritage",
    "Clothing & Fashion": "clothing fashion store",
    "Electronics & Mobile": "electronics mobile phone store",
    "Stationery & Gifts": "stationery gift shop",
    "Hardware & Home": "hardware home improvement",
    "Beauty & Spa": "beauty spa",
    "Tailor & Laundry": "tailor laundry",
    "Car / Bike Service": "car bike service repair",
    "Clinic / Doctor": "clinic doctor",
    "Dental / Optical": "dentist optical",
    "Hotel & Stay": "hotel",
    "Guest House / Homestay": "guest house homestay",
    "Travel / Car Rental": "car rental travel agency",
    "Beach / Park / Sports": "beach park sports",
    "Temple / Place of Worship": "temple place of worship",
    "Event / Wedding Venue": "wedding event venue",
}

def _community_google_category(place: Dict[str, Any], requested: str = "All") -> str:
    raw = " ".join([
        str(place.get("primaryType") or ""),
        str((place.get("primaryTypeDisplayName") or {}).get("text") if isinstance(place.get("primaryTypeDisplayName"), dict) else place.get("primaryTypeDisplayName") or ""),
        " ".join([str(x) for x in (place.get("types") or [])]),
    ]).lower()
    if any(x in raw for x in ["pharmacy", "drugstore"]): return "Pharmacy / Chemist"
    if any(x in raw for x in ["hair_care", "barber", "beauty_salon"]): return "Barber & Salon"
    if any(x in raw for x in ["grocery", "supermarket", "convenience_store"]): return "Kirana & Essentials"
    if "ice_cream" in raw: return "Ice Cream & Dairy"
    if any(x in raw for x in ["butcher", "meat"]): return "Cold Storage & Meat"
    if "bakery" in raw: return "Bakery & Sweets"
    if any(x in raw for x in ["chinese_restaurant", "snack", "fast_food"]): return "Indo-Chinese & Snacks"
    if any(x in raw for x in ["cafe", "coffee_shop"]): return "Chai & Quick Bites"
    if any(x in raw for x in ["seafood", "seafood_restaurant"]): return "Diner & Seafood"
    if any(x in raw for x in ["restaurant", "meal_takeaway", "food"]): return "Bar & Restaurant"
    if any(x in raw for x in ["shopping_mall", "market"]): return "Market, Bazaar & Mall"
    if any(x in raw for x in ["movie_theater", "cinema"]): return "Movie Cinema & Theater"
    if any(x in raw for x in ["park", "tourist_attraction", "historical_landmark", "beach"]): return "Picnic Spot & Landscape"
    if any(x in raw for x in ["resort", "hotel"]): return "Resort & Farmhouse"
    if any(x in raw for x in ["lodging", "guest_house"]): return "Hotel & Stay"
    if any(x in raw for x in ["temple", "church", "mosque", "hindu_temple", "place_of_worship"]): return "Heritage & Sight"
    return "General"

def _community_google_place_to_row(place: Dict[str, Any], city: str, category: str = "All") -> Dict[str, Any]:
    display = place.get("displayName") or {}
    loc = place.get("location") or {}
    photos = [str(x.get("name")) for x in (place.get("photos") or []) if isinstance(x, dict) and x.get("name")]
    image_urls = _google_photo_proxy_urls(photos, 1200)
    row = {
        "id": str(place.get("id") or "google-unknown"),
        "source": "GOOGLE",
        "google_place_id": str(place.get("id") or ""),
        "name": str(display.get("text") or "Local Place"),
        "category": _community_google_category(place, category),
        "subcategory": str(place.get("primaryTypeDisplayName") or ""),
        "address": str(place.get("formattedAddress") or ""),
        "city": city,
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
        "rating": place.get("rating"),
        "review_count": place.get("userRatingCount"),
        "open_now": (place.get("regularOpeningHours") or {}).get("openNow"),
        "hours": (place.get("regularOpeningHours") or {}).get("weekdayDescriptions") or [],
        "maps_url": str(place.get("googleMapsUri") or ""),
        "website_url": str(place.get("websiteUri") or ""),
        "contact_phone": str(place.get("internationalPhoneNumber") or place.get("nationalPhoneNumber") or ""),
        "image_url": image_urls[0] if image_urls else "",
        "photos": image_urls,
        "google_photo_names": photos,
        "upvotes": 0,
        "community_endorsements": 0,
        "community_tags": [],
        "must_try_tip": "",
        "contributor_name": "Google Places",
        "community_notice": None,
        "providers": {},
        "booking": {},
    }
    return row

async def _search_open_community_places(
    city: str, query_text: str, category: str = "All", limit: int = 10
) -> List[Dict[str, Any]]:
    clean_city, clean_query = city.strip(), query_text.strip()
    if not clean_city or not clean_query:
        return []
    destination = await _resolve_destination_with_open_data(clean_city, "", "India")
    if not destination:
        return []
    lat, lng = float(destination["latitude"]), float(destination["longitude"])
    safe_regex = re.escape(clean_query)
    query = f'''
[out:json][timeout:20];
(
  nwr(around:15000,{lat},{lng})["name"~"{safe_regex}",i];
);
out center tags;
'''
    try:
        async with _overpass_semaphore:
            async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=8.0)) as client:
                response = await client.post(
                    OVERPASS_API_URL, data={"data": query},
                    headers={"User-Agent": OPEN_DATA_USER_AGENT, "Accept": "application/json"},
                )
        if response.status_code != 200:
            print(f"[Community Open Data Notice] Overpass HTTP {response.status_code}: {response.text[:500]}")
            return []
        elements = (response.json() or {}).get("elements") or []
        rows, seen = [], set()
        def tag(tags, key): return str(tags.get(key) or "").strip()
        for element in elements:
            tags = element.get("tags") or {}
            name = tag(tags, "name")
            if not name or name.lower() in seen:
                continue
            if element.get("lat") is not None and element.get("lon") is not None:
                item_lat, item_lng = float(element["lat"]), float(element["lon"])
            else:
                center = element.get("center") or {}
                if center.get("lat") is None or center.get("lon") is None:
                    continue
                item_lat, item_lng = float(center["lat"]), float(center["lon"])
            amenity, shop = tag(tags, "amenity"), tag(tags, "shop")
            tourism, leisure, healthcare = tag(tags, "tourism"), tag(tags, "leisure"), tag(tags, "healthcare")
            category_text = " ".join(x for x in [amenity, shop, tourism, leisure, healthcare] if x)
            mapped_category = _community_google_category(
                {"primaryType": category_text, "primaryTypeDisplayName": {"text": category_text}, "types": [amenity, shop, tourism, leisure, healthcare]}, "All"
            )
            if category != "All" and mapped_category != category:
                continue
            address = ", ".join(x for x in [tag(tags,"addr:housenumber"), tag(tags,"addr:street"), tag(tags,"addr:suburb"), tag(tags,"addr:city"), tag(tags,"addr:postcode")] if x) or clean_city
            osm_id = f"osm:{element.get('type','')}/{element.get('id','')}"
            maps_url = f"https://www.openstreetmap.org/?mlat={item_lat}&mlon={item_lng}#map=17/{item_lat}/{item_lng}"
            image = tag(tags, "image")
            rows.append({
                "id": osm_id, "source": "OPENSTREETMAP", "provider": "OpenStreetMap", "google_place_id": "",
                "name": name, "category": mapped_category, "subcategory": category_text, "address": address, "city": clean_city,
                "latitude": item_lat, "longitude": item_lng, "rating": None, "review_count": None, "open_now": None,
                "hours": [tag(tags,"opening_hours")] if tag(tags,"opening_hours") else [], "maps_url": maps_url,
                "website_url": tag(tags,"website") or tag(tags,"contact:website"), "contact_phone": tag(tags,"phone") or tag(tags,"contact:phone"),
                "image_url": image, "photos": [image] if image else [], "google_photo_names": [], "upvotes": 0, "community_endorsements": 0,
                "community_tags": [], "must_try_tip": "", "contributor_name": "OpenStreetMap",
                "community_notice": "OpenStreetMap result — verify current details before submitting.", "providers": {}, "booking": {},
                "data_state": "VERIFIED_OPEN_DATA", "attribution_required": "OpenStreetMap",
            })
            seen.add(name.lower())
            if len(rows) >= max(1, min(limit,20)): break
        return rows
    except Exception as exc:
        print(f"[Community Open Data Notice] {exc}")
        return []

@app.get("/api/v1/community/place-discovery/search")
async def discover_community_place(
    city: str = Query(...), q: str = Query(...), category: str = Query("All"), limit: int = Query(10, ge=1, le=20)
):
    clean_city, clean_q = city.strip(), q.strip()
    requested_category = category.strip() or "All"
    if not clean_q:
        return {"status":"success","provider":"NONE","google_live":False,"places":[]}
    category_hint = COMMUNITY_GOOGLE_CATEGORY_QUERIES.get(requested_category, "")
    google_query = " ".join(x for x in [clean_q, category_hint, clean_city] if x).strip()
    google_places = await _google_text_search(google_query, max_result_count=limit)
    if google_places:
        rows = [_community_google_place_to_row(p, clean_city, requested_category) for p in google_places]
        for row in rows:
            row["provider"] = "Google Places"; row["data_state"] = "VERIFIED"
        return {"status":"success","provider":"GOOGLE","google_live":True,"city":clean_city,"places":rows}
    open_rows = await _search_open_community_places(clean_city, clean_q, requested_category, limit)
    return {"status":"success","provider":"OPENSTREETMAP" if open_rows else "NONE","google_live":False,"city":clean_city,"places":open_rows}


def _merge_community_place_with_google(community: Dict[str, Any], google: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = dict(community)
    row["source"] = "GOOGLE+COMMUNITY" if google else "COMMUNITY"
    if google:
        for key in ["google_place_id", "rating", "review_count", "open_now", "hours", "website_url", "photos"]:
            if not row.get(key): row[key] = google.get(key)
        if not row.get("maps_url"): row["maps_url"] = google.get("maps_url", "")
        if not row.get("image_url"): row["image_url"] = google.get("image_url", "")
        if not row.get("latitude"): row["latitude"] = google.get("latitude")
        if not row.get("longitude"): row["longitude"] = google.get("longitude")
        row["google_place_id"] = row.get("google_place_id") or google.get("google_place_id")
    row["community_endorsements"] = int(row.get("upvotes") or row.get("community_endorsements") or 0)
    row["community_notice"] = row.get("community_notice") or row.get("daily_notice") or None
    row["providers"] = row.get("providers") or {
        "zomato": {"url": row.get("zomato_url") or ""},
        "swiggy": {"url": row.get("swiggy_url") or ""},
    }
    row["booking"] = row.get("booking") or {"url": row.get("booking_url") or "", "provider": row.get("booking_provider") or ""}
    return row

@app.get("/api/v1/community/places/search")
async def search_community_places(
    city: str = Query(...),
    q: str = Query(""),
    category: str = Query("All"),
    lat: Optional[float] = Query(None),
    lng: Optional[float] = Query(None),
    limit: int = Query(30, ge=1, le=50),
):
    """Unified Community Gems feed: real Google Places + Supabase community listings.
    Google keys remain server-side. No fabricated fallback places are returned.
    """
    clean_city = city.strip()
    requested_category = category.strip() or "All"
    search_term = q.strip()
    google_query = " ".join(x for x in [search_term, COMMUNITY_GOOGLE_CATEGORY_QUERIES.get(requested_category, "places"), clean_city] if x).strip()
    if not search_term and requested_category == "All":
        google_query = f"popular local places businesses restaurants shops attractions in {clean_city}"

    google_places = await _google_text_search(google_query, max_result_count=min(limit, 20))
    google_rows = [_community_google_place_to_row(p, clean_city, requested_category) for p in google_places]
    if not google_rows and search_term:
        google_rows = await _search_open_community_places(clean_city, search_term, requested_category, min(limit, 20))

    community_rows: List[Dict[str, Any]] = []
    if supabase:
        try:
            query = supabase.table("community_places").select("*").eq("city", clean_city)
            if requested_category != "All": query = query.eq("category", requested_category)
            if search_term:
                query = query.or_(f"name.ilike.%{search_term}%,address.ilike.%{search_term}%,category.ilike.%{search_term}%,description.ilike.%{search_term}%")
            res = query.order("upvotes", desc=True).limit(limit).execute()
            community_rows = res.data or []
        except Exception as exc:
            print(f"[Community Gems Supabase search notice]: {exc}")

    # Match community records to Google primarily by place_id, then normalized name/address.
    google_by_id = {str(r.get("google_place_id")): r for r in google_rows if r.get("google_place_id")}
    google_by_key = {}
    for r in google_rows:
        key = (str(r.get("name") or "").lower().strip(), str(r.get("address") or "").lower().strip())
        google_by_key[key] = r

    merged: List[Dict[str, Any]] = []
    used_google = set()
    for c in community_rows:
        gid = str(c.get("google_place_id") or "")
        key = (str(c.get("name") or "").lower().strip(), str(c.get("address") or "").lower().strip())
        g = google_by_id.get(gid) or google_by_key.get(key)
        if g: used_google.add(g.get("id"))
        merged.append(_merge_community_place_with_google(c, g))
    for g in google_rows:
        if g.get("id") not in used_google:
            merged.append(g)

    # Optional nearest-first ordering when the client has a usable position.
    if lat is not None and lng is not None:
        def distance_key(row):
            try:
                dlat = float(row.get("latitude")) - lat
                dlng = float(row.get("longitude")) - lng
                return dlat * dlat + dlng * dlng
            except Exception:
                return 10**9
        merged.sort(key=distance_key)
    else:
        merged.sort(key=lambda r: (-(int(r.get("upvotes") or 0)), -(float(r.get("rating") or 0))))

    return {"status": "success", "city": clean_city, "places": merged[:limit], "google_live": bool(google_places)}



@app.get("/api/v1/community/places/nearby")
async def search_community_places_nearby(
    lat: float = Query(...),
    lng: float = Query(...),
    category: str = Query("All"),
    limit: int = Query(30, ge=1, le=50),
):
    """Live nearby Community Gems discovery using the same Google provider.
    Location is only used when the user explicitly taps Nearby/Live View.
    """
    requested_category = category.strip() or "All"
    category_query = COMMUNITY_GOOGLE_CATEGORY_QUERIES.get(requested_category, "places")
    query = f"popular local places businesses restaurants shops attractions" if requested_category == "All" else category_query
    google_places = await _google_text_search(
        query,
        max_result_count=min(limit, 20),
        latitude=lat,
        longitude=lng,
        radius_meters=15000,
    )
    rows = [_community_google_place_to_row(p, "Nearby", requested_category) for p in google_places]
    # When a specific category is selected, retain only correctly classified results.
    if requested_category != "All":
        rows = [r for r in rows if r.get("category") == requested_category]
    return {
        "status": "success",
        "provider": "GOOGLE",
        "google_live": bool(rows),
        "places": rows[:limit],
    }

@app.post("/api/v1/gems/create")
async def create_gem(request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        creator_id = body.get("creator_id")

        # Do not query users.is_banned here: that column is not present in the
        # deployed Supabase users table and causes PostgreSQL 42703 on every
        # gem submission. Account moderation must use a real, migrated column
        # or a verified auth/role policy; this endpoint only inserts the gem.
        title = str(body.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Gem name is required.")

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
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Community Gems create error]: {e}")
        raise HTTPException(status_code=500, detail="Could not save the community gem. Check the gems table schema and backend logs.") from e

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


@app.post("/api/v1/community/private-message")
async def send_moderated_private_message(request: Request):
    """Server-authoritative 1-to-1 message path supervised by Supervisor Bot."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        sender_id = str(body.get("sender_id") or "").strip()
        receiver_id = str(body.get("receiver_id") or "").strip()
        sender_name = str(body.get("sender_name") or "Local Scout").strip()[:120]
        sender_avatar_url = body.get("sender_avatar_url")
        message = str(body.get("message") or "").strip()

        if not sender_id or not receiver_id or not message:
            raise HTTPException(status_code=400, detail="sender_id, receiver_id and message are required.")
        if len(message) > 500:
            raise HTTPException(status_code=400, detail="Message is too long. Please keep it concise.")

        # Account enforcement before moderation.
        try:
            user_row = supabase.table("users").select("is_banned, muted_until").eq("id", sender_id).single().execute()
            if user_row.data:
                if user_row.data.get("is_banned"):
                    raise HTTPException(status_code=403, detail="Account banned for community violations.")
                muted_until = user_row.data.get("muted_until")
                if muted_until:
                    try:
                        muted_dt = datetime.fromisoformat(str(muted_until).replace("Z", "+00:00"))
                        if muted_dt > datetime.now(muted_dt.tzinfo):
                            raise HTTPException(status_code=403, detail="Account temporarily muted.")
                    except ValueError:
                        pass
        except HTTPException:
            raise
        except Exception:
            # Preserve compatibility with installations where the users row is incomplete.
            pass

        violation = evaluate_supervisor_bot_flag(message)
        if violation:
            try:
                supabase.table("community_reports").insert({
                    "reporter_id": None,
                    "offender_id": sender_id,
                    "community_id": "private",
                    "reason": "Supervisor Bot Auto-Flag",
                    "details": f"Private message flagged: '{message}' | Issue: {violation}",
                    "status": "pending",
                }).execute()
            except Exception:
                pass
            raise HTTPException(status_code=403, detail=f"Message blocked by Supervisor Bot: {violation}.")

        inserted = supabase.table("direct_messages").insert({
            "sender_id": sender_id,
            "sender_name": sender_name,
            "sender_avatar_url": sender_avatar_url,
            "receiver_id": receiver_id,
            "message": message,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

        return {"status": "success", "message": inserted.data[0] if inserted.data else {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/community/group-message")
async def send_moderated_group_message(request: Request):
    """Server-authoritative Travel Crew message path supervised by Supervisor Bot."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    try:
        body = await request.json()
        group_id = str(body.get("group_id") or "").strip()
        sender_id = str(body.get("sender_id") or "").strip()
        sender_name = str(body.get("sender_name") or "Local Scout").strip()[:120]
        sender_avatar_url = body.get("sender_avatar_url")
        message = str(body.get("message") or "").strip()

        if not group_id or not sender_id or not message:
            raise HTTPException(status_code=400, detail="group_id, sender_id and message are required.")
        if len(message) > 500:
            raise HTTPException(status_code=400, detail="Message is too long. Please keep it concise.")

        try:
            user_row = supabase.table("users").select("is_banned, muted_until").eq("id", sender_id).single().execute()
            if user_row.data and user_row.data.get("is_banned"):
                raise HTTPException(status_code=403, detail="Account banned for community violations.")
        except HTTPException:
            raise
        except Exception:
            pass

        violation = evaluate_supervisor_bot_flag(message)
        if violation:
            try:
                supabase.table("community_reports").insert({
                    "reporter_id": None,
                    "offender_id": sender_id,
                    "community_id": group_id,
                    "reason": "Supervisor Bot Auto-Flag",
                    "details": f"Group message flagged: '{message}' | Issue: {violation}",
                    "status": "pending",
                }).execute()
            except Exception:
                pass
            raise HTTPException(status_code=403, detail=f"Message blocked by Supervisor Bot: {violation}.")

        inserted = supabase.table("chat_group_messages").insert({
            "group_id": group_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "sender_avatar_url": sender_avatar_url,
            "message": message,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

        return {"status": "success", "message": inserted.data[0] if inserted.data else {}}
    except HTTPException:
        raise
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
# Legacy constants retained because the older helper functions remain in this
# file for compatibility, but Omni Rail now uses RailRadar for its live API.
RAILWAY_REFERENCE_LABEL = "RailRadar API"
RAILWAY_TIMETABLE_VERSIONS = {
    "W": "RailRadar",
    "C": "RailRadar",
    "H": "RailRadar",
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


class _RailHTMLTableParser(HTMLParser):
    """Small stdlib HTML table extractor used when BeautifulSoup is unavailable."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._table: Optional[List[List[str]]] = None
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._cell_tag: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._table is None:
                self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._cell_tag = tag

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            value = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(value)
            self._cell = None
            self._cell_tag = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _rail_html_tables(html: str) -> List[List[List[str]]]:
    """Return HTML tables as simple row/cell lists with a stdlib fallback."""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            output: List[List[List[str]]] = []
            for table in soup.find_all("table"):
                rows: List[List[str]] = []
                for row in table.find_all("tr"):
                    cells = [
                        re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                        for cell in row.find_all(["th", "td"])
                    ]
                    if cells:
                        rows.append(cells)
                if rows:
                    output.append(rows)
            return output
        except Exception:
            pass

    parser = _RailHTMLTableParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return []
    return parser.tables


def _rail_header_score(cell: str) -> int:
    norm = _rail_normalize(cell)
    if norm in {"trainno", "trainnumber", "train"}:
        return 5
    if "trainno" in norm or "trainnumber" in norm:
        return 4
    if norm in {"speed", "coaches", "duration", "originatingfrom", "endingat"}:
        return 2
    return 0


def _rail_station_header_index(headers: List[str], station: str) -> Optional[int]:
    aliases = [
        _rail_station_display_name(station),
        str(station or ""),
        _rail_legacy_station_param(station),
        _rail_slug_from_station(station),
    ]
    normalized_aliases = [_rail_normalize(x) for x in aliases if x]
    for idx, header in enumerate(headers):
        norm = _rail_normalize(header)
        if not norm:
            continue
        if any(alias and (alias == norm or alias in norm or norm in alias) for alias in normalized_aliases):
            return idx
    return None


def _rail_merge_header_rows(rows: List[List[str]], start: int, end: int) -> List[str]:
    """Merge a multi-row table header while preserving column positions."""
    width = max((len(rows[i]) for i in range(start, end + 1)), default=0)
    merged = [""] * width
    for i in range(start, end + 1):
        row = rows[i]
        for col in range(width):
            value = row[col].strip() if col < len(row) else ""
            if value and value not in {"-", "—", "–"}:
                if not merged[col]:
                    merged[col] = value
                elif _rail_normalize(value) != _rail_normalize(merged[col]):
                    merged[col] = f"{merged[col]} {value}".strip()
    return merged


def _rail_train_from_row(
    values: List[str],
    line: str,
    requested_from_name: str,
    requested_to_name: str,
    train_idx: Optional[int],
    speed_idx: Optional[int],
    coach_idx: Optional[int],
    origin_idx: Optional[int],
    ending_idx: Optional[int],
    duration_idx: Optional[int],
    from_time_idx: int,
    to_time_idx: Optional[int],
    source_url: str,
) -> Optional[Dict[str, Any]]:
    def value_at(index: Optional[int]) -> str:
        return values[index].strip() if index is not None and index < len(values) else ""

    train_cell = value_at(train_idx)
    dep_raw = value_at(from_time_idx)
    arr_raw = value_at(to_time_idx)
    if not dep_raw or dep_raw in {"—", "-", "–"}:
        return None

    dep_min = _rail_time_to_minutes(dep_raw)
    if dep_min is None:
        return None
    arr_min = _rail_time_to_minutes(arr_raw) if arr_raw else None
    if arr_min is not None and arr_min < dep_min:
        arr_min += 1440

    speed = value_at(speed_idx)
    coaches = value_at(coach_idx) or None
    origin = value_at(origin_idx) or None
    ending = value_at(ending_idx) or None
    duration = value_at(duration_idx) or None

    service_type, service_label = _rail_detect_service(train_cell, speed)
    cleaned_train_no = _rail_clean_train_no(train_cell)
    if cleaned_train_no == "—" and train_cell:
        cleaned_train_no = train_cell

    return {
        "time": dep_raw,
        "departure_time": dep_raw,
        "arrival_time": arr_raw if arr_raw not in {"", "—", "-", "–"} else None,
        "timestamp_minutes": dep_min,
        "arrival_minutes": arr_min,
        "train_no": cleaned_train_no,
        "name": f"{origin or requested_from_name} → {ending or requested_to_name}",
        "service_type": service_type,
        "service": service_label,
        "category": service_label,
        "platform": None,
        "platform_note": "Platform not published in this timetable response.",
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
        "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(line, "Unknown"),
    }


def _rail_parse_timetable_html(
    html: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse Mumbai Lifeline tables, including multi-row headers and layout changes."""
    tables = _rail_html_tables(html)
    if not tables:
        return []

    requested_from_name = _rail_station_display_name(from_station)
    requested_to_name = _rail_station_display_name(to_station)
    trains: List[Dict[str, Any]] = []

    for rows in tables:
        if not rows:
            continue

        # Locate the real timetable header. It may be split over two rows.
        header_row_index: Optional[int] = None
        station_row_index: Optional[int] = None
        header_end = -1

        for row_index, row in enumerate(rows[:80]):
            score = sum(_rail_header_score(cell) for cell in row)
            if score < 4:
                continue
            header_row_index = row_index
            if _rail_station_header_index(row, from_station) is not None:
                station_row_index = row_index
                header_end = row_index
                break
            for look_ahead in range(row_index + 1, min(row_index + 4, len(rows))):
                if _rail_station_header_index(rows[look_ahead], from_station) is not None:
                    station_row_index = look_ahead
                    header_end = look_ahead
                    break
            break

        if header_row_index is None:
            # Some legacy tables omit the literal Train No heading. Find a row
            # that contains the requested station and several time-like cells.
            for row_index, row in enumerate(rows[:80]):
                if _rail_station_header_index(row, from_station) is None:
                    continue
                time_count = sum(1 for cell in row if _rail_time_to_minutes(cell) is not None)
                if time_count >= 1:
                    header_row_index = max(0, row_index - 1)
                    station_row_index = row_index
                    header_end = row_index
                    break

        if header_row_index is None:
            continue

        if station_row_index is None:
            station_row_index = header_row_index
        header_cells = _rail_merge_header_rows(rows, header_row_index, header_end)

        train_idx = _rail_find_header(header_cells, ["Train No", "Train Number", "Train"])
        speed_idx = _rail_find_header(header_cells, ["Speed"])
        coach_idx = _rail_find_header(header_cells, ["Coaches"])
        origin_idx = _rail_find_header(header_cells, ["Originating From", "Origin"])
        ending_idx = _rail_find_header(header_cells, ["Ending At", "Destination"])
        duration_idx = _rail_find_header(header_cells, ["Duration"])
        from_time_idx = _rail_station_header_index(header_cells, from_station)
        to_time_idx = _rail_station_header_index(header_cells, to_station)

        # If the merged header failed, use the station-specific header row.
        if from_time_idx is None:
            from_time_idx = _rail_station_header_index(rows[station_row_index], from_station)
        if to_time_idx is None:
            to_time_idx = _rail_station_header_index(rows[station_row_index], to_station)

        if from_time_idx is None:
            continue

        data_start = max(header_row_index, station_row_index) + 1
        table_trains: List[Dict[str, Any]] = []
        for row in rows[data_start:]:
            values = [re.sub(r"\s+", " ", str(v or "")).strip() for v in row]
            if not values:
                continue
            parsed = _rail_train_from_row(
                values=values,
                line=line,
                requested_from_name=requested_from_name,
                requested_to_name=requested_to_name,
                train_idx=train_idx,
                speed_idx=speed_idx,
                coach_idx=coach_idx,
                origin_idx=origin_idx,
                ending_idx=ending_idx,
                duration_idx=duration_idx,
                from_time_idx=from_time_idx,
                to_time_idx=to_time_idx,
                source_url=source_url,
            )
            if parsed:
                table_trains.append(parsed)

        if table_trains:
            trains.extend(table_trains)
            # Prefer the first real timetable table; navigation tables should
            # never prevent later data from being considered, but once a table
            # has produced valid rows it is already authoritative for this URL.
            break

    unique: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for train in trains:
        key = (
            str(train.get("train_no", "")),
            int(train.get("timestamp_minutes", -1)),
            str(train.get("destination", "")),
        )
        unique[key] = train
    return sorted(unique.values(), key=lambda item: int(item.get("timestamp_minutes", 0)))


def _rail_strip_html_to_lines(html: str) -> List[str]:
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text("\n", strip=True)
            return [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]
        except Exception:
            pass
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h[1-6])\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.I)
    text = re.sub(r"&amp;", "&", text, flags=re.I)
    return [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]


def _rail_parse_mobile_timetable_html(
    html: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse Mumbai Lifeline's lightweight timetable even without BeautifulSoup."""
    lines = _rail_strip_html_to_lines(html)
    if not lines:
        return []

    requested_from = _rail_station_display_name(from_station)
    requested_from_norm = _rail_normalize(requested_from)
    trains: List[Dict[str, Any]] = []

    # Mobile pages normally contain lines such as:
    # "Virar to Churchgate 7:09 am 7:27 am". Some variants insert extra
    # whitespace or omit the selected station, so accept any station-to-dest
    # pair with two recognizable times and then validate the origin.
    time_re = r"(\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?))"
    pattern = re.compile(
        rf"^(.+?)\s+to\s+(.+?)\s+{time_re}\s+{time_re}$",
        re.I,
    )

    for raw_line in lines:
        match = pattern.match(raw_line)
        if not match:
            continue
        origin = match.group(1).strip()
        destination = match.group(2).strip()
        dep_raw = match.group(3).strip()
        arr_raw = match.group(4).strip()

        if _rail_normalize(origin) != requested_from_norm:
            continue

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
            "name": f"{requested_from} → {destination}",
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
            "source": requested_from,
            "destination": destination,
            "origin": requested_from,
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


# RailRadar replaces the former HTML timetable fetcher. The compatibility
# wrapper is defined later, after the RailRadar normalization helpers.


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




def _ist_now() -> datetime:
    """Return current India Standard Time regardless of Render's host timezone."""
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _railradar_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {RAILRADAR_API_KEY}",
        "Accept": "application/json",
        "User-Agent": "OmniTouristOS/1.0",
    }


async def _railradar_get(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 20.0,
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    """
    Server-side RailRadar GET wrapper.

    The API key never leaves Render. The Flutter app only sees our normalized
    /api/v1/railway-inquiry contract.
    """
    if not RAILRADAR_API_KEY:
        return 0, {}, "RAILRADAR_API_KEY is not configured on the backend."

    endpoint = f"{RAILRADAR_BASE_URL}/{str(path).lstrip('/')}"
    try:
        timeout = httpx.Timeout(timeout_seconds, connect=min(8.0, timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                endpoint,
                params=params or {},
                headers=_railradar_headers(),
            )

        try:
            payload = response.json()
        except Exception:
            payload = {"raw_response": response.text[:5000]}

        if not isinstance(payload, dict):
            payload = {"raw_response": payload}

        if not response.is_success:
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = str(error_obj.get("message") or error_obj.get("code") or "RailRadar request failed.")
            else:
                message = f"RailRadar returned HTTP {response.status_code}."
            return response.status_code, payload, message

        return response.status_code, payload, None

    except Exception as exc:
        return 0, {}, f"RailRadar request failed: {type(exc).__name__}: {exc}"


def _rr_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _rr_time_minutes(value: Any) -> Optional[int]:
    """Convert RailRadar HH:MM / ISO timestamps into minutes after midnight."""
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # ISO timestamps such as 2026-10-05T06:30:00+05:30.
    if "T" in raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone(timedelta(hours=5, minutes=30)))
            return parsed.hour * 60 + parsed.minute
        except Exception:
            pass

    match = re.search(r"\b(\d{1,2}):(\d{2})\b", raw)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def _rr_time_display(value: Any) -> Optional[str]:
    minutes = _rr_time_minutes(value)
    if minutes is None:
        return None
    return _rail_minutes_to_12h(minutes)


def _rr_service_type(train: Dict[str, Any]) -> Tuple[str, str]:
    name = str(train.get("name") or "").strip()
    train_type = str(train.get("type") or "").strip()
    category = str(train.get("category") or "").strip()
    text = f"{name} {train_type} {category}".upper()

    if "LADIES" in text or "WOMEN" in text:
        return "LADIES", "Ladies Local"
    if re.search(r"\bAC\b", text) or "AC LOCAL" in text:
        return "AC", "AC Local"
    if "FAST" in text:
        return "F", "Fast Local"
    if "SLOW" in text:
        return "S", "Slow Local"
    if "LOCAL" in text or "SUBURBAN" in text or "MEMU" in text:
        return "LOCAL", "Local"
    return "LOCAL", train_type or "Local"


def _rr_is_mumbai_local(train: Dict[str, Any]) -> bool:
    """Identify suburban/local services without inventing a classification."""
    name = str(train.get("name") or "").upper()
    train_type = str(train.get("type") or "").upper()
    category = str(train.get("category") or "").upper()
    text = f"{name} {train_type} {category}"
    return any(token in text for token in ("LOCAL", "SUBURBAN", "MEMU", "FAST", "SLOW", "LADIES"))


def _rr_run_days(value: Any) -> str:
    if isinstance(value, list):
        days = [str(x).strip().title() for x in value if str(x).strip()]
        return ", ".join(days) if days else "As published"
    return str(value).strip() if value else "As published"


def _rr_normalize_between_train(
    item: Dict[str, Any],
    from_station: str,
    to_station: str,
    source_url: str,
) -> Optional[Dict[str, Any]]:
    train = item.get("train")
    from_stop = item.get("from")
    to_stop = item.get("to")

    if not isinstance(train, dict):
        return None
    if not isinstance(from_stop, dict):
        from_stop = {}
    if not isinstance(to_stop, dict):
        to_stop = {}

    dep_raw = from_stop.get("departure")
    arr_raw = to_stop.get("arrival")
    dep_min = _rr_time_minutes(dep_raw)
    if dep_min is None:
        return None

    arr_min = _rr_time_minutes(arr_raw)
    if arr_min is not None and arr_min < dep_min:
        arr_min += 1440

    service_type, service_label = _rr_service_type(train)
    train_no = str(train.get("number") or "—")
    train_name = str(train.get("name") or f"{_rail_station_display_name(from_station)} → {_rail_station_display_name(to_station)}")

    live = item.get("live")
    if not isinstance(live, dict):
        live = {}

    delay = live.get("delayMinutes")
    platform = live.get("platform")
    live_type = str(live.get("type") or "").strip()

    status = "Scheduled"
    status_msg = "Scheduled timetable entry"
    is_live = bool(live)
    if live_type:
        status = live_type.replace("-", " ").title()
        status_msg = f"RailRadar live status: {status}"
    if delay is not None:
        try:
            delay_int = int(delay)
            status_msg = f"{status} • {delay_int} min delay"
        except Exception:
            pass

    return {
        "time": _rr_time_display(dep_raw),
        "departure_time": _rr_time_display(dep_raw),
        "arrival_time": _rr_time_display(arr_raw),
        "timestamp_minutes": dep_min,
        "arrival_minutes": arr_min,
        "train_no": train_no,
        "number": train_no,
        "name": train_name,
        "train_name": train_name,
        "service_type": service_type,
        "service": service_label,
        "category": str(train.get("category") or train.get("type") or service_label),
        "platform": str(platform) if platform is not None else None,
        "status": status,
        "status_msg": status_msg,
        "delay_minutes": delay,
        "crowd": None,
        "door_side": None,
        "coaches": None,
        "coach_count": None,
        "composition": None,
        "source": _rail_station_display_name(from_station),
        "destination": _rail_station_display_name(to_station),
        "origin": _rail_station_display_name(from_station),
        "ending_at": _rail_station_display_name(to_station),
        "duration": (
            f"{int(item.get('duration')) // 60}h {int(item.get('duration')) % 60}m"
            if isinstance(item.get("duration"), (int, float)) and int(item.get("duration")) >= 60
            else (
                f"{int(item.get('duration'))} min"
                if isinstance(item.get("duration"), (int, float))
                else None
            )
        ),
        "days": _rr_run_days(train.get("runDays")),
        "live": is_live,
        "live_feed_available": is_live,
        "data_source": source_url,
        "data_source_type": "RailRadar API",
    }


def _rr_normalize_station_train(
    item: Dict[str, Any],
    station_code: str,
    source_url: str,
) -> Optional[Dict[str, Any]]:
    train = item.get("train")
    stop = item.get("stop")

    if not isinstance(train, dict):
        return None
    if not isinstance(stop, dict):
        stop = {}

    dep_raw = stop.get("departure")
    arr_raw = stop.get("arrival")
    chosen_raw = dep_raw or arr_raw
    minute = _rr_time_minutes(chosen_raw)
    if minute is None:
        return None

    service_type, service_label = _rr_service_type(train)
    train_no = str(train.get("number") or "—")
    train_name = str(train.get("name") or "Mumbai Local")
    destination = train.get("destination")
    if isinstance(destination, dict):
        destination = destination.get("name") or destination.get("code")
    source = train.get("source")
    if isinstance(source, dict):
        source = source.get("name") or source.get("code")

    return {
        "time": _rr_time_display(chosen_raw),
        "departure_time": _rr_time_display(dep_raw),
        "arrival_time": _rr_time_display(arr_raw),
        "timestamp_minutes": minute,
        "arrival_minutes": _rr_time_minutes(arr_raw),
        "train_no": train_no,
        "number": train_no,
        "name": train_name,
        "train_name": train_name,
        "service_type": service_type,
        "service": service_label,
        "category": str(train.get("category") or train.get("type") or service_label),
        "platform": None,
        "status": "Scheduled",
        "status_msg": "Scheduled timetable entry",
        "delay_minutes": None,
        "crowd": None,
        "door_side": None,
        "coaches": None,
        "coach_count": None,
        "composition": None,
        "source": str(source or _rail_station_display_name(station_code)),
        "destination": str(destination or "—"),
        "origin": str(source or _rail_station_display_name(station_code)),
        "ending_at": str(destination or "—"),
        "duration": None,
        "days": _rr_run_days(train.get("runDays")),
        "live": False,
        "live_feed_available": False,
        "stop_sequence": stop.get("sequence"),
        "stop_type": stop.get("stopType"),
        "data_source": source_url,
        "data_source_type": "RailRadar API",
    }


def _rr_live_normalize(payload: Dict[str, Any], train_query: str) -> Dict[str, Any]:
    """Normalize RailRadar live telemetry into a UI-friendly, station-aware payload.

    RailRadar's currentLocation often contains only a stationCode. The route array
    contains the corresponding stationName and scheduled/actual times, so enrich
    the live response from that route instead of showing only ``RailRadar: Running``.
    """
    data = _rr_data(payload)
    train = data.get("train") if isinstance(data.get("train"), dict) else {}
    current = data.get("currentLocation") if isinstance(data.get("currentLocation"), dict) else {}
    previous = data.get("previousHalt") if isinstance(data.get("previousHalt"), dict) else {}
    next_halt = data.get("nextHalt") if isinstance(data.get("nextHalt"), dict) else {}
    route = data.get("route") if isinstance(data.get("route"), list) else []
    route = [x for x in route if isinstance(x, dict)]

    def code_of(obj: Dict[str, Any]) -> Optional[str]:
        value = obj.get("stationCode") or obj.get("code")
        return str(value).strip().upper() if value not in (None, "") else None

    def name_of(obj: Dict[str, Any]) -> Optional[str]:
        value = obj.get("stationName") or obj.get("name")
        return str(value).strip() if value not in (None, "") else None

    def route_match(code: Optional[str]) -> Optional[Dict[str, Any]]:
        if not code:
            return None
        for stop in route:
            if str(stop.get("stationCode") or stop.get("code") or "").strip().upper() == code:
                return stop
        return None

    current_code = code_of(current) or code_of(previous)
    current_route = route_match(current_code)
    current_seq = current.get("sequence") or (current_route or {}).get("sequence")

    # Route-derived station names are more reliable than exposing a bare station code.
    current_label = name_of(current) or name_of(current_route) or name_of(previous)
    if not current_label and current_code:
        current_label = current_code

    # Work out the previous and next actual route stops from sequence whenever
    # possible. This avoids RailRadar payload variations around previousHalt/nextHalt.
    previous_route = None
    next_route = None
    if isinstance(current_seq, (int, float)):
        earlier = [x for x in route if isinstance(x.get("sequence"), (int, float)) and x.get("sequence") < current_seq]
        later = [x for x in route if isinstance(x.get("sequence"), (int, float)) and x.get("sequence") > current_seq]
        if earlier:
            previous_route = max(earlier, key=lambda x: x.get("sequence", -1))
        if later:
            next_route = min(later, key=lambda x: x.get("sequence", 10**9))

    previous_obj = previous_route or previous
    next_obj = next_route or next_halt
    previous_label = name_of(previous_obj) or code_of(previous_obj) or current_label
    next_label = name_of(next_obj) or code_of(next_obj)

    # If RailRadar's nextHalt points at the current station, prefer the next route stop.
    if next_label and current_code and code_of(next_obj) == current_code and next_route:
        next_obj = next_route
        next_label = name_of(next_obj) or code_of(next_obj)

    delay = data.get("delayMinutes")
    if delay is None:
        delay = data.get("overallDelayMinutes")
    if delay is None and isinstance(current_route, dict):
        delay = current_route.get("delayDeparture") or current_route.get("delayArrival")

    platform = current.get("platform") or next_halt.get("platform")
    if platform is None and isinstance(current_route, dict):
        platform = current_route.get("platform")
    if platform is None and isinstance(next_route, dict):
        platform = next_route.get("platform")

    # Parse a provider ISO timestamp without assuming UTC. RailRadar timestamps
    # include +05:30 for India, so datetime.fromisoformat preserves the offset.
    def parse_dt(value: Any) -> Optional[datetime]:
        if value in (None, ""):
            return None
        try:
            text = str(value).strip().replace("Z", "+00:00")
            return datetime.fromisoformat(text)
        except Exception:
            return None

    last_departure_at = None
    if isinstance(current_route, dict):
        last_departure_at = current_route.get("actualDeparture") or current_route.get("scheduledDeparture")
    if not last_departure_at and isinstance(previous_route, dict):
        last_departure_at = previous_route.get("actualDeparture") or previous_route.get("scheduledDeparture")
    if not last_departure_at:
        last_departure_at = data.get("lastDepartureAt")

    next_eta_minutes = None
    if isinstance(next_obj, dict):
        eta_value = (
            next_obj.get("expectedArrival")
            or next_obj.get("expectedArrivalTime")
            or next_obj.get("scheduledArrival")
        )
        eta_dt = parse_dt(eta_value)
        if eta_dt is not None:
            try:
                if delay is not None:
                    eta_dt = eta_dt + timedelta(minutes=float(delay))
                now = datetime.now(eta_dt.tzinfo) if eta_dt.tzinfo else datetime.now()
                next_eta_minutes = max(0, int(round((eta_dt - now).total_seconds() / 60.0)))
            except Exception:
                next_eta_minutes = None

    progress = current.get("segmentProgress")
    if progress is None:
        progress = current.get("segment_progress")
    try:
        if progress is not None:
            progress = max(0.0, min(1.0, float(progress)))
    except Exception:
        progress = None

    status = str(data.get("status") or current.get("status") or "unknown").replace("_", " ").title()
    is_live = bool(data.get("isLive", True))

    # A live response with telemetry but no explicit isLive should still be usable.
    if current or data.get("lastUpdatedAt"):
        is_live = bool(data.get("isLive", True))

    return {
        "status": "success",
        "type": "live_train",
        "train_no": str(data.get("trainNumber") or train.get("number") or train_query),
        "train_name": str(data.get("trainName") or train.get("name") or "Mumbai Local"),
        "current_station": current_label,
        "next_station": next_label,
        "last_departure_station": previous_label,
        "platform": str(platform) if platform is not None else None,
        "door_side": None,
        "delay_minutes": delay,
        "crowd": None,
        "status_msg": f"RailRadar: {status}" if is_live else "RailRadar has no live movement fix for this run.",
        "live": is_live,
        "live_feed_available": is_live,
        "current_location": current_label,
        "next_stop": next_label,
        "data_source_label": "RailRadar Live Train API",
        "last_updated_at": data.get("lastUpdatedAt"),
        "last_departure_at": last_departure_at,
        "next_eta_minutes": next_eta_minutes,
        "segment_progress": progress,
        "speed_kmh": current.get("speedKmh") or current.get("speed_kmh"),
        "bearing_degrees": current.get("bearingDegrees") or current.get("bearing_degrees"),
        "current_station_code": current_code,
        "next_station_code": code_of(next_obj),
    }


def _rr_pnr_normalize(payload: Dict[str, Any], pnr: str) -> Dict[str, Any]:
    data = _rr_data(payload)

    def first(*keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
        return None

    train = data.get("train")
    if not isinstance(train, dict):
        train = {}

    from_station = first("fromStation", "from", "source")
    to_station = first("toStation", "to", "destination")

    def station_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("name") or value.get("code")
        return value

    passengers = data.get("passengers")
    if not isinstance(passengers, list):
        passengers = data.get("passengerDetails")
    if not isinstance(passengers, list):
        passengers = []

    return {
        "status": "success",
        "type": "pnr",
        "pnr": str(first("pnrNumber", "pnr") or pnr),
        "train_no": str(first("trainNumber", "trainNo") or train.get("number") or "—"),
        "train_name": str(first("trainName") or train.get("name") or "—"),
        "from_station": station_value(from_station) or "—",
        "to_station": station_value(to_station) or "—",
        "journey_date": str(first("journeyDate", "date") or "—"),
        "booking_status": str(first("bookingStatus", "status", "chartStatus") or "Status unavailable"),
        "status": str(first("bookingStatus", "status", "chartStatus") or "Status unavailable"),
        "status_msg": str(first("statusMessage", "message") or "PNR status returned by RailRadar."),
        "passengers": passengers,
        "live": True,
        "data_source_label": "RailRadar API",
    }


def _rr_route_matches_service(train: Dict[str, Any], wanted: str) -> bool:
    wanted = str(wanted or "ALL").upper()
    if wanted in {"", "ALL"}:
        return True
    service_type = str(train.get("service_type") or "").upper()
    service = str(train.get("service") or "").upper()
    category = str(train.get("category") or "").upper()
    text = f"{service_type} {service} {category}"
    if wanted == "FAST":
        return "FAST" in text or service_type == "F"
    if wanted == "SLOW":
        return "SLOW" in text or service_type == "S"
    if wanted == "AC":
        return "AC" in text or service_type == "AC"
    if wanted == "LADIES":
        return "LADIES" in text or "WOMEN" in text or service_type == "LADIES"
    return True


async def _railradar_fetch_between(
    from_station: str,
    to_station: str,
    *,
    date_value: Optional[str] = None,
    live: bool = False,
) -> Dict[str, Any]:
    from_code = str(from_station).strip().upper()
    to_code = str(to_station).strip().upper()
    date_value = date_value or _ist_now().strftime("%Y-%m-%d")

    source_url = f"{RAILRADAR_BASE_URL}/trains/between/{urllib.parse.quote(from_code)}/{urllib.parse.quote(to_code)}"
    cache_key = f"railradar-between|{from_code}|{to_code}|{date_value}|{live}"

    if not live:
        cached = _rail_cache_get(cache_key)
        if cached is not None:
            return cached

    params: Dict[str, Any] = {
        "type": "local",
        "category": "Suburban",
        "date": date_value,
        "live": str(bool(live)).lower(),
    }

    status_code, payload, error = await _railradar_get(
        f"trains/between/{urllib.parse.quote(from_code)}/{urllib.parse.quote(to_code)}",
        params=params,
    )

    # If a provider build rejects the optional category filter, retry with the
    # documented type=local filter alone.
    if error and status_code == 400:
        params.pop("category", None)
        status_code, payload, error = await _railradar_get(
            f"trains/between/{urllib.parse.quote(from_code)}/{urllib.parse.quote(to_code)}",
            params=params,
        )

    data = _rr_data(payload)
    provider_trains = data.get("trains") if isinstance(data.get("trains"), list) else []

    normalized: List[Dict[str, Any]] = []
    for item in provider_trains:
        if not isinstance(item, dict):
            continue
        train = _rr_normalize_between_train(
            item,
            from_station=from_code,
            to_station=to_code,
            source_url=source_url,
        )
        if train:
            normalized.append(train)

    normalized.sort(key=lambda item: int(item.get("timestamp_minutes", 0)))

    result = {
        "trains": normalized,
        "source_url": source_url,
        "source_label": "RailRadar API",
        "line": _rail_lines_for_station(from_code)[0] if _rail_lines_for_station(from_code) else "ALL",
        "line_name": _rail_line_name(_rail_lines_for_station(from_code)[0]) if _rail_lines_for_station(from_code) else "Mumbai Suburban Network",
        "from": _rail_station_display_name(from_code),
        "to": _rail_station_display_name(to_code),
        "date": date_value,
        "live": live,
        "error": error,
        "http_status": status_code or None,
        "provider_count": len(provider_trains),
    }

    if not live:
        _rail_cache_set(cache_key, result)

    return result



async def _railradar_fetch_station_live_board(
    station_code: str,
    *,
    hours: int = 2,
) -> Dict[str, Any]:
    """Fetch RailRadar's live station board.

    This is intentionally separate from the static station timetable endpoint.
    For a commuter board we need trains that merely HALT at the selected station
    (for example Virar -> Churchgate trains halting at Naigaon), plus the live
    departure/platform/delay fields supplied by RailRadar.
    """
    station = str(station_code).strip().upper()
    hours = hours if hours in {2, 4, 6, 8} else 2
    source_url = f"{RAILRADAR_BASE_URL}/stations/{urllib.parse.quote(station)}/live"

    status_code, payload, error = await _railradar_get(
        f"stations/{urllib.parse.quote(station)}/live",
        params={
            "hours": str(hours),
            "includeIntermediate": "false",
        },
    )

    data = _rr_data(payload)
    provider_trains = data.get("trains") if isinstance(data.get("trains"), list) else []
    normalized: List[Dict[str, Any]] = []

    for item in provider_trains:
        if not isinstance(item, dict):
            continue
        train_obj = item.get("train")
        stop = item.get("stop") if isinstance(item.get("stop"), dict) else {}
        live = item.get("live") if isinstance(item.get("live"), dict) else {}
        if not isinstance(train_obj, dict):
            continue
        if not _rr_is_mumbai_local(train_obj):
            continue

        expected_raw = live.get("expectedDepartureTime") or stop.get("departure") or stop.get("arrival")
        minute = _rr_time_minutes(expected_raw)
        if minute is None:
            continue

        service_type, service_label = _rr_service_type(train_obj)
        train_no = str(train_obj.get("number") or "—")
        train_name = str(train_obj.get("name") or "Mumbai Local")

        destination = train_obj.get("destination")
        if isinstance(destination, dict):
            destination = destination.get("name") or destination.get("code")
        source = train_obj.get("source")
        if isinstance(source, dict):
            source = source.get("name") or source.get("code")

        live_type = str(live.get("type") or "scheduled").strip()
        delay = live.get("delayMinutes")
        platform = live.get("platform")
        status = live_type.replace("-", " ").title() if live_type else "Scheduled"
        status_msg = f"Live: {status}"
        if delay is not None:
            try:
                status_msg = f"{status} • {int(delay)} min delay"
            except Exception:
                pass

        normalized.append({
            "time": _rr_time_display(expected_raw),
            "departure_time": _rr_time_display(expected_raw),
            "scheduled_departure_time": _rr_time_display(stop.get("departure")),
            "arrival_time": _rr_time_display(stop.get("arrival")),
            "timestamp_minutes": minute,
            "arrival_minutes": _rr_time_minutes(stop.get("arrival")),
            "train_no": train_no,
            "number": train_no,
            "name": train_name,
            "train_name": train_name,
            "service_type": service_type,
            "service": service_label,
            "category": str(train_obj.get("category") or train_obj.get("type") or service_label),
            "platform": str(platform) if platform is not None else None,
            "status": status,
            "status_msg": status_msg,
            "delay_minutes": delay,
            "crowd": None,
            "door_side": None,
            "coaches": None,
            "coach_count": None,
            "composition": None,
            "source": str(source or _rail_station_display_name(station)),
            "destination": str(destination or "—"),
            "origin": str(source or _rail_station_display_name(station)),
            "ending_at": str(destination or "—"),
            "duration": None,
            "days": _rr_run_days(train_obj.get("runDays")),
            "live": True,
            "live_feed_available": True,
            "live_type": live_type,
            "expected_departure_at": live.get("expectedDepartureTime"),
            "stop_sequence": stop.get("sequence"),
            "data_source": source_url,
            "data_source_type": "RailRadar Live Station API",
        })

    normalized.sort(key=lambda item: (int(item.get("timestamp_minutes", 0)), str(item.get("train_no", ""))))
    return {
        "trains": normalized,
        "source_url": source_url,
        "source_label": "RailRadar Live Station API",
        "station_code": station,
        "station_name": (
            str(data.get("station", {}).get("name"))
            if isinstance(data.get("station"), dict) and data.get("station", {}).get("name")
            else _rail_station_display_name(station)
        ),
        "error": error,
        "http_status": status_code or None,
        "provider_count": len(provider_trains),
    }


async def _railradar_fetch_station_board(
    station_code: str,
    *,
    include_intermediate: bool = False,
) -> Dict[str, Any]:
    station = str(station_code).strip().upper()
    source_url = f"{RAILRADAR_BASE_URL}/stations/{urllib.parse.quote(station)}/trains"
    cache_key = f"railradar-station|{station}|{include_intermediate}"

    cached = _rail_cache_get(cache_key)
    if cached is not None:
        return cached

    status_code, payload, error = await _railradar_get(
        f"stations/{urllib.parse.quote(station)}/trains",
        params={"includeIntermediate": str(bool(include_intermediate)).lower()},
    )

    data = _rr_data(payload)
    provider_trains = data.get("trains") if isinstance(data.get("trains"), list) else []

    normalized: List[Dict[str, Any]] = []
    for item in provider_trains:
        if not isinstance(item, dict):
            continue
        train_obj = item.get("train")
        if not isinstance(train_obj, dict):
            continue

        # This screen is specifically Mumbai suburban. RailRadar's station
        # board can also contain non-suburban trains at interchange stations.
        if not _rr_is_mumbai_local(train_obj):
            continue

        train = _rr_normalize_station_train(item, station, source_url)
        if train:
            normalized.append(train)

    # Some provider responses may not expose an explicit "Local" type even for
    # suburban records. Only then fall back to the returned rows rather than
    # fabricating a classification.
    if not normalized and provider_trains:
        for item in provider_trains:
            if not isinstance(item, dict):
                continue
            train = _rr_normalize_station_train(item, station, source_url)
            if train:
                normalized.append(train)

    normalized.sort(
        key=lambda item: (
            int(item.get("timestamp_minutes", 0)),
            str(item.get("train_no", "")),
        )
    )

    result = {
        "trains": normalized,
        "source_url": source_url,
        "source_label": "RailRadar API",
        "station_code": station,
        "station_name": (
            str(data.get("station", {}).get("name"))
            if isinstance(data.get("station"), dict) and data.get("station", {}).get("name")
            else _rail_station_display_name(station)
        ),
        "error": error,
        "http_status": status_code or None,
        "provider_count": len(provider_trains),
    }
    _rail_cache_set(cache_key, result)
    return result



async def _rail_fetch_timetable(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for the existing interchange engine.

    The old implementation scraped a public timetable website. It is now
    backed by RailRadar while preserving the same normalized return shape.
    """
    result = await _railradar_fetch_between(
        from_station=from_station,
        to_station=to_station,
        date_value=_ist_now().strftime("%Y-%m-%d"),
        live=False,
    )

    trains = []
    for train in result.get("trains", []):
        minute = train.get("timestamp_minutes")
        if minute is None:
            continue
        minute = int(minute) % 1440
        start = int(after_minutes) % 1440
        end = int(before_minutes) % 1440
        if start <= end:
            in_window = start <= minute <= end
        else:
            in_window = minute >= start or minute <= end
        if in_window:
            item = dict(train)
            item["line"] = line
            item["line_name"] = _rail_line_name(line)
            trains.append(item)

    return {
        **result,
        "trains": trains,
        "line": line,
        "line_name": _rail_line_name(line),
    }


@app.get("/api/v1/railradar-test")
async def railradar_test(
    date: Optional[str] = Query(None),
    live: bool = Query(False),
):
    """Server-side RailRadar connectivity diagnostic for NIG -> CCG."""
    if not RAILRADAR_API_KEY:
        return {
            "status": "error",
            "railradar_configured": False,
            "message": "RAILRADAR_API_KEY is not configured on the backend.",
        }

    test_date = (date or _ist_now().strftime("%Y-%m-%d")).strip()
    result = await _railradar_fetch_between(
        "NIG",
        "CCG",
        date_value=test_date,
        live=live,
    )

    return {
        "status": "success" if not result.get("error") else "error",
        "railradar_configured": True,
        "railradar_reachable": result.get("http_status") is not None,
        "http_status": result.get("http_status"),
        "endpoint_tested": result.get("source_url"),
        "route": "NIG -> CCG",
        "date": test_date,
        "live": live,
        "train_count": len(result.get("trains", [])),
        "provider_count": result.get("provider_count", 0),
        "error": result.get("error"),
        "provider_normalized_trains": result.get("trains", [])[:20],
    }


@app.post("/api/v1/railway-inquiry")
async def railway_inquiry(request: Request):
    """
    Backward-compatible Omni Rail gateway.

    Flutter continues to call /api/v1/railway-inquiry. RailRadar stays behind
    this server endpoint; the API key is never embedded in the app.
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

        if not RAILRADAR_API_KEY:
            return {
                "status": "error",
                "type": normalized_type,
                "message": "Railway service is not configured on the backend. Add RAILRADAR_API_KEY in Render.",
                "trains": [],
            }

        # ----------------------------- ROUTE SEARCH -----------------------------
        if normalized_type == "route_search":
            try:
                route_payload = (
                    query_value
                    if isinstance(query_value, dict)
                    else json.loads(str(query_value)) if query_value else {}
                )
            except Exception:
                route_payload = {}

            from_station = str(
                route_payload.get("from")
                or route_payload.get("source")
                or route_payload.get("from_station")
                or "NIG"
            ).strip().upper()

            to_station = str(
                route_payload.get("to")
                or route_payload.get("destination")
                or route_payload.get("to_station")
                or "CCG"
            ).strip().upper()

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

            # Direct RailRadar route search. For Mumbai suburban pairs this is
            # the authoritative provider response and avoids the old HTML source.
            if direct_line and not line_error:
                # A commuter route must include trains that ORIGINATE elsewhere
                # but HALT at the selected origin station (e.g. Virar -> Churchgate
                # when the user selects Naigaon). The live station-board endpoint
                # is the correct source for that use case.
                station_live = await _railradar_fetch_station_live_board(
                    from_station,
                    hours=2,
                )
                station_trains = station_live.get("trains", [])

                destination_norm = _rail_normalize(to_station)
                trains = []
                for train in station_trains:
                    destination = _rail_normalize(str(train.get("destination") or ""))
                    ending_at = _rail_normalize(str(train.get("ending_at") or ""))
                    if destination_norm and destination_norm not in destination and destination_norm not in ending_at:
                        # For Churchgate/CCG, allow the exact code/name variants.
                        if to_station == "CCG" and "CHURCHGATE" not in f"{destination} {ending_at}" and "CCG" not in f"{destination} {ending_at}":
                            continue
                        elif to_station != "CCG":
                            continue
                    if _rr_route_matches_service(train, requested_service):
                        trains.append(train)

                result = {
                    **station_live,
                    "trains": trains,
                    "source_url": station_live.get("source_url"),
                    "source_label": "RailRadar Live Station API",
                    "from": _rail_station_display_name(from_station),
                    "to": _rail_station_display_name(to_station),
                    "date": _ist_now().strftime("%Y-%m-%d"),
                    "live": True,
                }

                if result.get("error") and not result.get("trains"):
                    return {
                        "status": "error",
                        "type": "route_search",
                        "station_code": from_station,
                        "station_name": _rail_station_display_name(from_station),
                        "from": _rail_station_display_name(from_station),
                        "to": _rail_station_display_name(to_station),
                        "line": direct_line,
                        "line_name": _rail_line_name(direct_line),
                        "trains": [],
                        "route_search": True,
                        "requires_interchange": False,
                        "service_filter": requested_service,
                        "message": str(result.get("error")),
                        "source_error": str(result.get("error")),
                        "data_source_label": "RailRadar API",
                    }

                trains = [
                    train
                    for train in result.get("trains", [])
                    if _rr_route_matches_service(train, requested_service)
                ]

                # The Flutter planner sends the requested clock time. Respect it
                # instead of silently replacing it with the server's current time.
                requested_minutes = route_payload.get("time_minutes")
                try:
                    requested_minutes = int(requested_minutes) if requested_minutes is not None else None
                except Exception:
                    requested_minutes = None

                if requested_minutes is None:
                    requested_minutes = _ist_now().hour * 60 + _ist_now().minute

                window_start = max(0, requested_minutes - 60)
                window_end = min(1439, requested_minutes + 120)

                around_time = [
                    train
                    for train in trains
                    if window_start <= int(train.get("timestamp_minutes", 0)) <= window_end
                ]

                # If the requested time is near a boundary or provider has only
                # a sparse result set, retain the full provider result rather than
                # displaying an empty board.
                display_trains = around_time or trains

                return {
                    "status": "success",
                    "type": "route_search",
                    "station_code": from_station,
                    "station_name": _rail_station_display_name(from_station),
                    "from": _rail_station_display_name(from_station),
                    "to": _rail_station_display_name(to_station),
                    "line": direct_line,
                    "line_name": _rail_line_name(direct_line),
                    "trains": display_trains[:100],
                    "route_search": True,
                    "requires_interchange": False,
                    "service_filter": requested_service,
                    "requested_time": _rail_minutes_to_12h(requested_minutes),
                    "window_start": _rail_minutes_to_12h(window_start),
                    "window_end": _rail_minutes_to_12h(window_end),
                    "current_time": _rail_minutes_to_12h(
                        _ist_now().hour * 60 + _ist_now().minute
                    ),
                    "data_source": result.get("source_url"),
                    "data_source_label": "RailRadar API",
                    "source_notice": (
                        "Scheduled Mumbai suburban timetable data is supplied by RailRadar. "
                        "Live delay/platform values are shown only when a live RailRadar field is present."
                    ),
                    "source_error": result.get("error"),
                }

            # Cross-line journeys retain the existing Omni Rail interchange
            # algorithm, but its underlying timetable fetch now uses RailRadar.
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
                    _ist_now().hour * 60 + _ist_now().minute
                )
                cross_line["data_source_label"] = "RailRadar API"
                cross_line["source_notice"] = (
                    "Connected journeys are assembled from RailRadar scheduled timetable entries."
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
            station = str(query_value or "NIG").strip().upper()
            station_lines = _rail_lines_for_station(station)

            if not station_lines:
                return {
                    "status": "error",
                    "type": "station_board",
                    "station_code": station,
                    "station_name": _rail_station_display_name(station),
                    "message": "Station not found in the Omni Rail station directory.",
                    "trains": [],
                }

            # Use RailRadar's live station board for the visible commuter board.
            # The static timetable endpoint is retained for compatibility elsewhere.
            result = await _railradar_fetch_station_live_board(station, hours=2)
            trains = list(result.get("trains", []))

            current = _ist_now()
            current_min = current.hour * 60 + current.minute

            # De-duplicate provider rows.
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for train in trains:
                key = (
                    str(train.get("train_no")),
                    int(train.get("timestamp_minutes", -1)),
                    str(train.get("destination")),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(train)

            deduped.sort(
                key=lambda item: (
                    int(item.get("timestamp_minutes", 0)),
                    str(item.get("train_no", "")),
                )
            )

            upcoming = [
                train
                for train in deduped
                if int(train.get("timestamp_minutes", 0)) >= current_min
            ][:12]

            unique_line_set: List[str] = []
            for line in station_lines:
                if line not in unique_line_set:
                    unique_line_set.append(line)

            return {
                "status": "success",
                "type": "station_board",
                "station_code": station,
                "station_name": result.get("station_name") or _rail_station_display_name(station),
                "line": unique_line_set[0] if len(unique_line_set) == 1 else "ALL",
                "line_name": (
                    _rail_line_name(unique_line_set[0])
                    if len(unique_line_set) == 1
                    else "Mumbai Suburban Network"
                ),
                "lines": unique_line_set,
                "line_names": [_rail_line_name(line) for line in unique_line_set],
                "current_time": _rail_minutes_to_12h(current_min),
                "board_window": "Full scheduled day",
                "trains": deduped[:450],
                "next_trains": upcoming,
                "train_count": len(deduped),
                "source_urls": [result.get("source_url")] if result.get("source_url") else [],
                "source_errors": [str(result.get("error"))] if result.get("error") else [],
                "source_label": "RailRadar API",
                "source_notice": (
                    "Mumbai suburban scheduled timetable data is supplied by RailRadar. "
                    "Platform and delay values are left empty unless a live provider field supplies them."
                ),
                "station_directory": [
                    {
                        "code": code,
                        "name": data["name"],
                        "line": data["line"],
                    }
                    for code, data in RAILWAY_STATIONS.items()
                    if _rail_normalize(data["name"])
                    == _rail_normalize(result.get("station_name") or _rail_station_display_name(station))
                ],
            }

        # ----------------------------- LIVE TRAIN -----------------------------
        if normalized_type == "live_train":
            train_query = re.sub(r"\D", "", str(query_value or ""))
            if len(train_query) < 4 or len(train_query) > 6:
                return {
                    "status": "error",
                    "type": "live_train",
                    "message": "Enter the train number shown in the Omni Rail timetable.",
                }

            status_code, payload, error = await _railradar_get(
                f"trains/{urllib.parse.quote(train_query)}/live",
                params={
                    "authoritative": "true",
                    "haltsOnly": "true",
                },
            )

            if error:
                return {
                    "status": "error",
                    "type": "live_train",
                    "train_no": train_query,
                    "message": str(error),
                    "status_msg": str(error),
                    "live": False,
                    "live_feed_available": False,
                    "data_source_label": "RailRadar API",
                    "http_status": status_code or None,
                }

            return _rr_live_normalize(payload, train_query)

        # ----------------------------- PNR -----------------------------
        if normalized_type == "pnr":
            pnr = re.sub(r"\D", "", str(query_value or ""))
            if len(pnr) != 10:
                return {
                    "status": "error",
                    "type": "pnr",
                    "message": "PNR must contain exactly 10 digits.",
                }

            status_code, payload, error = await _railradar_get(
                f"pnr/{urllib.parse.quote(pnr)}"
            )

            if error:
                return {
                    "status": "error",
                    "type": "pnr",
                    "pnr": pnr,
                    "message": str(error),
                    "status_msg": str(error),
                    "live": False,
                    "data_source_label": "RailRadar API",
                    "http_status": status_code or None,
                }

            return _rr_pnr_normalize(payload, pnr)

        return {
            "status": "error",
            "type": normalized_type,
            "message": "Unsupported railway query type.",
            "trains": [],
        }

    except Exception as exc:
        print(f"[Omni Rail] railway_inquiry error: {type(exc).__name__}: {exc}")
        return {
            "status": "error",
            "message": f"Railway inquiry error: {exc}",
            "trains": [],
        }

# -------------------------------------------------------------

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