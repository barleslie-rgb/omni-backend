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
import urllib.parse
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, List, Dict, Any, Tuple
import xml.etree.ElementTree as ET

import httpx
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
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


async def _google_text_search(query: str, max_result_count: int = 10) -> List[Dict[str, Any]]:
    if not GOOGLE_PLACES_API_KEY:
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
    ])
    payload = {
        "textQuery": query,
        "pageSize": max(1, min(int(max_result_count), 20)),
        "languageCode": "en",
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
            print(f"[Google Places Notice] {response.status_code}: {response.text[:300]}")
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
        "images": _google_photo_proxy_urls(photo_names[:5]),
        "google_photo_names": photo_names[:5],
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
        "images": _google_photo_proxy_urls(photo_names[:5]),
        "google_photo_names": photo_names[:5],
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

    # Text Search normally returns photo resource names when places.photos is
    # requested. Some place records still arrive without photos; hydrate those
    # records through Place Details before giving up so the Explorer does not
    # unnecessarily fall back to a blank image. Google documents that photos
    # can be obtained from Text Search or Place Details.
    place_values = list(merged.values())
    missing_photo_places = [
        place for place in place_values[:20]
        if not (place.get("photos") or []) and place.get("id")
    ]
    if missing_photo_places:
        hydrated = await asyncio.gather(
            *[_google_place_details(str(place.get("id"))) for place in missing_photo_places],
            return_exceptions=True,
        )
        for original, detail in zip(missing_photo_places, hydrated):
            if isinstance(detail, dict) and detail.get("photos"):
                original["photos"] = detail.get("photos") or []

    normalized: List[Dict[str, Any]] = []
    for place in place_values:
        item = _normalize_google_place(place, destination["latitude"], destination["longitude"])
        if item:
            normalized.append(item)

    normalized.sort(key=lambda x: (-(float(x.get("rating") or 0)), -(int(x.get("reviews") or 0))))
    return destination, normalized[:20]


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
        booking_link = str(offer.get("link") or "").strip()

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

        place = await _google_place_details(place_id) if place_id else None
        if not place:
            # The card itself is still valid; details can be synthesized from the
            # provider-backed card fields without pretending a second provider lookup succeeded.
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
                "images": _google_photo_proxy_urls(photo_names[:5]),
                "google_photo_names": photo_names[:5],
                "source": "Google Places + Omni travel intelligence",
                "data_state": "VERIFIED",
                "details_state": "AI_GUIDED_WITH_VERIFIED_PROVIDER_FACTS",
            },
            "hotels": nearby_stays.get("hotels", []),
            "hotel_state": nearby_stays.get("status"),
            "hotel_provider": nearby_stays.get("provider"),
            "hotel_reason": nearby_stays.get("reason"),
            "attribution_required": ["Google Maps", "Booking.com"],
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
        if not destination:
            return {
                "status": "unavailable",
                "message": "Destination resolution is unavailable. Configure GOOGLE_PLACES_API_KEY or provide a provider-backed destination resolver.",
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
        if not stays.get("hotels"):
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
            "places_provider": "Google Places",
            "places_state": "VERIFIED" if places else "EMPTY",
            "landmarks": places,
            "places": places,
            "hotels": stays.get("hotels", []),
            "hotel_provider": stays.get("provider"),
            "hotel_state": stays.get("status"),
            "hotel_reason": stays.get("reason"),
            "hotel_request_id": stays.get("request_id"),
            "attribution_required": ["Google Maps", "Booking.com"],
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


def _rail_parse_timetable_html(
    html: str,
    line: str,
    from_station: str,
    to_station: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse Mumbai Lifeline's timetable table without assuming a fixed layout."""
    if BeautifulSoup is None:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    trains: List[Dict[str, Any]] = []
    requested_from_name = _rail_station_display_name(from_station)
    requested_to_name = _rail_station_display_name(to_station)

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        header_row_index: Optional[int] = None
        header_cells: List[str] = []

        # The page contains navigation/summary rows before the real timetable.
        # Search the first 60 rows for a row containing Train No and station data.
        for row_index, row in enumerate(rows[:60]):
            cells = row.find_all(["th", "td"])
            candidate = [cell.get_text(" ", strip=True) for cell in cells]
            normalized = [_rail_normalize(x) for x in candidate]
            has_train = any(x in {"trainno", "trainnumber", "train"} for x in normalized)
            has_station = any(
                _rail_normalize(requested_from_name) in x
                or _rail_normalize(from_station) in x
                for x in normalized
            )
            if has_train and has_station:
                header_row_index = row_index
                header_cells = candidate
                break

        if header_row_index is None:
            continue

        train_idx = _rail_find_header(header_cells, ["Train No", "Train Number", "Train"])
        speed_idx = _rail_find_header(header_cells, ["Speed"])
        coach_idx = _rail_find_header(header_cells, ["Coaches"])
        origin_idx = _rail_find_header(header_cells, ["Originating From", "Origin"])
        ending_idx = _rail_find_header(header_cells, ["Ending At", "Destination"])
        duration_idx = _rail_find_header(header_cells, ["Duration"])
        from_time_idx = _rail_find_header(
            header_cells,
            [requested_from_name, from_station, _rail_legacy_station_param(from_station)],
        )
        to_time_idx = _rail_find_header(
            header_cells,
            [requested_to_name, to_station, _rail_legacy_station_param(to_station)],
        )

        if train_idx is None or from_time_idx is None:
            continue

        for row in rows[header_row_index + 1:]:
            cells = row.find_all(["th", "td"])
            values = [c.get_text(" ", strip=True) for c in cells]
            if not values or train_idx >= len(values) or from_time_idx >= len(values):
                continue

            train_cell = values[train_idx].strip()
            dep_raw = values[from_time_idx].strip()
            arr_raw = (
                values[to_time_idx].strip()
                if to_time_idx is not None and to_time_idx < len(values)
                else ""
            )

            if not train_cell or not dep_raw or dep_raw in {"—", "-", "–"}:
                continue

            dep_min = _rail_time_to_minutes(dep_raw)
            arr_min = _rail_time_to_minutes(arr_raw) if arr_raw else None
            if dep_min is None:
                continue

            speed = values[speed_idx] if speed_idx is not None and speed_idx < len(values) else ""
            coaches = values[coach_idx] if coach_idx is not None and coach_idx < len(values) else None
            origin = values[origin_idx] if origin_idx is not None and origin_idx < len(values) else None
            ending = values[ending_idx] if ending_idx is not None and ending_idx < len(values) else None
            duration = values[duration_idx] if duration_idx is not None and duration_idx < len(values) else None

            service_type, service_label = _rail_detect_service(train_cell, speed)
            cleaned_train_no = _rail_clean_train_no(train_cell)
            if cleaned_train_no == "—":
                cleaned_train_no = train_cell

            if arr_min is not None and arr_min < dep_min:
                arr_min += 1440

            trains.append({
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
            })

        if trains:
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


async def _rail_fetch_timetable(
    line: str,
    from_station: str,
    to_station: str,
    after_minutes: int,
    before_minutes: int,
) -> Dict[str, Any]:
    line = str(line).upper()
    from_name = _rail_station_display_name(from_station)
    to_name = _rail_station_display_name(to_station)

    source_urls = _rail_candidate_urls(
        line=line,
        from_station=from_station,
        to_station=to_station,
        after_minutes=after_minutes,
        before_minutes=before_minutes,
    )
    primary_url = source_urls[0]

    cache_key = f"{line}|{from_name}|{to_name}|{after_minutes}|{before_minutes}"
    cached = _rail_cache_get(cache_key)
    if cached is not None:
        return cached

    result = {
        "trains": [],
        "source_url": primary_url,
        "source_urls": source_urls,
        "source_label": RAILWAY_REFERENCE_LABEL,
        "line": line,
        "line_name": _rail_line_name(line),
        "from": from_name,
        "to": to_name,
        "timetable_version": RAILWAY_TIMETABLE_VERSIONS.get(line, "Unknown"),
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "error": None,
    }

    errors: List[str] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/126.0 Mobile Safari/537.36 OmniTouristOS/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=RAILWAY_HTTP_TIMEOUT_SECONDS,
            headers=headers,
        ) as client:
            for index, url in enumerate(source_urls):
                try:
                    response = await client.get(url)
                    if response.status_code != 200:
                        errors.append(f"HTTP {response.status_code}: {url}")
                        continue

                    parsed = _rail_parse_timetable_html(
                        html=response.text,
                        line=line,
                        from_station=from_station,
                        to_station=to_station,
                        source_url=str(response.url),
                    )

                    if not parsed and index == len(source_urls) - 1:
                        parsed = _rail_parse_mobile_timetable_html(
                            html=response.text,
                            line=line,
                            from_station=from_station,
                            to_station=to_station,
                            source_url=str(response.url),
                        )

                    if parsed:
                        result["trains"] = parsed
                        result["source_url"] = str(response.url)
                        result["source_used"] = "mobile" if "/m/" in str(response.url) else "timetable"
                        result["error"] = None
                        _rail_cache_set(cache_key, result)
                        return result

                    errors.append(f"No timetable rows parsed: {url}")
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    result["error"] = "Timetable source was reached, but no schedule rows could be parsed. " + " | ".join(errors[-3:])
    result["source_errors"] = errors
    _rail_cache_set(cache_key, result)
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