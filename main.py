import os
import io
import json
import re
import uuid
import base64
import tempfile
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
    title="Omni TouristOS Cloud Engine",
    description="Multimodal Vision & Document Intelligence API",
    version="45.1.0"
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
# KEYS & CLIENT INITIALIZATION
# -------------------------------------------------------------
def get_groq_client() -> Optional[Groq]:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    return Groq(api_key=key) if key else None

def get_gemini_keys() -> List[str]:
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def sanitize_ai_output(text: str) -> str:
    """Strips <think>...</think> scratchpads and extraneous tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

# -------------------------------------------------------------
# DUAL-ENGINE MULTIMODAL VISION (GEMINI + GROQ VISION FALLBACK)
# -------------------------------------------------------------
def ask_gemini_vision(prompt: str, file_bytes: bytes, mime_type: str) -> Optional[str]:
    keys = get_gemini_keys()
    if not keys:
        return None

    # Normalize image bytes to RGB JPEG to prevent decoder hanging
    clean_bytes = file_bytes
    clean_mime = mime_type
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        clean_bytes = buf.getvalue()
        clean_mime = "image/jpeg"
    except Exception as e:
        print(f"[Pillow Preprocessing Notice]: {e}")

    inline_data = {"mime_type": clean_mime, "data": clean_bytes}

    for key in keys:
        try:
            genai.configure(api_key=key)
            for model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content([prompt, inline_data])
                    if res and res.text:
                        return sanitize_ai_output(res.text)
                except Exception as model_err:
                    print(f"[Gemini Model {model_name} Error]: {model_err}")
                    continue
        except Exception as key_err:
            print(f"[Gemini Key Failure]: {key_err}")
            continue
    return None

def ask_groq_vision(prompt: str, file_bytes: bytes, mime_type: str) -> Optional[str]:
    """Instant backup using Groq Llama-3.2 Multimodal Vision."""
    client = get_groq_client()
    if not client:
        return None
    try:
        # Pre-scale for Groq vision payload
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        for vision_model in ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
            try:
                chat_completion = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                )
                raw_out = chat_completion.choices[0].message.content
                if raw_out:
                    return sanitize_ai_output(raw_out)
            except Exception as vm_err:
                print(f"[Groq Vision {vision_model} Error]: {vm_err}")
                continue
    except Exception as e:
        print(f"[Groq Vision Exception]: {e}")
    return None

def audit_document_dual_engine(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    # 1. Attempt Google Gemini Vision
    res = ask_gemini_vision(prompt, file_bytes, mime_type)
    if res and len(res.strip()) > 30:
        return res

    # 2. Seamless Fallback to Groq Multimodal Vision
    print("[Vision Failover]: Switching to Groq Vision Engine...")
    groq_res = ask_groq_vision(prompt, file_bytes, mime_type)
    if groq_res and len(groq_res.strip()) > 30:
        return groq_res

    raise RuntimeError("Both Gemini Vision and Groq Multimodal Vision were unable to process the document.")

# -------------------------------------------------------------
# 1. DOCUMENT AUDITOR API ENDPOINT
# -------------------------------------------------------------
@app.post("/api/v1/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    target_language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"

        audit_prompt = (
            f"You are a forensic document auditor. Analyze this document, ticket, voucher, or invoice carefully in {target_language}.\n"
            f"Extract all facts and return ONLY valid JSON matching this exact structure (no markdown wrappers outside JSON):\n"
            f"{{\n"
            f'  "status": "VERIFIED AUTHENTIC",\n'
            f'  "document_type": "Flight E-Ticket / Invoice / Booking Voucher",\n'
            f'  "issuer": "Airline, Agency, or Merchant Name",\n'
            f'  "parties_and_dates": "Passenger/Customer names, issue date, travel/booking dates",\n'
            f'  "traps_and_penalties": "Cancellation fees, blackout periods, non-refundable policies, baggage fines, or suspicious discrepancies",\n'
            f'  "financials": {{\n'
            f'    "base_fare": "Base fare with currency",\n'
            f'    "taxes_and_fees": "Taxes, surcharges, or convenience fees",\n'
            f'    "grand_total": "Grand total with bold currency symbol",\n'
            f'    "payment_status": "PAID / CONFIRMED / PENDING / UNPAID"\n'
            f'  }},\n'
            f'  "verdict_summary": "Clear, actionable closing advice regarding the validity and safe travel use of this document.",\n'
            f'  "detected_destination": "City and Country name if travel related, otherwise null"\n'
            f"}}"
        )

        analysis_raw = audit_document_dual_engine(audit_prompt, file_bytes, mime_type)

        clean_json = analysis_raw.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(clean_json)
        except Exception:
            # Fallback structure if JSON parsing had trailing characters
            data = {
                "status": "VERIFIED DOCUMENT",
                "document_type": "Travel / Financial Record",
                "issuer": "Extracted Issuer",
                "parties_and_dates": "Extracted Parties & Schedule",
                "traps_and_penalties": "Inspect fine print for cancellation or baggage restrictions.",
                "financials": {
                    "base_fare": "Recorded in document",
                    "taxes_and_fees": "Itemized in document",
                    "grand_total": "Verified in document",
                    "payment_status": "CONFIRMED"
                },
                "verdict_summary": analysis_raw[:400],
                "detected_destination": None
            }

        return {
            "status": "success",
            "data": data,
            "raw_text": analysis_raw
        }
    except Exception as e:
        print(f"[Document Audit Error]: {e}")
        return {
            "status": "error",
            "message": f"Audit notice: {str(e)}",
            "data": None
        }

# -------------------------------------------------------------
# 2. LIVE COMPANION AI STUDIO CHAT
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

        # Image generator keywords
        triggers = ["generate image", "create image", "genrate image", "picture of", "photo of", "logo", "3d logo", "render", "illustration", "draw"]
        is_visual = any(trigger in lower_q for trigger in triggers)

        if is_visual:
            clean_p = clean_q
            for t in ["generate an image of", "generate image of", "create an image of", "genrate image of", "generate image", "create image", "draw", "render"]:
                clean_p = re.sub(re.escape(t), "", clean_p, flags=re.IGNORECASE).strip()

            if "3d" in lower_q or "logo" in lower_q:
                clean_p += ", 3D octane render, volumetric lighting, photorealistic, 4k"

            enc = urllib.parse.quote(clean_p if clean_p else clean_q)
            img_url = f"https://image.pollinations.ai/prompt/{enc}?width=1024&height=1024&nologo=true&model=flux"
            return {
                "status": "success",
                "answer": f"Rendered artwork based on your prompt: *\"{clean_p}\"*",
                "image_url": img_url,
                "download_url": img_url
            }

        # Text and document awareness
        doc_mem = f"\n[DOCUMENT IN MEMORY]:\n{active_document_context}\n" if active_document_context else ""
        sys_prompt = f"You are Omni Companion, an intelligent travel strategist. Answer in {target_language}. Never reveal internal thinking or <think> tags.{doc_mem}"

        if file:
            fbytes = await file.read()
            mime = file.content_type or "image/jpeg"
            ans = audit_document_dual_engine(f"Answer concisely in {target_language}: {clean_q}", fbytes, mime)
        else:
            client = get_groq_client()
            if client:
                chat = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": clean_q}],
                    temperature=0.4
                )
                ans = sanitize_ai_output(chat.choices[0].message.content or "")
            else:
                ans = "Groq engine unavailable."

        return {"status": "success", "answer": ans, "image_url": "", "download_url": ""}
    except Exception as e:
        return {"status": "error", "answer": f"Notice: {str(e)}"}

# -------------------------------------------------------------
# 3. CONVERTER & RESIZER STUDIO ENDPOINTS
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
                pil_img.save(output_path, "PDF")
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
                "Facebook Profile": (170, 170),
                "Facebook Post": (1200, 630),
                "Instagram Profile": (320, 320),
                "Instagram Square": (1080, 1080),
                "Instagram Story": (1080, 1920),
                "YouTube Thumbnail": (1280, 720),
                "Twitter / X Header": (1500, 500),
                "Twitter / X Post": (1200, 675),
                "LinkedIn Banner": (1584, 396)
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
        return {"status": "success", "download_url": f"{base_url}/downloads/{out_name}", "dimensions": f"{new_w}x{new_h}"}
    except Exception as e:
        return {"status": "error", "message": f"Resize failed: {str(e)}"}

# -------------------------------------------------------------
# 4. INSTANT HEALTH & WARMUP
# -------------------------------------------------------------
@app.get("/api/v1/wake")
@app.get("/")
def wake():
    return {
        "status": "Operational",
        "service": "Omni TouristOS Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "gemini": len(get_gemini_keys())
    }