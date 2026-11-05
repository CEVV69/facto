import os
import re
import secrets
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

API_KEY = os.environ.get("GEMINI_API_KEY", "")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip().strip("\"'").strip()
MODEL = os.environ.get("MODEL", "gemini-3.8-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

SYSTEM = (
    "Eres un verificador de datos para debates. Recibes una transcripción de voz (puede tener errores y estar cortada) "
    "con contexto previo. Evalúa el ÚLTIMO FRAGMENTO y usa el contexto para completar la idea si quedó a medias. "
    "Verifica cualquier afirmación factual comprobable: cifras, porcentajes, estadísticas, fechas, hechos históricos, "
    "citas atribuidas, comparaciones, resultados de encuestas o estudios; también si se presenta como 'se dice' o 'dicen que'. "
    "Búscala en la web y responde SOLO con este formato:\n"
    "AFIRMACIÓN: la afirmación en una frase corta\n"
    "VEREDICTO: Correcto | Incorrecto | Impreciso | No verificable\n"
    "FACTO: el dato correcto con cifras y año, máximo 2 frases cortas, sin introducciones\n"
    "Responde exactamente NADA solo si el fragmento no contiene ninguna afirmación factual comprobable "
    "(únicamente opiniones, saludos, preguntas o relleno).\n"
    "No escribas nada fuera de este formato."
)


NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}


class Req(BaseModel):
    chunk: str
    context: str = ""


def _ok(code: str) -> bool:
    sent = code.strip().strip("\"'").strip().encode()
    return bool(ACCESS_CODE) and secrets.compare_digest(sent, ACCESS_CODE.encode())


def _field(label: str, text: str, stops: list[str]) -> str:
    m = re.search(label + r":\s*(.+?)\s*(?=(?:" + "|".join(stops) + r"):|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def _host(url: str) -> str:
    h = urlparse(url).hostname or url
    return h[4:] if h.startswith("www.") else h


@app.get("/")
def home():
    return FileResponse("index.html", headers=NOCACHE)


@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json", headers=NOCACHE)


@app.get("/sw.js")
def sw():
    return FileResponse("sw.js", media_type="application/javascript", headers=NOCACHE)


@app.get("/icons/{name}")
def icon(name: str):
    if name not in ("icon-192.png", "icon-512.png", "apple-touch-icon.png"):
        raise HTTPException(404)
    return FileResponse(f"icons/{name}")


@app.get("/api/health")
def health():
    # Diagnóstico: no revela el código, solo si está configurado y cuántos caracteres tiene.
    return {"access_code_set": bool(ACCESS_CODE), "access_code_len": len(ACCESS_CODE), "api_key_set": bool(API_KEY), "model": MODEL, "provider": "gemini"}


@app.post("/api/check")
def check(x_access_code: str = Header(default="")):
    if not _ok(x_access_code):
        raise HTTPException(401, "Código de acceso incorrecto")
    return {"ok": True}


@app.post("/api/verify")
async def verify(r: Req, x_access_code: str = Header(default="")):
    if not _ok(x_access_code):
        raise HTTPException(401, "Código de acceso incorrecto")
    if not API_KEY:
        raise HTTPException(500, "Falta GEMINI_API_KEY en el servidor")
    async with httpx.AsyncClient(timeout=60) as c:
        resp = await c.post(
            GEMINI_URL,
            headers={"x-goog-api-key": API_KEY, "content-type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"Contexto previo: {r.context}\n\nÚltimo fragmento: {r.chunk}"}],
                    }
                ],
                "tools": [{"google_search": {}}],
                "generationConfig": {"maxOutputTokens": 700},
            },
        )
    j = resp.json()
    if "error" in j:
        raise HTTPException(502, j["error"].get("message", "Error de la API"))
    cand = (j.get("candidates") or [{}])[0]
    finish = cand.get("finishReason", "")
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))

    clean = re.sub(r"[*_#`>]", "", text)
    afirm = _field("AFIRMACI[ÓO]N", clean, ["VEREDICTO", "FACTO"])
    veredicto = _field("VEREDICTO", clean, ["AFIRMACI[ÓO]N", "FACTO"]) or "No verificable"
    facto = _field("FACTO", clean, ["AFIRMACI[ÓO]N", "VEREDICTO"])
    if not afirm or not facto:
        why = "nada" if re.search(r"\bNADA\b", clean) and not afirm else ("cortado" if finish == "MAX_TOKENS" else "formato")
        return {"card": None, "why": why, "stop": finish, "raw": clean.strip()[:300]}

    fuentes, seen = [], set()
    for chunk in (cand.get("groundingMetadata") or {}).get("groundingChunks") or []:
        web = chunk.get("web") or {}
        url = web.get("uri")
        if not url:
            continue
        name = web.get("title") or _host(url)
        if name not in seen:
            seen.add(name)
            fuentes.append({"name": name, "url": url})
    return {"card": {"afirmacion": afirm, "veredicto": veredicto, "facto": facto, "fuentes": fuentes[:3]}}
