import os
import re
import secrets
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip().strip("\"'").strip()
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")

SYSTEM = (
    "Eres un verificador de datos para debates. Recibes lo último que se dijo "
    "(transcripción de voz, puede tener errores). Si hay una afirmación factual verificable "
    "(cifra, fecha, cita, hecho), búscala en la web y responde SOLO con este formato:\n"
    "AFIRMACIÓN: la afirmación en una frase corta\n"
    "VEREDICTO: Correcto | Incorrecto | Impreciso | No verificable\n"
    "FACTO: el dato correcto con cifras y año, máximo 2 frases cortas, sin introducciones\n"
    "Si solo hay opiniones o relleno, responde exactamente: NADA\n"
    "No escribas nada fuera de este formato."
)


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
    return FileResponse("index.html")


@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def sw():
    return FileResponse("sw.js", media_type="application/javascript")


@app.get("/icons/{name}")
def icon(name: str):
    if name not in ("icon-192.png", "icon-512.png", "apple-touch-icon.png"):
        raise HTTPException(404)
    return FileResponse(f"icons/{name}")


@app.get("/api/health")
def health():
    # Diagnóstico: no revela el código, solo si está configurado y cuántos caracteres tiene.
    return {"access_code_set": bool(ACCESS_CODE), "access_code_len": len(ACCESS_CODE), "api_key_set": bool(API_KEY), "model": MODEL}


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
        raise HTTPException(500, "Falta ANTHROPIC_API_KEY en el servidor")
    async with httpx.AsyncClient(timeout=60) as c:
        resp = await c.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 600,
                "system": SYSTEM,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                "messages": [
                    {
                        "role": "user",
                        "content": f"Contexto previo: {r.context}\n\nÚltimo fragmento: {r.chunk}",
                    }
                ],
            },
        )
    j = resp.json()
    if "error" in j:
        raise HTTPException(502, j["error"].get("message", "Error de la API"))

    text, cited, found = "", [], []
    for b in j.get("content", []):
        if b.get("type") == "text":
            text += b.get("text", "")
            for ci in b.get("citations") or []:
                if ci.get("url"):
                    cited.append(ci["url"])
        elif b.get("type") == "web_search_tool_result" and isinstance(b.get("content"), list):
            for x in b["content"]:
                if x.get("url"):
                    found.append(x["url"])

    afirm = _field("AFIRMACI[ÓO]N", text, ["VEREDICTO", "FACTO"])
    veredicto = _field("VEREDICTO", text, ["AFIRMACI[ÓO]N", "FACTO"]) or "No verificable"
    facto = _field("FACTO", text, ["AFIRMACI[ÓO]N", "VEREDICTO"])
    if not afirm or not facto:
        return {"card": None}

    fuentes, seen = [], set()
    for url in (cited or found[:2]):
        name = _host(url)
        if name not in seen:
            seen.add(name)
            fuentes.append({"name": name, "url": url})
    return {"card": {"afirmacion": afirm, "veredicto": veredicto, "facto": facto, "fuentes": fuentes[:3]}}
