import os
import secrets

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip().strip("\"'").strip()
MODEL = "claude-haiku-5-5"

SYSTEM = (
    "Eres asistente de verificación de datos en un debate. Recibes lo último que se dijo "
    "(transcripción de voz, puede tener errores). Si hay una afirmación factual verificable "
    "(cifra, fecha, cita, hecho), búscala en la web y responde en español con este formato exacto:\n"
    "AFIRMACIÓN: ...\nVEREDICTO: Correcto / Incorrecto / Impreciso / No verificable\n"
    "DATO: dato correcto y fuente, en 2 líneas máximo.\n"
    "Si solo hay opiniones o relleno, responde exactamente: NADA"
)


class Req(BaseModel):
    chunk: str
    context: str = ""


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
    return {"access_code_set": bool(ACCESS_CODE), "access_code_len": len(ACCESS_CODE), "api_key_set": bool(API_KEY)}


@app.post("/api/verify")
async def verify(r: Req, x_access_code: str = Header(default="")):
    sent = x_access_code.strip().strip("\"'").strip().encode()
    if not ACCESS_CODE or not secrets.compare_digest(sent, ACCESS_CODE.encode()):
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
    text, links = "", []
    for b in j.get("content", []):
        if b.get("type") == "text":
            text += b.get("text", "")
        if b.get("type") == "web_search_tool_result" and isinstance(b.get("content"), list):
            for x in b["content"]:
                if x.get("url"):
                    links.append({"url": x["url"], "title": x.get("title", "")})
    return {"text": text.strip(), "links": links[:3]}
