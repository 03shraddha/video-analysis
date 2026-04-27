"""
FastAPI application — main entry point.
Pipeline: Capture → Guardrail → (Detection + Audio in parallel) → Evidence.

The input guardrail (cheap model pre-screen) runs between the motion check and the
expensive GPT call — frames with no visible disposal activity are dropped early.
The output guardrail gates evidence filing on a confidence threshold.
SDK tracing wraps each frame so every run is visible at platform.openai.com/traces.
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from agents import trace  # OpenAI Agents SDK — sends traces to platform.openai.com/traces

from pipeline.capture_agent import run_capture_agent
from pipeline.detection_agent import run_detection_agent
from pipeline.audio_agent import run_audio_agent
from pipeline.evidence_agent import run_evidence_agent
from pipeline import guardrails
from models.schemas import IncomingFrame
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Dumping Watch")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/evidence_log", StaticFiles(directory="evidence_log"), name="evidence_log")
templates = Jinja2Templates(directory="templates")

# --- State ---
_prev_frame_b64: str | None = None
_incident_count: int = 0
_frame_count: int = 0
_guardrail_skips: int = 0  # frames blocked by input guardrail (token saves counter)


# --- WebSocket manager ---
class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self.connections.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


ws_manager = WSManager()


# --- Request model for /analyze ---
class AnalyzeRequest(BaseModel):
    image_base64: str
    audio_base64: str
    session_id: str = "default"


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "location": config.LOCATION_LABEL,
    })


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    global _prev_frame_b64, _incident_count, _frame_count, _guardrail_skips
    _frame_count += 1

    with trace("frame_analysis", metadata={"frame": _frame_count, "session": req.session_id}):
        frame = IncomingFrame(
            image_base64=req.image_base64,
            audio_base64=req.audio_base64,
            timestamp=datetime.utcnow(),
            session_id=req.session_id,
            location_label=config.LOCATION_LABEL,
        )

        # Stage 1: Capture — motion + quality check (pure Python, no API cost)
        capture = run_capture_agent(frame.image_base64, frame.audio_base64, _prev_frame_b64)
        _prev_frame_b64 = frame.image_base64

        await ws_manager.broadcast({"type": "agent_status", "agent": "Capture", "state": capture.motion_flag})

        if capture.motion_flag == "SKIP":
            return {"skipped": True, "reason": capture.reason, "frames": _frame_count}

        # Stage 2: Input guardrail — semantic pre-screen with cheap model
        # Drops frames with no visible disposal activity before the expensive vision call.
        await ws_manager.broadcast({"type": "agent_status", "agent": "Guardrail", "state": "processing"})
        guardrail_ok, guardrail_reason = await guardrails.input_guardrail(capture.image_b64)

        if not guardrail_ok:
            _guardrail_skips += 1
            await ws_manager.broadcast({
                "type": "agent_status",
                "agent": "Guardrail",
                "state": "skip",
                "guardrail_skips": _guardrail_skips,
            })
            return {
                "skipped": True,
                "reason": f"guardrail:{guardrail_reason}",
                "frames": _frame_count,
                "guardrail_skips": _guardrail_skips,
            }

        await ws_manager.broadcast({
            "type": "agent_status",
            "agent": "Guardrail",
            "state": "done",
            "guardrail_skips": _guardrail_skips,
        })

        # Stage 3: Vision + Audio in parallel (expensive — only reached after guardrail passes)
        await ws_manager.broadcast({"type": "agent_status", "agent": "Vision", "state": "processing"})
        await ws_manager.broadcast({"type": "agent_status", "agent": "Audio", "state": "processing"})

        detection, audio = await asyncio.gather(
            run_detection_agent(capture.image_b64),
            run_audio_agent(capture.audio_b64),
        )

        # Broadcast bbox updates for live canvas overlay
        await ws_manager.broadcast({
            "type": "bbox_update",
            "boxes": [b.model_dump() for b in detection.bboxes],
            "detection": {
                "dumping_detected": detection.dumping_detected,
                "confidence": detection.confidence,
                "activity": detection.activity_description,
                "persons": [
                    {
                        "clothing": p.clothing_description,
                        "physical": p.physical_description,
                        "face_b64": p.face_crop_b64,
                    }
                    for p in detection.persons
                ],
                "vehicles": [
                    {
                        "type": v.vehicle_type,
                        "plate": v.license_plate,
                        "plate_confidence": v.plate_confidence,
                        "vehicle_b64": v.vehicle_crop_b64,
                    }
                    for v in detection.vehicles
                ],
            },
            "audio": {"sound_type": audio.sound_type, "transcript": audio.transcript},
        })

        await ws_manager.broadcast({"type": "agent_status", "agent": "Vision", "state": "done"})
        await ws_manager.broadcast({"type": "agent_status", "agent": "Audio", "state": "done"})

        # Stage 4: Output guardrail — confidence gate before evidence / email
        file_evidence, evidence_reason = guardrails.output_guardrail(detection)

        if file_evidence:
            await ws_manager.broadcast({"type": "agent_status", "agent": "Evidence", "state": "processing"})

            evidence = run_evidence_agent(frame.image_base64, detection, audio)
            _incident_count += 1

            await ws_manager.broadcast({
                "type": "incident",
                "incident_id": evidence.incident_id,
                "timestamp": evidence.timestamp.isoformat(),
                "location": evidence.location,
                "activity": detection.activity_description,
                "material": detection.dumped_material,
                "plate": detection.vehicles[0].license_plate if detection.vehicles else None,
                "person_clothing": detection.persons[0].clothing_description if detection.persons else "",
                "face_b64": detection.persons[0].face_crop_b64 if detection.persons else None,
                "vehicle_b64": detection.vehicles[0].vehicle_crop_b64 if detection.vehicles else None,
                "email_sent": evidence.email_sent,
                "incident_count": _incident_count,
            })

            await ws_manager.broadcast({"type": "agent_status", "agent": "Evidence", "state": "done"})
            return {"incident_filed": True, "incident_id": evidence.incident_id, "email_sent": evidence.email_sent}

        return {
            "incident_filed": False,
            "dumping_detected": detection.dumping_detected,
            "confidence": detection.confidence,
            "output_guardrail_reason": evidence_reason,
            "frames": _frame_count,
        }


@app.get("/evidence")
async def list_evidence():
    items = []
    for f in sorted(Path("evidence_log").glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(f.read_text())
            items.append(data)
        except Exception:
            continue
    return JSONResponse(content={"incidents": items, "total": len(items)})


@app.get("/evidence/{incident_id}")
async def get_evidence(incident_id: str):
    for f in Path("evidence_log").glob(f"*{incident_id}*.json"):
        return JSONResponse(content=json.loads(f.read_text()))
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/health")
async def health():
    from openai import AsyncOpenAI
    openai_ok = False
    try:
        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        await client.models.list()
        openai_ok = True
    except Exception:
        pass

    sendgrid_ok = bool(config.SENDGRID_API_KEY and len(config.SENDGRID_API_KEY) > 10)
    return {
        "status": "ok",
        "openai": "connected" if openai_ok else "error",
        "sendgrid": "configured" if sendgrid_ok else "missing key",
        "location": config.LOCATION_LABEL,
        "threshold": config.CONFIDENCE_THRESHOLD,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
