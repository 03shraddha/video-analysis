from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel


class IncomingFrame(BaseModel):
    image_base64: str       # base64 JPEG from WebRTC canvas capture
    audio_base64: str       # base64 WAV chunk (~500ms)
    timestamp: datetime
    session_id: str
    location_label: str


class BoundingBox(BaseModel):
    x: float            # fraction from left edge (0.0–1.0)
    y: float            # fraction from top edge
    w: float            # fraction of image width
    h: float            # fraction of image height
    label: str          # "person" | "vehicle" | "face"
    confidence: float


class PersonEvidence(BaseModel):
    bbox: BoundingBox
    face_bbox: BoundingBox | None = None    # tighter box just around face
    face_crop_b64: str | None = None        # cropped face JPEG, base64
    clothing_description: str = ""
    physical_description: str = ""
    carrying: str = ""


class VehicleEvidence(BaseModel):
    bbox: BoundingBox
    vehicle_crop_b64: str | None = None
    vehicle_type: str = ""              # "motorcycle", "tempo/LCV", "auto-rickshaw", "car", "cycle-van"
    color_description: str = ""
    license_plate: str | None = None    # "MH 12 AB 3456"
    plate_confidence: float = 0.0
    plate_partial: bool = False


class DumpingDetection(BaseModel):
    dumping_detected: bool
    confidence: float
    activity_description: str = ""
    dumped_material: str = ""           # "household bags", "construction debris", "e-waste"
    bboxes: list[BoundingBox] = []      # all boxes for canvas overlay
    persons: list[PersonEvidence] = []
    vehicles: list[VehicleEvidence] = []


class AudioClue(BaseModel):
    sound_type: str = "quiet"           # "vehicle_engine" | "bag_dropping" | "voice" | "cycle_sounds" | "quiet"
    transcript: str = ""
    confidence: float = 0.0


class CaptureResult(BaseModel):
    motion_flag: Literal["ANALYZE", "SKIP"]
    quality_ok: bool
    reason: str = ""
    image_b64: str = ""
    audio_b64: str = ""
    blur_score: float = 0.0         # populated by capture agent; higher = blurrier
    brightness_score: float = 0.0   # populated by capture agent; used to log skip reason


class PipelineResult(BaseModel):
    skipped: bool = False
    incident_filed: bool = False
    reason: str = ""
    detection: DumpingDetection | None = None
    evidence: "EvidencePackage | None" = None


class EvidencePackage(BaseModel):
    incident_id: str
    timestamp: datetime
    location: str
    detection: DumpingDetection
    audio: AudioClue
    snapshot_path: str
    face_crop_path: str = ""
    vehicle_crop_path: str = ""
    email_sent: bool = False
    email_recipient: str = ""
    repeat_info: dict[str, Any] = {}    # e.g. {"plate": "MH12AB3456", "seen_count": 3}
    severity_level: str = "LOW"         # "LOW" | "MEDIUM" | "HIGH"


@dataclass
class SessionContext:
    """Per-camera session state. Replaces module-level globals in app.py.
    Compatible with OpenAI Agents SDK RunContextWrapper pattern.
    """
    session_id: str
    prev_frame_b64: str | None = None
    incident_count: int = 0
    frame_count: int = 0
    recent_plates: list[str] = field(default_factory=list)        # last 20 seen plates for dedup
    recent_incident_ids: list[str] = field(default_factory=list)  # last 50 incident IDs

    def record_plate(self, plate: str | None) -> None:
        if plate:
            self.recent_plates = (self.recent_plates + [plate])[-20:]

    def record_incident(self, incident_id: str) -> None:
        self.recent_incident_ids = (self.recent_incident_ids + [incident_id])[-50:]
