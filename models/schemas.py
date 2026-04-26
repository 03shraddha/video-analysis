from __future__ import annotations
from datetime import datetime
from typing import Literal
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
