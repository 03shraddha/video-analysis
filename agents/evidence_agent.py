"""
Evidence Agent — saves incident files + sends HTML email via SendGrid.
Saves: full snapshot, face crop, vehicle crop, and JSON metadata.
Email includes embedded images (inline base64, no attachments needed).
"""
import uuid
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, MimeType
from jinja2 import Environment, FileSystemLoader
from models.schemas import DumpingDetection, AudioClue, EvidencePackage
import config

logger = logging.getLogger(__name__)

EVIDENCE_DIR = Path("evidence_log")
EVIDENCE_DIR.mkdir(exist_ok=True)

_jinja = Environment(loader=FileSystemLoader("templates"))


def _save_image(b64: str, path: Path) -> bool:
    try:
        path.write_bytes(base64.b64decode(b64))
        return True
    except Exception as e:
        logger.warning(f"Could not save image {path}: {e}")
        return False


def _save_evidence_files(
    incident_id: str,
    image_b64: str,
    detection: DumpingDetection,
) -> tuple[str, str, str]:
    """Save snapshot, face crop, vehicle crop. Returns (snapshot_path, face_path, vehicle_path)."""
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    base = EVIDENCE_DIR / f"{ts}_{incident_id}"

    snapshot_path = str(base) + ".jpg"
    _save_image(image_b64, Path(snapshot_path))

    face_path = ""
    if detection.persons and detection.persons[0].face_crop_b64:
        face_path = str(base) + "_face.jpg"
        _save_image(detection.persons[0].face_crop_b64, Path(face_path))

    vehicle_path = ""
    if detection.vehicles and detection.vehicles[0].vehicle_crop_b64:
        vehicle_path = str(base) + "_vehicle.jpg"
        _save_image(detection.vehicles[0].vehicle_crop_b64, Path(vehicle_path))

    # Save JSON metadata
    meta = {
        "incident_id": incident_id,
        "timestamp": datetime.utcnow().isoformat(),
        "location": config.LOCATION_LABEL,
        "detection": detection.model_dump(),
    }
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2, default=str))

    return snapshot_path, face_path, vehicle_path


def _send_email(package: EvidencePackage, snapshot_b64: str) -> bool:
    try:
        template = _jinja.get_template("evidence_email.html")
        html_body = template.render(
            package=package,
            snapshot_b64=snapshot_b64,
            face_b64=package.detection.persons[0].face_crop_b64 if package.detection.persons else None,
            vehicle_b64=package.detection.vehicles[0].vehicle_crop_b64 if package.detection.vehicles else None,
            plate=package.detection.vehicles[0].license_plate if package.detection.vehicles else None,
        )
        message = Mail(
            from_email=config.FROM_EMAIL,
            to_emails=config.RECIPIENT_EMAIL,
            subject=f"⚠️ Illegal Dumping Detected — {package.location} [{package.incident_id[:8]}]",
            html_content=html_body,
        )
        sg = SendGridAPIClient(config.SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code in (200, 202)
    except Exception as e:
        logger.error(f"SendGrid error: {e}")
        return False


def run_evidence_agent(
    image_b64: str,
    detection: DumpingDetection,
    audio: AudioClue,
) -> EvidencePackage:
    incident_id = str(uuid.uuid4())
    snapshot_path, face_path, vehicle_path = _save_evidence_files(incident_id, image_b64, detection)

    package = EvidencePackage(
        incident_id=incident_id,
        timestamp=datetime.utcnow(),
        location=config.LOCATION_LABEL,
        detection=detection,
        audio=audio,
        snapshot_path=snapshot_path,
        face_crop_path=face_path,
        vehicle_crop_path=vehicle_path,
        email_recipient=config.RECIPIENT_EMAIL,
    )

    package.email_sent = _send_email(package, image_b64)
    return package
