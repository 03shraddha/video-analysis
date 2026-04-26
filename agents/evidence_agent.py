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


def check_repeat_offender(plate: str | None, clothing_description: str) -> dict:
    """Scan past evidence JSON files for matching plate or clothing description.

    Returns a dict with keys: is_repeat, count, last_seen, match_type.
    match_type is 'plate' (exact plate hit), 'clothing' (3+ word overlap), or 'none'.
    Plate matches take priority over clothing matches in reporting.
    """
    _default = {"is_repeat": False, "count": 0, "last_seen": None, "match_type": "none"}
    try:
        json_files = sorted(EVIDENCE_DIR.glob("*.json"))
        if not json_files:
            return _default

        # Normalise inputs once
        norm_plate = plate.upper().replace(" ", "") if plate else None
        # Split clothing into a set of lowercase words, ignoring short filler words
        clothing_words = {w.lower() for w in clothing_description.split() if len(w) > 2}

        plate_matches = []
        clothing_matches = []

        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue  # skip malformed files silently

            detection = data.get("detection", {})

            # --- plate check ---
            if norm_plate:
                for vehicle in detection.get("vehicles", []):
                    past_plate = vehicle.get("license_plate") or ""
                    if past_plate.upper().replace(" ", "") == norm_plate:
                        plate_matches.append(data.get("timestamp"))
                        break  # one match per incident is enough

            # --- clothing check (only if no plate match already counted) ---
            for person in detection.get("persons", []):
                past_clothing = person.get("clothing_description") or ""
                past_words = {w.lower() for w in past_clothing.split() if len(w) > 2}
                if len(clothing_words & past_words) >= 3:
                    clothing_matches.append(data.get("timestamp"))
                    break  # one match per incident

        # Plate matches take priority
        if plate_matches:
            timestamps = sorted(filter(None, plate_matches))
            return {
                "is_repeat": True,
                "count": len(plate_matches),
                "last_seen": timestamps[-1] if timestamps else None,
                "match_type": "plate",
            }

        if clothing_matches:
            timestamps = sorted(filter(None, clothing_matches))
            return {
                "is_repeat": True,
                "count": len(clothing_matches),
                "last_seen": timestamps[-1] if timestamps else None,
                "match_type": "clothing",
            }

        return _default

    except Exception as e:
        logger.warning(f"check_repeat_offender failed: {e}")
        return _default


def _save_evidence_files(
    incident_id: str,
    image_b64: str,
    detection: DumpingDetection,
) -> tuple[str, str, str, dict]:
    """Save snapshot, face crop, vehicle crop, repeat_info."""
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

    # Check for repeat offender before saving (current incident not yet on disk)
    plate = detection.vehicles[0].license_plate if detection.vehicles else None
    clothing_desc = detection.persons[0].clothing_description if detection.persons else ""
    repeat_info = check_repeat_offender(plate, clothing_desc or "")

    # Save JSON metadata
    meta = {
        "incident_id": incident_id,
        "timestamp": datetime.utcnow().isoformat(),
        "location": config.LOCATION_LABEL,
        "detection": detection.model_dump(),
        "repeat_info": repeat_info,
    }
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2, default=str))

    return snapshot_path, face_path, vehicle_path, repeat_info


def _send_email(package: EvidencePackage, snapshot_b64: str, repeat_info: dict) -> bool:
    try:
        template = _jinja.get_template("evidence_email.html")
        html_body = template.render(
            package=package,
            snapshot_b64=snapshot_b64,
            face_b64=package.detection.persons[0].face_crop_b64 if package.detection.persons else None,
            vehicle_b64=package.detection.vehicles[0].vehicle_crop_b64 if package.detection.vehicles else None,
            plate=package.detection.vehicles[0].license_plate if package.detection.vehicles else None,
            repeat_info=repeat_info,
            severity_level=package.severity_level,
        )

        # Prepend repeat-offender warning to the subject when applicable
        base_subject = f"⚠️ Illegal Dumping Detected — {package.location} [{package.incident_id[:8]}]"
        if repeat_info.get("is_repeat"):
            subject = f"⚠️ REPEAT OFFENDER (seen {repeat_info['count']}x) — {base_subject}"
        else:
            subject = base_subject

        # Route critical/high severity to escalation address
        recipient = (
            config.ESCALATION_EMAIL
            if package.severity_level in ("CRITICAL", "HIGH")
            else config.RECIPIENT_EMAIL
        )
        message = Mail(
            from_email=config.FROM_EMAIL,
            to_emails=recipient,
            subject=subject,
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
    severity = config.classify_severity(detection.dumped_material)
    snapshot_path, face_path, vehicle_path, repeat_info = _save_evidence_files(
        incident_id, image_b64, detection
    )

    recipient = (
        config.ESCALATION_EMAIL if severity in ("CRITICAL", "HIGH") else config.RECIPIENT_EMAIL
    )
    package = EvidencePackage(
        incident_id=incident_id,
        timestamp=datetime.utcnow(),
        location=config.LOCATION_LABEL,
        detection=detection,
        audio=audio,
        snapshot_path=snapshot_path,
        face_crop_path=face_path,
        vehicle_crop_path=vehicle_path,
        email_recipient=recipient,
        repeat_info=repeat_info,
        severity_level=severity,
    )

    package.email_sent = _send_email(package, image_b64, repeat_info)
    return package
