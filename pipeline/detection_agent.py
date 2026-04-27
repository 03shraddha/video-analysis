"""
Detection Agent — GPT-4o Vision.
Detects dumping activity, returns bounding boxes, person + vehicle evidence.
Bounding boxes are returned as image-fraction coordinates (0.0–1.0) so the
browser canvas can draw them without any local ML model.
"""
import json
import re
import base64
from io import BytesIO
from openai import AsyncOpenAI
from PIL import Image
from models.schemas import (
    DumpingDetection, BoundingBox, PersonEvidence, VehicleEvidence
)
import config

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a surveillance AI for an illegal dumping detection system deployed in India.
Analyze the video frame and return ONLY valid JSON — no markdown fences, no commentary.

Return this exact structure:
{
  "dumping_detected": true/false,
  "confidence": 0.0-1.0,
  "activity_description": "string",
  "dumped_material": "string (e.g. household bags, construction debris, e-waste, liquid waste)",
  "bboxes": [
    {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "label": "person|vehicle|face", "confidence": 0.0}
  ],
  "persons": [
    {
      "bbox": {"x":0.0,"y":0.0,"w":0.0,"h":0.0,"label":"person","confidence":0.0},
      "face_bbox": {"x":0.0,"y":0.0,"w":0.0,"h":0.0,"label":"face","confidence":0.0},
      "clothing_description": "e.g. blue striped shirt, white lungi",
      "physical_description": "e.g. ~40s male, average build, ~165cm",
      "carrying": "e.g. two black plastic bags, jute sacks"
    }
  ],
  "vehicles": [
    {
      "bbox": {"x":0.0,"y":0.0,"w":0.0,"h":0.0,"label":"vehicle","confidence":0.0},
      "vehicle_type": "motorcycle|tempo/LCV|auto-rickshaw|car|cycle-van|truck",
      "color_description": "e.g. white Maruti Alto-like",
      "license_plate": "MH 12 AB 3456 or null",
      "plate_confidence": 0.0,
      "plate_partial": false
    }
  ]
}

INSTRUCTIONS:
1. BOUNDING BOXES: Provide {x,y,w,h} as decimal fractions of image dimensions.
   x=0 is left edge, y=0 is top edge. Example: person in upper-left quarter →
   {"x":0.05,"y":0.05,"w":0.2,"h":0.35,"label":"person","confidence":0.9}

2. DUMPING ACTIVITY: Look for active disposal of trash, bags, rubble, furniture,
   electronics, liquids. Also flag suspicious loitering near a dumpster with bags.

3. PERSONS: Describe clothing precisely (colors, patterns, garment type).
   Physical description should match what a police report needs.
   Face bbox should be tight around the face for cropping.

4. VEHICLES — India includes: cars, motorcycles, auto-rickshaws, cycle-vans,
   tempos (LCVs), trucks. Read license plates character by character.
   India formats: "MH 12 AB 3456", "KA 05 MN 2345", "DL 4C AB 1234".
   If plate is partially obscured, read what is visible and set plate_partial=true.

5. This is legal evidence — be precise and factual. Do not speculate.
   If you cannot clearly see something, say so in the description.
"""


def _crop_region(image_b64: str, bbox: BoundingBox) -> str | None:
    """Crop a region from a base64 image, return as base64 JPEG."""
    try:
        data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(data)).convert("RGB")
        iw, ih = img.size
        x1 = max(0, int(bbox.x * iw))
        y1 = max(0, int(bbox.y * ih))
        x2 = min(iw, int((bbox.x + bbox.w) * iw))
        y2 = min(ih, int((bbox.y + bbox.h) * ih))
        if x2 <= x1 or y2 <= y1:
            return None
        cropped = img.crop((x1, y1, x2, y2))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


async def run_detection_agent(image_b64: str) -> DumpingDetection:
    response = await _client.chat.completions.create(
        model="gpt-5.5",
        max_completion_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "Analyze this surveillance frame."},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    # Strip markdown fences robustly (handles ```json, ```JSON, ``` with spaces)
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```\s*$', '', raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return DumpingDetection(dumping_detected=False, confidence=0.0,
                                activity_description="JSON parse error")

    # Parse bboxes — skip any entry that is malformed
    bboxes = []
    for b in data.get("bboxes", []):
        try:
            bboxes.append(BoundingBox(**b))
        except Exception:
            pass

    # Parse persons and attach face crops
    persons = []
    for p in data.get("persons", []):
        try:
            bbox = BoundingBox(**p["bbox"])
        except Exception:
            continue  # skip person with malformed bbox
        # Guard against empty dict or missing fields in face_bbox
        raw_face = p.get("face_bbox")
        face_bbox = None
        if raw_face and isinstance(raw_face, dict) and raw_face.get("w", 0) > 0:
            try:
                face_bbox = BoundingBox(**raw_face)
            except Exception:
                face_bbox = None
        person = PersonEvidence(
            bbox=bbox,
            face_bbox=face_bbox,
            clothing_description=p.get("clothing_description", ""),
            physical_description=p.get("physical_description", ""),
            carrying=p.get("carrying", ""),
        )
        # Crop face region
        if face_bbox:
            person.face_crop_b64 = _crop_region(image_b64, face_bbox)
        persons.append(person)

    # Parse vehicles and attach crops
    vehicles = []
    for v in data.get("vehicles", []):
        try:
            bbox = BoundingBox(**v["bbox"])
        except Exception:
            continue  # skip vehicle with malformed bbox
        vehicle = VehicleEvidence(
            bbox=bbox,
            vehicle_type=v.get("vehicle_type", ""),
            color_description=v.get("color_description", ""),
            license_plate=v.get("license_plate"),
            plate_confidence=v.get("plate_confidence", 0.0),
            plate_partial=v.get("plate_partial", False),
        )
        vehicle.vehicle_crop_b64 = _crop_region(image_b64, bbox)
        vehicles.append(vehicle)

    return DumpingDetection(
        dumping_detected=data.get("dumping_detected", False),
        confidence=data.get("confidence", 0.0),
        activity_description=data.get("activity_description", ""),
        dumped_material=data.get("dumped_material", ""),
        bboxes=bboxes,
        persons=persons,
        vehicles=vehicles,
    )
