import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
LOCATION_LABEL = os.getenv("LOCATION_LABEL", "Unknown Location")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL", RECIPIENT_EMAIL)  # fallback to main recipient

SEVERITY_KEYWORDS = {
    "CRITICAL": ["chemical", "hazardous", "toxic", "sewage", "biological", "medical", "asbestos", "industrial waste"],
    "HIGH":     ["construction", "debris", "rubble", "concrete", "bricks", "metal", "tyres", "e-waste", "electronic"],
    "MEDIUM":   ["furniture", "mattress", "appliance", "large", "bulk"],
    "LOW":      ["bag", "plastic", "household", "garbage", "litter", "food"],
}

def classify_severity(dumped_material: str) -> str:
    """Return CRITICAL/HIGH/MEDIUM/LOW based on dumped_material description."""
    material_lower = dumped_material.lower()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if any(kw in material_lower for kw in SEVERITY_KEYWORDS[level]):
            return level
    return "MEDIUM"  # default if unclear

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")
if not SENDGRID_API_KEY:
    raise ValueError("SENDGRID_API_KEY is not set in .env")
