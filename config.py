import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
LOCATION_LABEL = os.getenv("LOCATION_LABEL", "Unknown Location")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")
if not SENDGRID_API_KEY:
    raise ValueError("SENDGRID_API_KEY is not set in .env")
