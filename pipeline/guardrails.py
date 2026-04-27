"""
Guardrails for the dumping detection pipeline.

input_guardrail:  semantic pre-screen with a cheap vision model — skips frames that
                  clearly have no disposal activity before the expensive model is called.
                  Errs on the side of passing (fail-open) to avoid missing real incidents.

output_guardrail: confidence gate — prevents low-confidence detections from filing
                  evidence or triggering email alerts.
"""
import logging
from openai import AsyncOpenAI
from models.schemas import DumpingDetection
import config

logger = logging.getLogger(__name__)
_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# Permissive: only block frames clearly lacking any disposal activity.
# A false negative (missed real incident) is worse than a false positive here.
_PRESCREEN_PROMPT = (
    "Surveillance camera pre-screen. "
    "Is there a person or vehicle visible who might be disposing of waste, garbage, or debris? "
    "Answer YES if you see any human or vehicle carrying bags, boxes, or waste near a street or alley, even if unsure. "
    "Answer NO only if the frame clearly shows no humans or vehicles with potential waste. "
    "Reply with ONLY: YES or NO"
)


async def input_guardrail(image_b64: str) -> tuple[bool, str]:
    """Cheap semantic pre-screen before the expensive vision model.

    Returns (passed, reason).
    passed=False  →  skip the expensive gpt-5.5 detection call entirely.
    On any API error, returns True so the main pipeline is never blocked.

    Cost: gpt-4.1-mini + detail=low ≈ ~100 tokens vs ~2000+ for full detection.
    """
    try:
        resp = await _client.chat.completions.create(
            model="gpt-4.1-mini",
            max_tokens=5,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",  # ~85 image tokens vs ~1105 for "high"
                            },
                        },
                        {"type": "text", "text": _PRESCREEN_PROMPT},
                    ],
                }
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("YES"):
            return True, "activity_possible"
        logger.info("input_guardrail: no disposal activity detected — skipping expensive detection")
        return False, "no_activity"
    except Exception as e:
        logger.warning(f"input_guardrail error ({e}) — passing frame through")
        return True, "error_passthrough"


def output_guardrail(detection: DumpingDetection) -> tuple[bool, str]:
    """Confidence gate before filing evidence or sending an alert.

    Returns (file_evidence, reason).
    file_evidence=False  →  don't call evidence agent, don't send email.
    """
    if not detection.dumping_detected:
        return False, "no_dumping"
    if detection.confidence < config.CONFIDENCE_THRESHOLD:
        return False, f"low_conf_{detection.confidence:.0%}"
    return True, "confirmed"
