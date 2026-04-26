"""
Capture Agent — motion pre-screen before hitting paid APIs.
Pure Python (Pillow), no LLM call. Runs on every frame at 2 FPS.
~90% of quiet-alley frames will be flagged SKIP.
"""
import base64
from io import BytesIO
from PIL import Image, ImageFilter
from models.schemas import CaptureResult

# Pixel-difference threshold: higher = less sensitive to motion
MOTION_THRESHOLD = 12
MIN_IMAGE_BYTES = 5000  # reject tiny/corrupt images


def _b64_to_gray(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("L").resize((160, 120))


def detect_motion(curr_b64: str, prev_b64: str | None) -> tuple[bool, float]:
    """Return (motion_detected, avg_pixel_diff)."""
    if not prev_b64:
        return True, 99.0   # first frame — always analyze

    curr = _b64_to_gray(curr_b64)
    prev = _b64_to_gray(prev_b64)

    diff = 0.0
    curr_px = list(curr.getdata())
    prev_px = list(prev.getdata())
    for c, p in zip(curr_px, prev_px):
        diff += abs(c - p)
    avg_diff = diff / len(curr_px)
    return avg_diff >= MOTION_THRESHOLD, round(avg_diff, 2)


def validate_image(b64: str) -> tuple[bool, str]:
    """Return (ok, reason)."""
    try:
        data = base64.b64decode(b64)
        if len(data) < MIN_IMAGE_BYTES:
            return False, f"image too small ({len(data)} bytes)"
        img = Image.open(BytesIO(data))
        width, height = img.size
        if width < 100 or height < 100:
            return False, f"resolution too low ({width}x{height})"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def run_capture_agent(image_b64: str, audio_b64: str, prev_frame_b64: str | None) -> CaptureResult:
    quality_ok, quality_reason = validate_image(image_b64)
    if not quality_ok:
        return CaptureResult(
            motion_flag="SKIP",
            quality_ok=False,
            reason=quality_reason,
        )

    has_motion, diff = detect_motion(image_b64, prev_frame_b64)
    if not has_motion:
        return CaptureResult(
            motion_flag="SKIP",
            quality_ok=True,
            reason=f"no motion (diff={diff})",
        )

    return CaptureResult(
        motion_flag="ANALYZE",
        quality_ok=True,
        reason=f"motion detected (diff={diff})",
        image_b64=image_b64,
        audio_b64=audio_b64,
    )
