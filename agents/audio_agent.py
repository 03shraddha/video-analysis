"""
Audio Agent — OpenAI Whisper transcription + India-context sound classification.
Whisper handles Hindi, Marathi, and other regional languages natively.
"""
import base64
import json
import re
from openai import AsyncOpenAI
from models.schemas import AudioClue
import config

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

CLASSIFY_PROMPT = """
You received a transcription of surveillance audio from an alley in India.
Classify the dominant sounds and return ONLY valid JSON:

{
  "sound_type": "vehicle_engine|bag_dropping|voice|cycle_sounds|quiet",
  "transcript": "exact transcription or empty string",
  "confidence": 0.0-1.0
}

Sound type definitions:
- "vehicle_engine": any motorised vehicle (car, bike, auto-rickshaw, tempo)
- "bag_dropping": impact sounds of bags, sacks, or heavy objects being dropped/thrown
- "voice": human speech in any language — include transcript verbatim
- "cycle_sounds": non-motorised vehicle (cycle-van, handcart, pushcart)
- "quiet": ambient only, no relevant activity

This is legal evidence. Be precise.
"""


async def run_audio_agent(audio_b64: str) -> AudioClue:
    if not audio_b64:
        return AudioClue(sound_type="quiet", transcript="", confidence=0.0)

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return AudioClue(sound_type="quiet", transcript="", confidence=0.0)

    # Skip very short audio (< 1KB = probably silence)
    if len(audio_bytes) < 1000:
        return AudioClue(sound_type="quiet", transcript="", confidence=0.9)

    # Whisper transcription
    try:
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"
        transcript_response = await _client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
        transcript_text = str(transcript_response).strip()
    except Exception as e:
        transcript_text = f"[transcription error: {e}]"

    # Classify the transcription
    classify_response = await _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=150,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": f"Transcription: {transcript_text or '[silence]'}"},
        ],
    )

    raw = (classify_response.choices[0].message.content or "{}").strip()
    # Remove markdown code fences robustly (handles ```json, ```JSON, ``` with spaces)
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```\s*$', '', raw)

    try:
        data = json.loads(raw.strip())
        return AudioClue(
            sound_type=data.get("sound_type", "quiet"),
            transcript=transcript_text,
            confidence=data.get("confidence", 0.5),
        )
    except json.JSONDecodeError:
        return AudioClue(sound_type="quiet", transcript=transcript_text, confidence=0.3)
