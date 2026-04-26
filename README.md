# Dumping Watch

Illegal dumping detector for public spaces. Camera watches an alley, AI analyzes every frame, and fires an evidence email the moment someone dumps trash. Built for India - reads MH/KA/DL plates, understands Hindi/Marathi audio.

## Pipeline

```mermaid
flowchart LR
    CAM[Webcam / IP Camera] --> CA[Capture Agent]
    CA -->|motion detected| DA[Detection Agent]
    CA -->|motion detected| AA[Audio Agent]
    DA --> EA[Evidence Agent]
    AA --> EA
    EA --> EMAIL[SendGrid Email]
```

## Agents

| Agent | Model | Job |
|---|---|---|
| Capture | Python only | Pixel-diff motion + blur/brightness checks. Skips dark, blurry, and empty frames before any API call. |
| Detection | GPT-4.1 Vision | Draws bounding boxes, reads license plates, crops face and vehicle. |
| Audio | gpt-4o-transcribe + gpt-4.1-mini | Transcribes audio in any Indian language; classifies engine sounds, bag drops, voices. |
| Evidence | Python | Saves face/vehicle crops + JSON. Detects repeat offenders (plate + clothing match). Classifies severity (CRITICAL/HIGH/MEDIUM/LOW) and routes to the right authority. Fires HTML email via SendGrid. |

## Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + Uvicorn |
| Orchestration | OpenAI Agents SDK |
| Vision | GPT-4.1 (high detail mode) |
| Audio | gpt-4o-transcribe + gpt-4.1-mini |
| Alerts | SendGrid |
| Frontend | WebRTC + Canvas (no framework) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn app:app --reload
```

Open `localhost:8000` and grant camera access.

## Keys needed

`OPENAI_API_KEY` / `SENDGRID_API_KEY` / `RECIPIENT_EMAIL` / `FROM_EMAIL` / `LOCATION_LABEL` / `ESCALATION_EMAIL` (optional - receives CRITICAL/HIGH severity alerts; falls back to `RECIPIENT_EMAIL`)

## UI

Live feed with real-time bounding boxes (red for people, amber for vehicles), face crop display, plate readout, and a scrolling incident log. Each confirmed dump saves 4 files: full snapshot, face crop, vehicle crop, JSON metadata.

No facial recognition database, no Aadhaar lookup. Pure CCTV-grade evidence package for police handoff.
