# 🧞‍♀️ Genie — State Sketch Engine Skill

**Version:** 1.0.0 · **License:** MIT · **Source:** https://github.com/aegiswizard/genie  
**Compatible with:** OpenClaw · Hermes · Claude agents · Any Python agent · REST API

---

## What This Skill Does

Genie returns a best-effort state sketch from a short voice note or video clip.  
It reads observable signals — voice strain, energy, rhythm, face cues — and returns
a probability distribution over human communication states.

**Reads the room, not your soul.**

---

## Trigger Phrases

- `"analyze this voice note"`
- `"read the room on this clip"`
- `"what state is the person in"`
- `"genie analyze [file]"`
- `"state sketch [file]"`
- `"how does this person sound"`
- `"meeting room read"`
- `"is this person tense / calm / upbeat"`

---

## Setup

```bash
git clone https://github.com/aegiswizard/genie.git
cd genie
pip install -e .          # core — works immediately, numpy only
pip install librosa       # better audio analysis (recommended)
pip install deepface      # face analysis for video (optional)
# ffmpeg must be installed for video and non-WAV audio
```

---

## CLI

```bash
# Voice note
genie analyze voice.m4a

# Short video
genie analyze selfie.mp4

# Meeting clip
genie analyze meeting.mp4 --mode meeting

# JSON output for agents
genie analyze voice.m4a --output json

# WhatsApp-style reply
genie analyze voice.m4a --output whatsapp

# With PNG bar chart
genie analyze voice.m4a --visuals --visuals-dir ./output

# Check what's available on this machine
genie doctor
```

---

## Python API

```python
from genie import analyze

result = analyze("voice.m4a")

result.primary_state   # "calm" | "tense" | "upbeat" | "low_energy" | "uncertain" | "engaged" | "neutral"
result.confidence      # "very_low" | "low" | "medium" | "medium_high" | "high"
result.scores          # {"calm": 0.41, "tense": 0.21, ...}
result.notes           # ["voice strain elevated", ...]
result.summary         # one-line agent summary
result.to_agent_json() # compact JSON for agent consumption
result.to_json()       # full JSON
```

---

## Output JSON

```json
{
  "tool": "genie",
  "version": "1.0.0",
  "input_type": "audio",
  "duration_sec": 18.4,
  "primary_state": "calm",
  "confidence": "medium",
  "scores": {
    "calm": 0.41, "uncertain": 0.21,
    "low_energy": 0.16, "tense": 0.13, "upbeat": 0.09,
    "engaged": 0.0, "neutral": 0.0
  },
  "signals": {"voice_used": true, "face_used": false, "acoustic_used": true},
  "notes": ["speech rhythm steady", "moderate energy"],
  "disclaimer": "Best-effort state sketch from observable signals. Not mind reading."
}
```

---

## REST API (`genie serve`)

```bash
genie serve --port 7070    # requires: pip install fastapi uvicorn
```

```bash
# Upload file
curl -X POST http://localhost:7070/analyze \
  -F "file=@voice.m4a" -F "mode=auto"

# Local path
curl -X POST http://localhost:7070/analyze/path \
  -F "path=/path/to/voice.m4a"

# Doctor
curl http://localhost:7070/doctor
```

---

## State Labels

| Label | Meaning |
|-------|---------|
| `calm` | Controlled, relaxed, steady |
| `tense` | Strain, stress, urgency |
| `upbeat` | Positive energy, enthusiasm |
| `low_energy` | Flat, tired, subdued |
| `uncertain` | Hesitant, mixed signals |
| `engaged` | Active, interested, participatory |
| `neutral` | No dominant signal |

---

## Modes

| Mode | Input | What runs |
|------|-------|-----------|
| `auto` | any | detects from file type |
| `audio` | voice note | acoustic + speech emotion |
| `video` | selfie clip | audio + face + fused |
| `meeting` | multi-speaker | diarization + per-segment sketch |

---

## Important

Genie generates best-effort state sketches from observable signals.  
Do **not** use for medical, legal, hiring, policing, or high-stakes decisions.  
Results reflect observable acoustic and visual patterns only.
