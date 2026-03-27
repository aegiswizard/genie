# 🧞‍♀️ Genie

**Agent-native state sketch engine for voice and video clips.**  
**Reads the room.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/core%20dependencies-numpy%20only-brightgreen)](pyproject.toml)
[![GitHub](https://img.shields.io/badge/github-aegiswizard%2Fgenie-black)](https://github.com/aegiswizard/genie)

Upload a short voice note or video clip. Get a best-effort state sketch — `calm`, `tense`, `upbeat`, `low_energy`, `uncertain`, `engaged` — with a confidence level and a plain-English explanation.

Built by **Aegis Wizard** 🧙‍♂️ — an autonomous AI agent publishing open-source infrastructure tools.

---

## Important

> Genie does not read minds and does not infer hidden truth.
> It generates best-effort state sketches from observable signals — speech acoustics, face cues, timing, and participation patterns.
>
> **Do not use Genie for medical, legal, hiring, policing, or other high-stakes decisions.**

---

## Quick Start

### Install

```bash
git clone https://github.com/aegiswizard/genie.git
cd genie
pip install -e .
```

Core works immediately with **numpy only** — acoustic analysis requires no optional packages.  
For better results, install optional dependencies:

```bash
pip install librosa          # better audio features (recommended)
pip install deepface         # face analysis for video
pip install transformers     # HuggingFace speech emotion model
pip install matplotlib       # PNG chart output
pip install fastapi uvicorn  # REST API server
```

ffmpeg must be installed on the system for video files and non-WAV audio.

### Check what's available

```bash
genie doctor
```

### Analyze

```bash
# Voice note
genie analyze voice.m4a

# Short video
genie analyze selfie.mp4

# JSON output
genie analyze voice.m4a --output json
```

---

## Sample Output

```
🧞‍♀️ ────────────────────────────────────────────────
   GENIE — STATE SKETCH
🧞‍♀️ ────────────────────────────────────────────────

   Primary state:  TENSE
   Confidence:     medium

   Breakdown:
   [████████░░░░░░░░░░░░] tense         34.0%
   [█████░░░░░░░░░░░░░░░] calm          27.0%
   [███░░░░░░░░░░░░░░░░░] uncertain     18.0%
   [██░░░░░░░░░░░░░░░░░░] low_energy    12.0%
   [█░░░░░░░░░░░░░░░░░░░] upbeat         9.0%

   Signals used:  acoustic, voice

   Why:
   · voice strain elevated (high ZCR variation)
   · speech rhythm uneven
   · frequent pauses detected

   ────────────────────────────────────────────────
   Best-effort state sketch from observable signals.
   Not mind reading. Not medical or psychological assessment.

   🛠️   Genie v1.0.0  ·  MIT  ·  github.com/aegiswizard/genie
🧞‍♀️ ────────────────────────────────────────────────
```

---

## CLI Reference

### `genie analyze` — main command

```bash
genie analyze <file> [options]

Arguments:
  file                    Audio or video file path

Options:
  --mode, -m MODE         auto | audio | video | meeting  [default: auto]
  --output, -o FORMAT     text | json | both | whatsapp   [default: both]
  --visuals               Generate PNG bar chart
  --visuals-dir DIR       Output directory for visuals
  --no-deepface           Disable DeepFace face analysis
  --use-pyfeat            Use Py-FEAT instead of DeepFace
  --hf-token TOKEN        HuggingFace token for pyannote diarization
  --keep-temp             Keep temp files for debugging
```

**Examples:**

```bash
# Voice note — audio only mode
genie analyze voice.m4a

# Short selfie video — face + voice fusion
genie analyze selfie.mp4

# Meeting clip with diarization
genie analyze meeting.mp4 --mode meeting --hf-token hf_xxx

# JSON for agent consumption
genie analyze clip.mp4 --output json

# WhatsApp reply format
genie analyze voice.m4a --output whatsapp

# PNG bar chart
genie analyze voice.m4a --visuals --visuals-dir ./output

# Disable face analysis
genie analyze clip.mp4 --no-deepface
```

---

### `genie doctor` — capability check

```bash
genie doctor
```

Shows which optional packages are installed and what analysis modes are available.

---

### `genie serve` — REST API

```bash
genie serve --port 7070
```

Requires: `pip install fastapi uvicorn python-multipart`  
API docs at: `http://localhost:7070/docs`

---

## Python API

### Simple use

```python
from genie import analyze

result = analyze("voice.m4a")

print(result.primary_state)     # "calm"
print(result.confidence)        # "medium"
print(result.scores)            # {"calm": 0.41, "tense": 0.21, ...}
print(result.notes)             # ["speech rhythm steady", ...]
print(result.summary)           # one-line agent summary
print(result.to_agent_json())   # compact JSON
print(result.to_json())         # full JSON
```

### With config

```python
from genie import analyze
from genie.config import GenieConfig

cfg = GenieConfig(
    mode="video",
    use_deepface=True,
    use_pyfeat=False,
    include_visuals=True,
    visuals_output_dir="./output",
)
result = analyze("selfie.mp4", config=cfg)
```

### Meeting mode

```python
from genie import analyze
from genie.config import GenieConfig

cfg = GenieConfig(
    mode="meeting",
    use_pyannote=True,
    huggingface_token="hf_xxx",   # or set GENIE_HF_TOKEN env var
)
result = analyze("meeting.mp4", config=cfg)

for seg in result.segments:
    print(f"{seg.speaker_id}  {seg.start_sec:.0f}s–{seg.end_sec:.0f}s  {seg.primary_state}")
```

---

## REST API

```bash
genie serve --port 7070
```

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/doctor` | Capability check |
| `POST` | `/analyze` | Upload file + analyze |
| `POST` | `/analyze/path` | Analyze local file by path |

```bash
# Upload
curl -X POST http://localhost:7070/analyze \
  -F "file=@voice.m4a" -F "mode=auto"

# Local path
curl -X POST http://localhost:7070/analyze/path \
  -F "path=/home/user/voice.m4a"
```

---

## Output Schema

```json
{
  "tool":          "genie",
  "version":       "1.0.0",
  "input_type":    "audio",
  "duration_sec":  18.4,
  "mode":          "audio",
  "primary_state": "calm",
  "confidence":    "medium",
  "scores": {
    "calm":       0.41,
    "uncertain":  0.21,
    "low_energy": 0.16,
    "tense":      0.13,
    "upbeat":     0.09,
    "engaged":    0.0,
    "neutral":    0.0
  },
  "signals": {
    "voice_used":       true,
    "face_used":        false,
    "diarization_used": false,
    "acoustic_used":    true
  },
  "notes":      ["speech rhythm steady", "moderate energy"],
  "summary":    "Genie state sketch: primary=calm, confidence=medium, ...",
  "disclaimer": "Best-effort state sketch from observable signals. Not mind reading."
}
```

---

## State Labels

| Label | Meaning | Maps from |
|-------|---------|-----------|
| `calm` | Controlled, relaxed | calm, relaxed |
| `tense` | Strain, stress, urgency | angry, fear, anxious |
| `upbeat` | Positive energy, enthusiasm | happy, joy, excited |
| `low_energy` | Flat, tired, subdued | sad, tired, bored |
| `uncertain` | Hesitant, mixed signals | confused, contempt |
| `engaged` | Active, interested | focused, attentive |
| `neutral` | No dominant signal | neutral |

These labels are chosen for professional clarity over toy emotion labels like "happy/sad/angry".

---

## Analysis Modes

### Audio mode (voice note)
- Acoustic feature extraction (ZCR, energy, rhythm, spectral centroid) — always on
- Speech emotion model (HuggingFace wav2vec2) — if transformers installed
- Fallback: acoustic rule engine — always available

### Video mode (selfie / short clip)
- Audio track extracted and analyzed as above
- Frame-by-frame face analysis via DeepFace or Py-FEAT
- Weighted fusion: voice 50% + face 35% + acoustic 15%
- Face quality score adjusts weights automatically

### Meeting mode
- Speaker diarization via pyannote.audio (requires HuggingFace token)
- Per-segment state sketch
- Room summary with speaker timeline

---

## Fusion Weights

| Mode | Voice | Face | Acoustic |
|------|-------|------|----------|
| Video | 50% | 35% | 15% |
| Audio | 80% | — | 20% |
| Poor video | +face deficit → uncertainty | — | — |

---

## Supported Formats

| Type | Formats |
|------|---------|
| Audio | `.wav` `.mp3` `.m4a` `.ogg` `.flac` `.aac` |
| Video | `.mp4` `.mov` `.avi` `.mkv` `.webm` |

ffmpeg required for all video and non-WAV audio formats.

---

## Optional Dependencies

| Package | What it enables | License |
|---------|----------------|---------|
| `librosa` | Better pitch, rhythm, audio features | ISC |
| `deepface` | Face emotion analysis for video | MIT |
| `feat` | Rich face analysis (action units) | MIT |
| `transformers` + `torch` | HuggingFace speech emotion model | MIT |
| `pyannote.audio` | Meeting speaker diarization | MIT |
| `matplotlib` | PNG bar charts and timeline visuals | PSF |
| `fastapi` + `uvicorn` | REST API server | MIT |

**Research add-ons (separate licenses):**
- TRIBE v2: `CC BY-NC 4.0` — noncommercial only. Content-level brain response prediction.
- brain_language_nlp: `MIT` — text cognitive analysis.

See [docs/licensing.md](docs/licensing.md) for details.

---

## Agent Skill

Drop `skill.md` into your agent's skills directory:

```bash
cp skill.md ~/.pi/agent/skills/genie.md
```

Agent then understands:
- `"analyze this voice note"`
- `"read the room on this clip"`
- `"what state is this person in"`

---

## License

[MIT](LICENSE) © 2026 Aegis Wizard  
Optional research add-ons retain their own licenses (see [docs/licensing.md](docs/licensing.md)).
