# Genie 🧞‍♀️ — Architecture

```
Input (audio/video)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESS                                          │
│  audio.py   → load, resample, trim, features        │
│  video.py   → extract audio track + frames          │
│  frames.py  → quality assessment, frame selection   │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  ADAPTERS (each degrades gracefully if unavailable) │
│                                                      │
│  acoustic_adapter.py   ← always on (numpy only)     │
│  speech_emotion_adapter.py  ← HF wav2vec2 / fallback│
│  deepface_adapter.py   ← DeepFace (optional)        │
│  pyfeat_adapter.py     ← Py-FEAT (optional)         │
│  pyannote_adapter.py   ← diarization (optional)     │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  FUSION (fusion.py)                                  │
│  Weighted combination:                               │
│  Video: voice 50% + face 35% + acoustic 15%         │
│  Audio: voice 80% + acoustic 20%                    │
│  Poor quality → raise uncertainty automatically     │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  QUALITY + CONFIDENCE (quality.py)                   │
│  Signal quality → confidence level                  │
│  very_low | low | medium | medium_high | high        │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT (summarizer.py + schemas.py)                 │
│  GenieResult → JSON + text + WhatsApp + visuals      │
└─────────────────────────────────────────────────────┘
```

## Degradation model

Every adapter returns a ModalityResult with `available=True/False`.
If a module fails or is not installed, it returns `available=False` and
the fusion layer simply reduces its weight to zero. The pipeline never crashes.

## Acoustic rule engine

The core engine (`acoustic_adapter.py`) maps acoustic signal features
to state labels using psychoacoustic heuristics:

- ZCR variation → tense (vocal strain)
- Low energy + high silence → low_energy
- High speech rate → upbeat
- Many pauses → uncertain
- Steady energy + low ZCR → calm
- Sustained speech + pitch variation → engaged
