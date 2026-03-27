"""
Genie 🧞‍♀️ — Schemas
All structured data types. Every output is one of these — machine-readable JSON always.

State labels are chosen for professional clarity:
  calm       — controlled, relaxed communication
  tense      — strain, stress, elevated urgency
  upbeat     — positive energy, enthusiasm
  low_energy — flat, tired, subdued
  uncertain  — hesitant, mixed signals
  engaged    — active, interested, participatory
  neutral    — no dominant signal

These map from raw model emotion classes internally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# State labels
# ---------------------------------------------------------------------------

STATE_LABELS = ["calm", "tense", "upbeat", "low_energy", "uncertain", "engaged", "neutral"]

CONFIDENCE_LEVELS = ["very_low", "low", "medium", "medium_high", "high"]

# Mapping from common emotion model classes to Genie state labels
EMOTION_TO_STATE: Dict[str, str] = {
    # Positive
    "happy":     "upbeat",
    "joy":       "upbeat",
    "excited":   "upbeat",
    "surprise":  "upbeat",
    "enthusiasm":"upbeat",
    # Negative activation
    "angry":     "tense",
    "anger":     "tense",
    "fear":      "tense",
    "disgust":   "tense",
    "stressed":  "tense",
    "anxious":   "tense",
    # Low activation
    "sad":       "low_energy",
    "sadness":   "low_energy",
    "tired":     "low_energy",
    "bored":     "low_energy",
    "depressed": "low_energy",
    # Neutral / uncertain
    "neutral":   "neutral",
    "calm":      "calm",
    "relaxed":   "calm",
    "contempt":  "uncertain",
    "confused":  "uncertain",
    # Engaged
    "focused":   "engaged",
    "attentive": "engaged",
}


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------

@dataclass
class SignalQuality:
    audio_clarity:      float = 0.0   # 0.0–1.0
    face_visibility:    float = 0.0   # 0.0–1.0
    speech_amount:      float = 0.0   # fraction of clip with speech
    usable_frames:      int   = 0
    total_frames:       int   = 0
    snr_estimate:       float = 0.0   # dB estimate
    notes:              List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual modality results
# ---------------------------------------------------------------------------

@dataclass
class ModalityResult:
    """Raw probabilities from one analysis modality."""
    modality:    str              # "voice" | "face" | "acoustic"
    available:   bool = False
    scores:      Dict[str, float] = field(default_factory=dict)
    raw_classes: Dict[str, float] = field(default_factory=dict)
    notes:       List[str]        = field(default_factory=list)
    model_used:  str = ""


# ---------------------------------------------------------------------------
# Segment (for meeting mode / diarization)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start_sec:     float
    end_sec:       float
    speaker_id:    str
    primary_state: str
    scores:        Dict[str, float]
    confidence:    str
    notes:         List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main output schema
# ---------------------------------------------------------------------------

@dataclass
class GenieResult:
    """
    The complete Genie output. Always JSON-serializable.
    This is the schema any agent, REST client, or CLI consumer receives.
    """
    tool:          str = "genie"
    version:       str = "1.0.0"
    input_file:    str = ""
    input_type:    str = ""        # "audio" | "video" | "audio_only"
    duration_sec:  float = 0.0
    mode:          str = "auto"    # "audio" | "video" | "meeting" | "auto"

    # Core output
    primary_state: str = "neutral"
    confidence:    str = "low"
    scores:        Dict[str, float] = field(default_factory=dict)

    # Signals used
    signals: Dict[str, bool] = field(default_factory=lambda: {
        "voice_used":       False,
        "face_used":        False,
        "diarization_used": False,
        "acoustic_used":    False,
    })

    # Explanatory notes
    notes:         List[str] = field(default_factory=list)

    # Optional rich data
    quality:          Optional[SignalQuality]    = None
    voice_result:     Optional[ModalityResult]   = None
    face_result:      Optional[ModalityResult]   = None
    acoustic_result:  Optional[ModalityResult]   = None
    segments:         List[Segment]               = field(default_factory=list)

    # Human-readable summary
    summary:          str = ""

    # Disclaimer — always present
    disclaimer: str = (
        "Best-effort state sketch from observable signals. "
        "Not mind reading. Not medical, legal, or psychological assessment."
    )

    # Errors / warnings from processing
    warnings:  List[str] = field(default_factory=list)
    errors:    List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove large nested objects from top-level for clean agent consumption
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_agent_json(self) -> str:
        """Compact agent-facing JSON — scores, signals, notes, disclaimer."""
        out = {
            "tool":          self.tool,
            "version":       self.version,
            "input_type":    self.input_type,
            "duration_sec":  round(self.duration_sec, 1),
            "primary_state": self.primary_state,
            "confidence":    self.confidence,
            "scores":        {k: round(v, 3) for k, v in self.scores.items()},
            "signals":       self.signals,
            "notes":         self.notes,
            "disclaimer":    self.disclaimer,
        }
        if self.segments:
            out["segments"] = [
                {
                    "start_sec":     s.start_sec,
                    "end_sec":       s.end_sec,
                    "speaker_id":    s.speaker_id,
                    "primary_state": s.primary_state,
                    "confidence":    s.confidence,
                }
                for s in self.segments
            ]
        return json.dumps(out, indent=2)
