"""
Genie 🧞‍♀️ — Speech Emotion Adapter
Wraps x4nth055/emotion-recognition-using-speech (MIT).
Falls back to acoustic rule engine if model unavailable.

Source: https://github.com/x4nth055/emotion-recognition-using-speech (MIT)
Install: pip install speechbrain  # or the specific repo requirements
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..schemas import ModalityResult, EMOTION_TO_STATE


def _try_speechbrain():
    try:
        from speechbrain.pretrained import EncoderClassifier
        return EncoderClassifier
    except ImportError:
        return None


def _try_transformers_emotion():
    """Try HuggingFace transformers-based speech emotion model."""
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        return None


def analyze_audio_speech_emotion(
    audio_path: str,
    acoustic_features: dict = None,
) -> ModalityResult:
    """
    Run speech emotion recognition on an audio file.

    Priority chain:
      1. HuggingFace transformers pipeline (if available)
      2. Falls back to acoustic rule engine (always available)

    Never raises — always returns a ModalityResult.
    """
    notes: List[str] = []

    # Try HuggingFace transformers pipeline first
    pipeline_fn = _try_transformers_emotion()
    if pipeline_fn is not None:
        try:
            classifier = pipeline_fn(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                top_k=7,
            )
            results = classifier(audio_path)
            # results is list of {"label": ..., "score": ...}
            raw_scores: Dict[str, float] = {}
            for r in results:
                label = r["label"].lower()
                raw_scores[label] = float(r["score"])

            # Map to Genie states
            genie_scores: Dict[str, float] = {
                "calm": 0.0, "tense": 0.0, "upbeat": 0.0,
                "low_energy": 0.0, "uncertain": 0.0, "engaged": 0.0, "neutral": 0.0,
            }
            for label, score in raw_scores.items():
                state = EMOTION_TO_STATE.get(label, "neutral")
                genie_scores[state] = genie_scores.get(state, 0.0) + score

            total = sum(genie_scores.values())
            if total > 0:
                genie_scores = {k: v / total for k, v in genie_scores.items()}

            notes.append("speech emotion model: wav2vec2 (HuggingFace)")

            return ModalityResult(
                modality="voice",
                available=True,
                scores=genie_scores,
                raw_classes=raw_scores,
                notes=notes,
                model_used="wav2vec2-speech-emotion",
            )
        except Exception as e:
            notes.append(f"HuggingFace speech emotion failed: {e} — using acoustic engine")

    # Final fallback: acoustic rule engine
    if acoustic_features:
        from .acoustic_adapter import acoustic_features_to_states
        result = acoustic_features_to_states(acoustic_features)
        result.modality = "voice"
        result.notes = notes + result.notes + ["using acoustic rule engine (no ML model)"]
        return result

    return ModalityResult(
        modality="voice",
        available=False,
        notes=notes + ["no audio features available for speech emotion analysis"],
    )
