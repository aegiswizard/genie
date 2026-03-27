"""
Genie 🧞‍♀️ — Quality Assessment
Confidence scoring from signal quality + model agreement.
Confidence is half the product.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import SignalQuality, ModalityResult
from .config import QualityThresholds


def assess_signal_quality(
    acoustic_features: dict,
    snr_estimate: float,
    frame_paths: List[str],
    usable_frames: int,
) -> SignalQuality:
    """Assess overall signal quality for confidence calculation."""
    q = SignalQuality()

    silence_ratio    = acoustic_features.get("silence_ratio", 1.0)
    speech_duration  = acoustic_features.get("speech_duration_s", 0.0)
    energy_rms       = acoustic_features.get("energy_rms", 0.0)

    # Audio clarity: based on SNR and energy
    snr_norm = min(snr_estimate / 30.0, 1.0)
    energy_ok = 1.0 if energy_rms > 0.01 else 0.2
    q.audio_clarity = (snr_norm * 0.7 + energy_ok * 0.3)

    # Speech amount
    q.speech_amount = max(0.0, 1.0 - silence_ratio)

    # Face visibility
    q.total_frames   = len(frame_paths)
    q.usable_frames  = usable_frames
    if q.total_frames > 0:
        q.face_visibility = usable_frames / q.total_frames
    else:
        q.face_visibility = 0.0

    q.snr_estimate = snr_estimate

    # Quality notes
    if q.audio_clarity < 0.3:
        q.notes.append("poor audio quality")
    if q.speech_amount < 0.2:
        q.notes.append("very little speech detected")
    if q.face_visibility < 0.3 and q.total_frames > 0:
        q.notes.append("limited face visibility")
    if q.total_frames == 0:
        q.notes.append("no video frames — audio-only analysis")

    return q


def compute_confidence(
    scores: Dict[str, float],
    quality: SignalQuality,
    voice_available: bool,
    face_available: bool,
    thresholds: Optional[QualityThresholds] = None,
) -> str:
    """
    Compute confidence level from signal quality and model agreement.

    Returns one of: very_low | low | medium | medium_high | high
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    # Start with a base score out of 1.0
    conf_score = 0.0

    # Modality availability
    if voice_available:
        conf_score += 0.30
    if face_available:
        conf_score += 0.20

    # Audio quality
    conf_score += quality.audio_clarity * 0.20

    # Speech amount
    if quality.speech_amount >= thresholds.min_speech_fraction:
        conf_score += 0.15
    else:
        conf_score -= 0.15

    # Margin between top and second label
    if len(scores) >= 2:
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        if margin >= thresholds.min_margin_for_high:
            conf_score += 0.20
        elif margin >= thresholds.min_margin_for_medium:
            conf_score += 0.10
        else:
            conf_score -= 0.10

    # Face visibility (if video)
    if quality.total_frames > 0:
        if quality.face_visibility >= thresholds.min_face_visibility:
            conf_score += 0.10
        else:
            conf_score -= 0.10

    # Map to label
    conf_score = max(0.0, min(1.0, conf_score))

    if conf_score >= 0.80:
        return "high"
    elif conf_score >= 0.65:
        return "medium_high"
    elif conf_score >= 0.45:
        return "medium"
    elif conf_score >= 0.25:
        return "low"
    else:
        return "very_low"
