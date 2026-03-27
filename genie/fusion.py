"""
Genie 🧞‍♀️ — Fusion Layer
Weighted combination of voice, face, and acoustic modality results.

Fusion weights (configurable):
  Video mode:  voice 50% + face 35% + acoustic 15%
  Audio mode:  voice 80% + acoustic 20%
  Poor video:  reduce face weight, increase uncertainty
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import ModalityResult
from .config import FusionWeights

STATE_LABELS = ["calm", "tense", "upbeat", "low_energy", "uncertain", "engaged", "neutral"]


def fuse_modalities(
    voice_result: Optional[ModalityResult],
    face_result: Optional[ModalityResult],
    acoustic_result: Optional[ModalityResult],
    has_video: bool,
    face_quality: float = 1.0,
    weights: Optional[FusionWeights] = None,
) -> Dict[str, float]:
    """
    Weighted fusion of available modality results.

    Returns normalized probability dict over all Genie state labels.
    """
    if weights is None:
        weights = FusionWeights()

    fused = {label: 0.0 for label in STATE_LABELS}
    total_weight = 0.0

    def add_modality(result: Optional[ModalityResult], weight: float) -> None:
        nonlocal total_weight
        if result is None or not result.available or not result.scores:
            return
        for label in STATE_LABELS:
            fused[label] += result.scores.get(label, 0.0) * weight
        total_weight += weight

    if has_video:
        # Adjust face weight based on quality
        effective_face_weight = weights.face_video
        if face_quality < weights.face_quality_threshold:
            # Redistribute face weight to voice and uncertainty
            deficit = effective_face_weight * (1.0 - face_quality / weights.face_quality_threshold)
            effective_face_weight -= deficit

        add_modality(voice_result,    weights.voice_video)
        add_modality(face_result,     effective_face_weight)
        add_modality(acoustic_result, weights.acoustic_video)
    else:
        # Audio only
        add_modality(voice_result,    weights.voice_audio)
        add_modality(acoustic_result, weights.acoustic_audio)

    # If no modality contributed, use acoustic as sole source
    if total_weight == 0.0:
        if acoustic_result and acoustic_result.available and acoustic_result.scores:
            return dict(acoustic_result.scores)
        return {label: 1.0 / len(STATE_LABELS) for label in STATE_LABELS}

    # Normalize
    fused = {k: v / total_weight for k, v in fused.items()}

    return fused
