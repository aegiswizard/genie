"""
Genie 🧞‍♀️ — Py-FEAT Adapter
Wraps Py-FEAT (MIT) for richer facial action units and expressions.
Optional heavier alternative to DeepFace.

Source: https://py-feat.org (MIT — note model-specific licenses apply)
Install: pip install feat
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..schemas import ModalityResult, EMOTION_TO_STATE


def _try_feat():
    try:
        from feat import Detector
        return Detector
    except ImportError:
        return None


def analyze_frames_pyfeat(frame_paths: List[str]) -> ModalityResult:
    """
    Analyze frames using Py-FEAT for emotion + action units.
    Returns unavailable if feat not installed.
    """
    Detector = _try_feat()
    if Detector is None:
        return ModalityResult(
            modality="face",
            available=False,
            notes=["Py-FEAT not installed — pip install feat"],
            model_used="pyfeat (unavailable)",
        )

    if not frame_paths:
        return ModalityResult(modality="face", available=False,
                               notes=["No frames for Py-FEAT analysis"])

    try:
        detector = Detector(emotion_model="resmasknet")
        all_emotion_scores = []
        notes = []
        usable = 0

        for path in frame_paths[:20]:  # limit for speed
            try:
                result = detector.detect_image(path)
                emotions = result.emotions
                if emotions is not None and len(emotions) > 0:
                    # Average across detected faces
                    emotion_means = emotions.mean(axis=0).to_dict()
                    all_emotion_scores.append(emotion_means)
                    usable += 1
            except Exception:
                continue

        if not all_emotion_scores:
            return ModalityResult(modality="face", available=False,
                                   notes=["Py-FEAT found no faces"], model_used="pyfeat")

        notes.append(f"Py-FEAT: {usable}/{len(frame_paths)} frames analyzed")

        # Average and map
        all_keys = set(k for d in all_emotion_scores for k in d)
        avg_emotions = {k: float(np.mean([d.get(k, 0.0) for d in all_emotion_scores]))
                        for k in all_keys}

        genie_scores: Dict[str, float] = {
            "calm": 0.0, "tense": 0.0, "upbeat": 0.0,
            "low_energy": 0.0, "uncertain": 0.0, "engaged": 0.0, "neutral": 0.0,
        }
        for emotion_label, prob in avg_emotions.items():
            state = EMOTION_TO_STATE.get(emotion_label.lower(), "neutral")
            genie_scores[state] = genie_scores.get(state, 0.0) + prob

        total = sum(genie_scores.values())
        if total > 0:
            genie_scores = {k: v / total for k, v in genie_scores.items()}

        return ModalityResult(
            modality="face",
            available=True,
            scores=genie_scores,
            raw_classes=avg_emotions,
            notes=notes,
            model_used="pyfeat_resmasknet",
        )

    except Exception as e:
        return ModalityResult(modality="face", available=False,
                               notes=[f"Py-FEAT error: {e}"], model_used="pyfeat")
