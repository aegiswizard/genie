"""
Genie 🧞‍♀️ — DeepFace Adapter
Wraps DeepFace (MIT) for facial emotion and attribute analysis.
Gracefully unavailable if DeepFace not installed.

Source: https://github.com/serengil/deepface (MIT)
Install: pip install deepface
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..schemas import ModalityResult, EMOTION_TO_STATE


def _try_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        return None


def analyze_frames_deepface(frame_paths: List[str]) -> ModalityResult:
    """
    Analyze a list of frame image paths using DeepFace.
    Averages emotion probabilities across all usable frames.
    Never raises — returns unavailable result on any failure.
    """
    DeepFace = _try_deepface()
    if DeepFace is None:
        return ModalityResult(
            modality="face",
            available=False,
            notes=["DeepFace not installed — pip install deepface"],
            model_used="deepface (unavailable)",
        )

    if not frame_paths:
        return ModalityResult(
            modality="face",
            available=False,
            notes=["No frames provided for face analysis"],
        )

    all_emotion_scores: List[Dict[str, float]] = []
    usable = 0
    notes: List[str] = []

    for path in frame_paths:
        try:
            results = DeepFace.analyze(
                img_path=path,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # DeepFace returns a list when multiple faces detected
            if isinstance(results, list):
                results = results[0]

            emotion_dict = results.get("emotion", {})
            if emotion_dict:
                # Normalize to 0–1
                total = sum(emotion_dict.values())
                if total > 0:
                    normalized = {k: v / total for k, v in emotion_dict.items()}
                    all_emotion_scores.append(normalized)
                    usable += 1

        except Exception:
            continue  # Skip frames where no face detected or analysis fails

    if not all_emotion_scores:
        return ModalityResult(
            modality="face",
            available=False,
            notes=["No faces detected in any frame"],
            model_used="deepface",
        )

    notes.append(f"Face analysis: {usable}/{len(frame_paths)} frames usable")

    # Average across frames
    all_keys = set(k for d in all_emotion_scores for k in d)
    avg_emotions: Dict[str, float] = {}
    for key in all_keys:
        avg_emotions[key] = float(np.mean([d.get(key, 0.0) for d in all_emotion_scores]))

    # Map to Genie state labels
    genie_scores: Dict[str, float] = {
        "calm": 0.0, "tense": 0.0, "upbeat": 0.0,
        "low_energy": 0.0, "uncertain": 0.0, "engaged": 0.0, "neutral": 0.0,
    }
    for emotion_label, prob in avg_emotions.items():
        state = EMOTION_TO_STATE.get(emotion_label.lower(), "neutral")
        genie_scores[state] = genie_scores.get(state, 0.0) + prob

    # Normalize
    total = sum(genie_scores.values())
    if total > 0:
        genie_scores = {k: v / total for k, v in genie_scores.items()}

    # Add interpretation notes
    top_state = max(genie_scores, key=lambda k: genie_scores[k])
    if genie_scores.get("neutral", 0) > 0.5:
        notes.append("face cues mostly neutral")
    if genie_scores.get("upbeat", 0) > 0.3:
        notes.append("positive facial expressions detected")
    if genie_scores.get("tense", 0) > 0.3:
        notes.append("stress indicators in facial expression")

    return ModalityResult(
        modality="face",
        available=True,
        scores=genie_scores,
        raw_classes=avg_emotions,
        notes=notes,
        model_used="deepface",
    )
