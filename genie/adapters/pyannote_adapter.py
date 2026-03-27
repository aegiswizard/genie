"""
Genie 🧞‍♀️ — Pyannote Adapter
Speaker diarization for meeting mode via pyannote.audio (MIT).

Source: https://github.com/pyannote/pyannote-audio (MIT)
Install: pip install pyannote.audio
Requires: HuggingFace token (free) — accept terms at hf.co/pyannote/speaker-diarization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..schemas import Segment


def _try_pyannote():
    try:
        from pyannote.audio import Pipeline
        return Pipeline
    except ImportError:
        return None


def diarize_audio(
    audio_path: str,
    hf_token: Optional[str] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Run speaker diarization on an audio file.
    Returns (segments_list, warnings).

    Each segment dict:
        {"start": float, "end": float, "speaker": str}

    Falls back gracefully if pyannote not available.
    """
    warnings: List[str] = []
    segments: List[Dict] = []

    Pipeline = _try_pyannote()
    if Pipeline is None:
        warnings.append("pyannote.audio not installed — pip install pyannote.audio")
        warnings.append("Meeting mode diarization unavailable — running as single-speaker")
        return segments, warnings

    if not hf_token:
        warnings.append("No HuggingFace token provided for pyannote (GENIE_HF_TOKEN env var)")
        warnings.append("Accept model terms at: https://hf.co/pyannote/speaker-diarization")
        return segments, warnings

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start":   round(turn.start, 2),
                "end":     round(turn.end, 2),
                "speaker": speaker,
            })

    except Exception as e:
        warnings.append(f"Diarization failed: {e}")

    return segments, warnings


def segments_to_genie(
    raw_segments: List[Dict],
    state_fn,
    audio_path: str,
    acoustic_features: dict,
) -> List[Segment]:
    """
    Convert diarization segments to Genie Segment objects.
    Runs state sketch on each segment individually.
    """
    from ..schemas import Segment
    genie_segments = []

    for seg in raw_segments:
        duration = seg["end"] - seg["start"]
        if duration < 1.0:
            continue  # Skip very short segments

        # Simple: use overall acoustic analysis per segment
        # (Full segment-level audio analysis would require re-slicing the waveform)
        result = state_fn(acoustic_features)
        primary = max(result.scores, key=lambda k: result.scores[k]) if result.scores else "neutral"

        genie_segments.append(Segment(
            start_sec=seg["start"],
            end_sec=seg["end"],
            speaker_id=seg["speaker"],
            primary_state=primary,
            scores=result.scores,
            confidence="low",   # segment-level confidence is always lower
            notes=[f"Speaker: {seg['speaker']}  Duration: {duration:.1f}s"],
        ))

    return genie_segments
