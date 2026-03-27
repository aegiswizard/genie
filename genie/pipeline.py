"""
Genie 🧞‍♀️ — Pipeline
Main orchestrator. Accepts any supported media file, runs all available
modules, fuses results, and returns a GenieResult.

Mode selection:
  auto    → detects from file extension
  audio   → voice note / audio file only
  video   → selfie video / short clip (audio + face)
  meeting → multi-speaker, enables diarization
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from .config import GenieConfig, DEFAULT_CONFIG
from .schemas import GenieResult, SignalQuality
from .preprocess.audio import load_audio, extract_acoustic_features, estimate_snr
from .preprocess.video import extract_audio_from_video, extract_frames
from .preprocess.frames import select_best_frames
from .adapters.acoustic_adapter import acoustic_features_to_states
from .adapters.speech_emotion_adapter import analyze_audio_speech_emotion
from .adapters.deepface_adapter import analyze_frames_deepface
from .adapters.pyfeat_adapter import analyze_frames_pyfeat
from .adapters.pyannote_adapter import diarize_audio, segments_to_genie
from .fusion import fuse_modalities
from .quality import assess_signal_quality, compute_confidence
from .summarizer import format_text_report, format_agent_summary
from .utils import get_logger, make_temp_dir, cleanup_temp_dir, validate_input_file


logger = get_logger("genie.pipeline")


def run(
    input_path: str,
    config: Optional[GenieConfig] = None,
    mode: Optional[str] = None,
) -> GenieResult:
    """
    Run the full Genie pipeline on any supported media file.

    Args:
        input_path: Path to audio or video file.
        config:     GenieConfig (uses DEFAULT_CONFIG if None).
        mode:       Override mode: "audio" | "video" | "meeting" | "auto"

    Returns:
        GenieResult — always returns, never raises.
        On hard failure, result.errors will explain what went wrong.
    """
    cfg = config or DEFAULT_CONFIG
    if mode:
        cfg.mode = mode

    result = GenieResult(input_file=input_path)
    temp_dir = make_temp_dir(cfg.temp_dir)

    try:
        # ── 1. Validate input ───────────────────────────────────────────────
        is_valid, detected_type, err = validate_input_file(input_path)
        if not is_valid:
            result.errors.append(err or "Invalid input file")
            result.primary_state = "neutral"
            result.confidence    = "very_low"
            result.scores        = {l: 1/7 for l in ["calm","tense","upbeat","low_energy","uncertain","engaged","neutral"]}
            result.warnings.append("Could not process input file")
            return result

        result.input_type = detected_type
        has_video = (detected_type == "video")

        # Auto-mode resolution
        effective_mode = cfg.mode
        if effective_mode == "auto":
            effective_mode = "video" if has_video else "audio"
        result.mode = effective_mode

        # ── 2. Extract audio ────────────────────────────────────────────────
        audio_path = input_path
        if has_video:
            wav_path = os.path.join(temp_dir, "audio_extracted.wav")
            ok, warns = extract_audio_from_video(
                input_path, wav_path, cfg.audio.sample_rate
            )
            result.warnings.extend(warns)
            if ok:
                audio_path = wav_path
            else:
                result.warnings.append("Could not extract audio from video — face-only analysis")
                audio_path = None

        # ── 3. Load and preprocess audio ────────────────────────────────────
        waveform = None
        actual_sr = cfg.audio.sample_rate
        acoustic_features = {}

        if audio_path:
            waveform, actual_sr, load_warns = load_audio(
                audio_path, cfg.audio.sample_rate, temp_dir
            )
            result.warnings.extend(load_warns)

            if waveform is not None:
                result.duration_sec = round(len(waveform) / actual_sr, 2)
                acoustic_features   = extract_acoustic_features(waveform, actual_sr)
                snr                 = estimate_snr(waveform, actual_sr)
            else:
                result.warnings.append("Audio waveform could not be loaded")
                snr = 0.0
        else:
            snr = 0.0

        # ── 4. Extract video frames ─────────────────────────────────────────
        frame_paths = []
        if has_video:
            frames_dir = os.path.join(temp_dir, "frames")
            frame_paths, frame_warns = extract_frames(
                input_path, frames_dir,
                cfg.video.frame_interval_s,
                cfg.video.max_frames,
            )
            result.warnings.extend(frame_warns)
            frame_paths = select_best_frames(frame_paths, n=20)

        # ── 5. Acoustic analysis (always-on) ────────────────────────────────
        acoustic_result = None
        if acoustic_features:
            acoustic_result = acoustic_features_to_states(acoustic_features)
            result.acoustic_result = acoustic_result
            result.signals["acoustic_used"] = acoustic_result.available

        # ── 6. Speech emotion analysis ──────────────────────────────────────
        voice_result = None
        if audio_path and cfg.use_speech_emotion:
            voice_result = analyze_audio_speech_emotion(
                audio_path, acoustic_features
            )
            result.voice_result = voice_result
            result.signals["voice_used"] = voice_result.available

        # If voice model unavailable, fall back to acoustic as voice
        if voice_result is None or not voice_result.available:
            voice_result = acoustic_result

        # ── 7. Face analysis ────────────────────────────────────────────────
        face_result = None
        face_quality = 0.0

        if has_video and frame_paths:
            if cfg.use_pyfeat:
                face_result = analyze_frames_pyfeat(frame_paths)
            if (face_result is None or not face_result.available) and cfg.use_deepface:
                face_result = analyze_frames_deepface(frame_paths)

            if face_result and face_result.available:
                result.face_result = face_result
                result.signals["face_used"] = True
                face_quality = len(frame_paths) / max(cfg.video.max_frames, 1)
            else:
                if face_result:
                    result.warnings.extend(face_result.notes)

        # ── 8. Diarization (meeting mode) ────────────────────────────────────
        if effective_mode == "meeting" and audio_path and cfg.use_pyannote:
            raw_segments, diar_warns = diarize_audio(
                audio_path, cfg.huggingface_token
            )
            result.warnings.extend(diar_warns)
            if raw_segments:
                result.signals["diarization_used"] = True
                result.segments = segments_to_genie(
                    raw_segments,
                    lambda feats: acoustic_features_to_states(feats),
                    audio_path,
                    acoustic_features,
                )

        # ── 9. Fusion ────────────────────────────────────────────────────────
        fused_scores = fuse_modalities(
            voice_result=voice_result,
            face_result=face_result,
            acoustic_result=acoustic_result,
            has_video=has_video,
            face_quality=face_quality,
            weights=cfg.fusion,
        )
        result.scores = {k: round(v, 4) for k, v in fused_scores.items()}

        # Primary state
        result.primary_state = max(result.scores, key=lambda k: result.scores[k])

        # ── 10. Quality + confidence ─────────────────────────────────────────
        quality = assess_signal_quality(
            acoustic_features=acoustic_features,
            snr_estimate=snr,
            frame_paths=frame_paths,
            usable_frames=len(frame_paths),
        )
        result.quality    = quality
        result.confidence = compute_confidence(
            scores=result.scores,
            quality=quality,
            voice_available=bool(voice_result and voice_result.available),
            face_available=bool(face_result and face_result.available),
            thresholds=cfg.quality,
        )

        # ── 11. Collect notes ────────────────────────────────────────────────
        all_notes = []
        if voice_result and voice_result.notes:
            all_notes.extend(voice_result.notes)
        if face_result and face_result.notes:
            all_notes.extend(face_result.notes)
        if acoustic_result and acoustic_result.notes:
            for n in acoustic_result.notes:
                if n not in all_notes:
                    all_notes.append(n)
        if quality.notes:
            all_notes.extend(quality.notes)
        result.notes = all_notes[:6]  # cap for readability

        # ── 12. Summary ───────────────────────────────────────────────────────
        result.summary = format_agent_summary(result)

        return result

    except Exception as exc:
        logger.exception("Pipeline error")
        result.errors.append(f"Pipeline error: {exc}")
        result.primary_state = "neutral"
        result.confidence    = "very_low"
        if not result.scores:
            result.scores = {l: round(1/7, 4) for l in
                             ["calm","tense","upbeat","low_energy","uncertain","engaged","neutral"]}
        return result

    finally:
        if not cfg.keep_temp_files:
            cleanup_temp_dir(temp_dir)
