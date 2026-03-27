"""
Genie 🧞‍♀️ — Acoustic State Adapter
Core always-on fallback. Maps acoustic features to state labels.
No ML model required. Works on any machine with numpy.

This is the engine that makes Genie work immediately on install.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..schemas import ModalityResult, EMOTION_TO_STATE


def acoustic_features_to_states(features: dict) -> ModalityResult:
    """
    Map extracted acoustic features to Genie state label probabilities.

    This is rule-based signal analysis — interpretable, fast, always available.
    Based on psychoacoustic research on vocal communication:
      - High ZCR variation → tense (vocal strain)
      - Low energy, high silence ratio → low_energy
      - High speech rate → upbeat or tense
      - Low speech rate, moderate energy → calm
      - High pause count + moderate energy → uncertain
      - High energy variation → engaged or tense
    """
    if not features:
        return ModalityResult(
            modality="acoustic",
            available=False,
            notes=["No acoustic features — audio could not be loaded"],
        )

    scores: Dict[str, float] = {label: 0.0 for label in
                                 ["calm", "tense", "upbeat", "low_energy",
                                  "uncertain", "engaged", "neutral"]}

    notes: List[str] = []

    # Extract features with safe defaults
    energy_rms       = features.get("energy_rms", 0.0)
    zcr_std          = features.get("zcr_std", 0.0)
    energy_variation = features.get("energy_variation", 0.0)
    silence_ratio    = features.get("silence_ratio", 0.5)
    speech_rate      = features.get("speech_rate_estimate", 0.0)
    pause_count      = features.get("pause_count", 0)
    spectral_centroid= features.get("spectral_centroid", 0.0)
    speech_duration  = features.get("speech_duration_s", 0.0)
    pitch_variation  = features.get("pitch_variation", 0.0)

    # ── CALM signals ─────────────────────────────────────────────────────
    # Low ZCR variation, moderate energy, low pause count, regular rhythm
    calm_score = 0.0
    if zcr_std < 0.02:
        calm_score += 0.35
    if 0.01 < energy_rms < 0.15:
        calm_score += 0.25
    if energy_variation < 0.5:
        calm_score += 0.20
    if silence_ratio < 0.50 and pause_count < 5:
        calm_score += 0.20
    scores["calm"] = min(calm_score, 1.0)

    # ── TENSE signals ─────────────────────────────────────────────────────
    # High ZCR variation, elevated spectral centroid, uneven energy
    tense_score = 0.0
    if zcr_std > 0.035:
        tense_score += 0.40
        notes.append("voice strain elevated (high ZCR variation)")
    if spectral_centroid > 2500:
        tense_score += 0.20
        notes.append("high spectral brightness (vocal tension indicator)")
    if energy_variation > 0.8:
        tense_score += 0.20
        notes.append("speech rhythm uneven")
    if pitch_variation > 0.3:
        tense_score += 0.20
    scores["tense"] = min(tense_score, 1.0)

    # ── UPBEAT signals ────────────────────────────────────────────────────
    # High energy, fast speech rate, low silence
    upbeat_score = 0.0
    if energy_rms > 0.12:
        upbeat_score += 0.30
    if speech_rate > 4.0:
        upbeat_score += 0.35
        notes.append("fast speaking pace")
    if silence_ratio < 0.30:
        upbeat_score += 0.20
    if spectral_centroid > 2000:
        upbeat_score += 0.15
    scores["upbeat"] = min(upbeat_score, 1.0)

    # ── LOW ENERGY signals ────────────────────────────────────────────────
    # Low energy, high silence, slow speech rate
    low_energy_score = 0.0
    if energy_rms < 0.03:
        low_energy_score += 0.40
        notes.append("low vocal energy")
    if silence_ratio > 0.60:
        low_energy_score += 0.30
        notes.append("high silence ratio")
    if speech_rate < 1.5 and speech_duration > 2.0:
        low_energy_score += 0.30
        notes.append("slow speaking pace")
    scores["low_energy"] = min(low_energy_score, 1.0)

    # ── UNCERTAIN signals ─────────────────────────────────────────────────
    # Many pauses, moderate everything, no dominant signal
    uncertain_score = 0.0
    if pause_count > 8:
        uncertain_score += 0.35
        notes.append("frequent pauses detected")
    if 0.40 < silence_ratio < 0.70:
        uncertain_score += 0.25
    if energy_variation > 0.6 and zcr_std < 0.03:
        uncertain_score += 0.20
    if speech_rate < 2.5 and energy_rms > 0.02:
        uncertain_score += 0.20
    scores["uncertain"] = min(uncertain_score, 1.0)

    # ── ENGAGED signals ───────────────────────────────────────────────────
    # Sustained speech, low silence, varied pitch
    engaged_score = 0.0
    if speech_duration > 5.0 and silence_ratio < 0.40:
        engaged_score += 0.35
    if energy_rms > 0.06 and energy_variation < 0.7:
        engaged_score += 0.25
    if pitch_variation > 0.15:
        engaged_score += 0.20
    if speech_rate > 2.5 and speech_rate < 5.0:
        engaged_score += 0.20
    scores["engaged"] = min(engaged_score, 1.0)

    # ── NEUTRAL ───────────────────────────────────────────────────────────
    # Baseline when nothing else dominates
    total = sum(scores.values())
    if total < 0.5:
        scores["neutral"] += 0.4
        notes.append("limited signal detected — weak acoustic cues")
    scores["neutral"] = min(scores.get("neutral", 0.0), 1.0)

    # Normalize to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        scores = {k: 1.0 / len(scores) for k in scores}

    return ModalityResult(
        modality="acoustic",
        available=True,
        scores=scores,
        notes=notes,
        model_used="acoustic_rule_engine_v1",
    )
