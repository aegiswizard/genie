"""
Genie 🧞‍♀️ — Configuration
Single source of truth for all runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


@dataclass
class FusionWeights:
    """How much each modality contributes when fusing signals."""
    # Video mode weights
    voice_video:    float = 0.50
    face_video:     float = 0.35
    acoustic_video: float = 0.15

    # Audio-only mode weights
    voice_audio:    float = 0.80
    acoustic_audio: float = 0.20

    # When face quality is poor, shift weight
    face_quality_threshold: float = 0.40  # below this: reduce face weight


@dataclass
class QualityThresholds:
    """Thresholds for confidence scoring."""
    min_speech_fraction:    float = 0.20   # less than 20% speech → low confidence
    min_audio_clarity:      float = 0.30
    min_face_visibility:    float = 0.40
    min_margin_for_medium:  float = 0.08   # top vs second label must differ by this
    min_margin_for_high:    float = 0.18
    min_usable_frames:      int   = 5


@dataclass
class AudioConfig:
    sample_rate:      int   = 16000
    mono:             bool  = True
    trim_silence:     bool  = True
    silence_threshold_db: float = -40.0
    chunk_duration_s: float = 5.0       # chunk size for long clips
    max_duration_s:   float = 300.0     # hard cap at 5 minutes


@dataclass
class VideoConfig:
    max_frames:       int   = 60         # max frames to extract
    frame_interval_s: float = 0.5        # sample one frame per N seconds
    min_face_size:    int   = 48         # pixels — smaller faces are low quality


@dataclass
class GenieConfig:
    """Master configuration. Can be set from CLI flags, YAML, or code."""

    # Mode
    mode: Literal["auto", "audio", "video", "meeting"] = "auto"

    # Which adapters to attempt
    use_speech_emotion:  bool = True    # MIT speech emotion model
    use_deepface:        bool = True    # MIT face analysis
    use_pyfeat:          bool = False   # MIT face (richer but heavier)
    use_pyannote:        bool = False   # MIT diarization (requires HF token)
    use_acoustic_only:   bool = True    # always-on fallback

    # API tokens for optional models
    huggingface_token:   Optional[str] = None   # needed for pyannote

    # Output
    output_format: Literal["text", "json", "both"] = "both"
    include_visuals:     bool = False
    visuals_output_dir:  Optional[str] = None

    # Processing
    audio:   AudioConfig   = field(default_factory=AudioConfig)
    video:   VideoConfig   = field(default_factory=VideoConfig)
    fusion:  FusionWeights = field(default_factory=FusionWeights)
    quality: QualityThresholds = field(default_factory=QualityThresholds)

    # Temp file management
    keep_temp_files:     bool = False
    temp_dir:            Optional[str] = None


# Default singleton
DEFAULT_CONFIG = GenieConfig()
