"""
Genie 🧞‍♀️ — Video Preprocessing
Extract audio track and key frames from video files.
Requires ffmpeg for video support.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_from_video(
    video_path: str,
    output_wav: str,
    sample_rate: int = 16000,
) -> Tuple[bool, List[str]]:
    """
    Extract mono audio track from a video file.
    Returns (success, warnings).
    """
    warnings = []
    if not _ffmpeg_available():
        warnings.append("ffmpeg not available — cannot extract audio from video")
        return False, warnings
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",                    # no video
            "-ac", "1",               # mono
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            output_wav,
        ], capture_output=True, check=True)
        return True, warnings
    except subprocess.CalledProcessError as e:
        warnings.append(f"ffmpeg audio extraction failed: {e}")
        return False, warnings


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval_s: float = 0.5,
    max_frames: int = 60,
) -> Tuple[List[str], List[str]]:
    """
    Extract frames at regular intervals from a video.
    Returns (list_of_frame_paths, warnings).
    """
    warnings = []
    frame_paths = []

    if not _ffmpeg_available():
        warnings.append("ffmpeg not available — cannot extract frames from video")
        return frame_paths, warnings

    os.makedirs(output_dir, exist_ok=True)

    # Calculate fps for extraction
    fps = 1.0 / frame_interval_s
    duration = get_video_duration(video_path)
    if duration:
        n_frames = min(int(duration / frame_interval_s), max_frames)
    else:
        n_frames = max_frames

    if n_frames == 0:
        warnings.append("Video too short for frame extraction")
        return frame_paths, warnings

    try:
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps:.3f},scale=-1:360",   # 360p height, keep aspect
            "-frames:v", str(n_frames),
            "-q:v", "3",                              # quality
            output_pattern,
        ], capture_output=True, check=True)

        frame_paths = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])

    except subprocess.CalledProcessError as e:
        warnings.append(f"Frame extraction failed: {e}")

    return frame_paths, warnings
