"""Genie 🧞‍♀️ — Utilities: logging, io, validation"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path


def get_logger(name: str = "genie") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s [genie] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def make_temp_dir(base: str = None) -> str:
    import tempfile
    d = tempfile.mkdtemp(prefix="genie_", dir=base)
    return d


def cleanup_temp_dir(path: str) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_ALL   = SUPPORTED_AUDIO | SUPPORTED_VIDEO


def validate_input_file(path: str) -> tuple:
    """
    Validate input file. Returns (is_valid, input_type, error_msg).
    input_type: "audio" | "video" | None
    """
    p = Path(path)
    if not p.exists():
        return False, None, f"File not found: {path}"
    if not p.is_file():
        return False, None, f"Not a file: {path}"
    ext = p.suffix.lower()
    if ext in SUPPORTED_VIDEO:
        return True, "video", None
    if ext in SUPPORTED_AUDIO:
        return True, "audio", None
    return False, None, (
        f"Unsupported format '{ext}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_ALL))}"
    )


def resolve_output_path(input_path: str, suffix: str, output_dir: str = None) -> str:
    p    = Path(input_path)
    name = p.stem + suffix
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return str(Path(output_dir) / name)
    return str(p.parent / name)
