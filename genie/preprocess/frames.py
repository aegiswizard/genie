"""
Genie 🧞‍♀️ — Frame Quality
Assess frame quality and face visibility for confidence scoring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _try_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _try_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def load_frame(path: str) -> Tuple[np.ndarray, bool]:
    """Load a frame as numpy array. Returns (array, success)."""
    cv2 = _try_cv2()
    if cv2 is not None:
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), True

    pil = _try_pil()
    if pil is not None:
        try:
            img = pil.open(path).convert("RGB")
            return np.array(img), True
        except Exception:
            pass

    return np.array([]), False


def assess_frame_quality(frame: np.ndarray) -> float:
    """
    Estimate frame quality 0.0–1.0 using Laplacian variance (blur detection).
    Higher = sharper = better quality.
    """
    if frame.size == 0:
        return 0.0

    cv2 = _try_cv2()
    if cv2 is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize: <100 = very blurry, >1000 = sharp
            return float(min(laplacian_var / 1000.0, 1.0))
        except Exception:
            pass

    # Fallback: std of pixel values as roughness proxy
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    return float(min(np.std(gray) / 128.0, 1.0))


def select_best_frames(frame_paths: List[str], n: int = 10) -> List[str]:
    """
    Select the N sharpest/highest quality frames from a list.
    """
    if not frame_paths:
        return []
    if len(frame_paths) <= n:
        return frame_paths

    scored = []
    for path in frame_paths:
        frame, ok = load_frame(path)
        quality = assess_frame_quality(frame) if ok else 0.0
        scored.append((quality, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored[:n]]
