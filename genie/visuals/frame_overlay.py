"""Genie 🧞‍♀️ — Frame Overlay Visual"""
from __future__ import annotations
from typing import List, Optional


def render_frame_overlay(frame_path: str, label: str, confidence: str, output_path: str) -> bool:
    """Overlay state label on a video frame. Requires PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        img  = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        text = f"{label.upper()}  ({confidence})"
        draw.rectangle([0, h - 40, w, h], fill=(0, 0, 0, 160))
        draw.text((10, h - 30), text, fill=(255, 255, 255))
        img.save(output_path)
        return True
    except Exception:
        return False
