"""
Genie 🧞‍♀️ — TRIBE v2 Research Adapter (Optional)

⚠️  RESEARCH MODULE — CC BY-NC 4.0 LICENSE
    TRIBE v2 is NOT MIT licensed. It is CC BY-NC 4.0 — noncommercial use only.
    Do not use in commercial products without separate licensing.

TRIBE predicts fMRI brain responses to naturalistic stimuli
(video, audio, text) for an average subject on the fsaverage5 cortical mesh.

This is content-level brain response prediction — NOT person-state analysis.

Source: https://github.com/cfosco/TRIBE (CC BY-NC 4.0)
Install: see TRIBE repo for requirements

Separation from Genie base:
  Genie base: "What does this person seem like?" (MIT)
  TRIBE addon: "What might this content do to an average brain?" (CC BY-NC 4.0 — research)
"""

from __future__ import annotations

from typing import Dict, Optional


def tribe_available() -> bool:
    try:
        import tribe  # noqa
        return True
    except ImportError:
        return False


def analyze_content_tribe(
    media_path: str,
    modality: str = "audio",  # "audio" | "video" | "text"
) -> Dict:
    """
    Run TRIBE v2 content-level brain response prediction.
    Returns empty dict if TRIBE not installed.

    ⚠️  CC BY-NC 4.0 — noncommercial research use only.
    """
    if not tribe_available():
        return {
            "available": False,
            "reason": "TRIBE not installed — see https://github.com/cfosco/TRIBE",
            "license_note": "CC BY-NC 4.0 — noncommercial research use only",
        }

    try:
        import tribe
        # TRIBE-specific API (adapt as per current TRIBE version)
        predictor = tribe.Predictor()
        result = predictor.predict(media_path, modality=modality)
        return {
            "available": True,
            "modality":  modality,
            "prediction": result,
            "subject":   "average (fsaverage5)",
            "license_note": "CC BY-NC 4.0 — noncommercial research use only",
            "disclaimer": (
                "This predicts average fMRI brain response to the content, "
                "NOT the private mental state of the person in the clip."
            ),
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"TRIBE error: {e}",
            "license_note": "CC BY-NC 4.0 — noncommercial research use only",
        }
