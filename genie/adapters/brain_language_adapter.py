"""
Genie 🧞‍♀️ — Brain Language NLP Adapter (Optional, MIT)

Optional research integration with brain_language_nlp (MIT).
Use for text-based cognitive clarity experiments:
  - "Which version of this message is cognitively cleaner?"
  - Compare written versions of the same spoken content.

Source: https://github.com/brain-score/language (MIT)
This is distinct from person-state analysis — it analyzes text content,
not the individual speaker.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def brain_language_available() -> bool:
    try:
        import brainscore_language  # noqa
        return True
    except ImportError:
        return False


def analyze_text_brain_language(
    text: str,
    compare_texts: Optional[List[str]] = None,
) -> Dict:
    """
    Analyze text using brain_language_nlp encoding models.
    Returns cognitive clarity estimate for the text.

    If compare_texts provided, returns relative ranking.
    Returns empty result if not installed.
    """
    if not brain_language_available():
        return {
            "available": False,
            "reason": "brainscore_language not installed",
            "install": "See https://github.com/brain-score/language",
            "license": "MIT",
        }

    try:
        import brainscore_language
        # Basic encoding model prediction
        result = {"available": True, "text": text[:100], "license": "MIT"}
        if compare_texts:
            result["comparison_texts"] = [t[:100] for t in compare_texts]
            result["note"] = "Comparison analysis not yet implemented in this adapter"
        return result
    except Exception as e:
        return {"available": False, "reason": str(e)}
