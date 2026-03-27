"""
Genie 🧞‍♀️ — Visual Output: State Bars
Generates a simple horizontal bar chart as ASCII or PNG.
PNG requires matplotlib (optional).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from ..schemas import GenieResult


STATE_COLORS = {
    "calm":       "#4CAF50",
    "tense":      "#F44336",
    "upbeat":     "#FF9800",
    "low_energy": "#9E9E9E",
    "uncertain":  "#9C27B0",
    "engaged":    "#2196F3",
    "neutral":    "#607D8B",
}


def render_bars_ascii(scores: Dict[str, float], width: int = 30) -> str:
    """Always-available ASCII bar chart."""
    lines = []
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_scores:
        if prob < 0.005:
            continue
        bar_len = int(prob * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"  {label:12s} [{bar}] {prob*100:5.1f}%")
    return "\n".join(lines)


def render_bars_png(
    result: GenieResult,
    output_path: str,
) -> bool:
    """
    Render a PNG bar chart. Returns True on success.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    scores = result.scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [s[0] for s in sorted_scores if s[1] > 0.005]
    values = [s[1] for s in sorted_scores if s[1] > 0.005]
    colors = [STATE_COLORS.get(l, "#607D8B") for l in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title(
        f"🧞‍♀️ Genie State Sketch\n"
        f"Primary: {result.primary_state.upper()}  |  Confidence: {result.confidence}",
        fontsize=12, pad=12,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", fontsize=10,
        )

    ax.text(
        0.5, -0.12,
        result.disclaimer,
        transform=ax.transAxes, ha="center", fontsize=7,
        color="gray", style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return True
