"""Genie 🧞‍♀️ — Timeline Visual"""
from __future__ import annotations
from typing import List
from ..schemas import Segment, GenieResult


def render_timeline_ascii(segments: List[Segment], total_duration: float) -> str:
    if not segments:
        return "  No segments for timeline."
    lines = ["  TIMELINE", "  " + "─" * 50]
    for seg in segments:
        pct_start = int(seg.start_sec / max(total_duration, 1) * 40)
        pct_end   = int(seg.end_sec   / max(total_duration, 1) * 40)
        bar = " " * pct_start + "█" * max(1, pct_end - pct_start)
        lines.append(
            f"  {seg.speaker_id:8s} [{bar:<40s}] "
            f"{seg.start_sec:.0f}s–{seg.end_sec:.0f}s {seg.primary_state}"
        )
    return "\n".join(lines)


def render_timeline_png(result: GenieResult, output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    from .bars import STATE_COLORS
    segments = result.segments
    if not segments:
        return False

    fig, ax = plt.subplots(figsize=(10, max(2, len({s.speaker_id for s in segments}) * 0.8)))
    speakers = sorted(set(s.speaker_id for s in segments))
    y_map    = {sp: i for i, sp in enumerate(speakers)}

    for seg in segments:
        y   = y_map[seg.speaker_id]
        col = STATE_COLORS.get(seg.primary_state, "#607D8B")
        ax.barh(y, seg.end_sec - seg.start_sec, left=seg.start_sec,
                color=col, edgecolor="white", height=0.6)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title("🧞‍♀️ Genie — Meeting Timeline")
    ax.legend(
        handles=[mpatches.Patch(color=c, label=l) for l, c in STATE_COLORS.items()],
        loc="upper right", fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return True
