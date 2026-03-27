"""
Genie 🧞‍♀️ — Summarizer
Generates human-readable text output from GenieResult.
"""

from __future__ import annotations

from .schemas import GenieResult

DIVIDER = "─" * 48


def format_text_report(result: GenieResult) -> str:
    """Full human-readable text report."""
    lines = [
        "",
        f"🧞‍♀️ {DIVIDER}",
        "   GENIE — STATE SKETCH",
        f"🧞‍♀️ {DIVIDER}",
        "",
        f"   Primary state:  {result.primary_state.upper()}",
        f"   Confidence:     {result.confidence.replace('_', ' ')}",
        "",
        "   Breakdown:",
    ]

    if result.scores:
        sorted_scores = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_scores:
            if prob >= 0.005:
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"   [{bar}] {label:12s} {prob*100:5.1f}%")
    lines.append("")

    if result.signals:
        active = [k.replace("_used", "").replace("_", " ")
                  for k, v in result.signals.items() if v]
        if active:
            lines.append(f"   Signals used:  {', '.join(active)}")
            lines.append("")

    if result.notes:
        lines.append("   Why:")
        for note in result.notes:
            lines.append(f"   · {note}")
        lines.append("")

    if result.segments:
        lines.append(f"   Meeting segments: {len(result.segments)}")
        for seg in result.segments[:8]:
            lines.append(
                f"   [{seg.start_sec:.0f}s–{seg.end_sec:.0f}s] "
                f"{seg.speaker_id}: {seg.primary_state} ({seg.confidence})"
            )
        lines.append("")

    if result.warnings:
        lines.append("   ⚠️  Notes:")
        for w in result.warnings:
            lines.append(f"   · {w}")
        lines.append("")

    lines += [
        f"   {DIVIDER}",
        f"   {result.disclaimer}",
        "",
        "   🛠️   Genie v1.0.0  ·  MIT  ·  github.com/aegiswizard/genie",
        "",
        f"🧞‍♀️ {DIVIDER}",
        "",
    ]

    return "\n".join(lines)


def format_whatsapp_reply(result: GenieResult) -> str:
    """Compact WhatsApp-friendly output."""
    top3 = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)[:3]
    breakdown = "\n".join(
        f"- {label}: {prob*100:.0f}%" for label, prob in top3
    )
    notes_str = "\n".join(f"· {n}" for n in result.notes[:2]) if result.notes else ""

    return (
        f"🧞‍♀️ Genie room read\n\n"
        f"Primary state: {result.primary_state}\n"
        f"Confidence: {result.confidence.replace('_', ' ')}\n\n"
        f"Breakdown:\n{breakdown}\n\n"
        + (f"Quick take:\n{notes_str}\n\n" if notes_str else "")
        + f"Note: {result.disclaimer}"
    )


def format_agent_summary(result: GenieResult) -> str:
    """One-line summary for agent embedding."""
    top_scores = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)
    top_str = ", ".join(
        f"{label}: {prob*100:.0f}%"
        for label, prob in top_scores[:3]
        if prob > 0.05
    )
    return (
        f"Genie state sketch: primary={result.primary_state}, "
        f"confidence={result.confidence}, scores=[{top_str}]. "
        f"{'; '.join(result.notes[:2]) if result.notes else 'No dominant signals.'}"
    )
