"""
Genie 🧞‍♀️ — CLI
genie analyze <file> [options]
"""

from __future__ import annotations

import argparse
import os
import sys

BANNER = """
  🧞‍♀️  Genie — State Sketch Engine  v1.0.0
      Reads the room, not your soul.
      github.com/aegiswizard/genie  ·  MIT License
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="genie",
        description="🧞‍♀️ Genie — Best-effort human state sketch from voice and video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  genie analyze voice_note.m4a
  genie analyze selfie.mp4
  genie analyze meeting.mp4 --mode meeting
  genie analyze voice.wav --output json
  genie analyze clip.mp4 --visuals --visuals-dir ./output
  genie analyze voice.m4a --no-deepface
  genie doctor

supported formats:
  audio:  .wav  .mp3  .m4a  .ogg  .flac  .aac
  video:  .mp4  .mov  .avi  .mkv  .webm

notes:
  · ffmpeg must be installed for non-WAV audio and all video formats
  · DeepFace (pip install deepface) enables face analysis in video mode
  · pyannote.audio (pip install pyannote.audio) enables meeting diarization
  · All analysis is local. No data leaves your machine.
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    # ── analyze ─────────────────────────────────────────────────────────────
    a = sub.add_parser("analyze", help="Analyze a voice note or video clip")
    a.add_argument("file",       help="Path to audio or video file")
    a.add_argument("--mode", "-m",
                   choices=["auto","audio","video","meeting"], default="auto",
                   help="Analysis mode (default: auto-detect)")
    a.add_argument("--output", "-o",
                   choices=["text","json","both","whatsapp"], default="both",
                   help="Output format (default: both)")
    a.add_argument("--visuals",     action="store_true", help="Generate PNG bar chart")
    a.add_argument("--visuals-dir", default=None,        help="Directory for visual outputs")
    a.add_argument("--no-deepface", action="store_true", help="Disable DeepFace face analysis")
    a.add_argument("--use-pyfeat",  action="store_true", help="Use Py-FEAT instead of DeepFace")
    a.add_argument("--hf-token",    default=os.environ.get("GENIE_HF_TOKEN"),
                   help="HuggingFace token for pyannote (or set GENIE_HF_TOKEN env var)")
    a.add_argument("--keep-temp",   action="store_true", help="Keep temp files for debugging")

    # ── doctor ───────────────────────────────────────────────────────────────
    sub.add_parser("doctor", help="Check which Genie capabilities are available on this machine")

    # ── serve ────────────────────────────────────────────────────────────────
    sv = sub.add_parser("serve", help="Start Genie REST API server")
    sv.add_argument("--port", "-p", type=int, default=7070)
    sv.add_argument("--host",       default="0.0.0.0")

    # ── version ──────────────────────────────────────────────────────────────
    sub.add_parser("version", help="Print version")

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        print("genie 1.0.0")
        sys.exit(0)

    # ── doctor ───────────────────────────────────────────────────────────────
    if args.command == "doctor":
        print(BANNER)
        _run_doctor()
        sys.exit(0)

    # ── serve ─────────────────────────────────────────────────────────────────
    if args.command == "serve":
        print(BANNER)
        from genie.api import create_app
        try:
            import uvicorn
        except ImportError:
            print("❌  uvicorn not installed: pip install uvicorn", file=sys.stderr)
            sys.exit(1)
        app = create_app()
        print(f"  🧞‍♀️ Genie API → http://{args.host}:{args.port}/docs\n")
        uvicorn.run(app, host=args.host, port=args.port)
        sys.exit(0)

    # ── analyze ───────────────────────────────────────────────────────────────
    if args.command == "analyze":
        print(BANNER, file=sys.stderr)

        from genie.config import GenieConfig
        from genie.pipeline import run
        from genie.summarizer import format_text_report, format_whatsapp_reply
        from genie.visuals.bars import render_bars_png, render_bars_ascii

        cfg = GenieConfig(
            mode=args.mode,
            use_deepface=not args.no_deepface,
            use_pyfeat=args.use_pyfeat,
            huggingface_token=args.hf_token,
            include_visuals=args.visuals,
            visuals_output_dir=args.visuals_dir,
            keep_temp_files=args.keep_temp,
        )

        print(f"  ⏳ Analyzing: {args.file}\n", file=sys.stderr)
        result = run(args.file, config=cfg, mode=args.mode)

        fmt = args.output

        if fmt in ("text", "both"):
            print(format_text_report(result))

        if fmt == "whatsapp":
            print(format_whatsapp_reply(result))

        if fmt in ("json", "both"):
            print(result.to_agent_json())

        if args.visuals:
            from genie.utils import resolve_output_path
            out_path = resolve_output_path(args.file, "_genie.png", args.visuals_dir)
            ok = render_bars_png(result, out_path)
            if ok:
                print(f"\n  📊 Chart saved: {out_path}", file=sys.stderr)
            else:
                print(render_bars_ascii(result.scores))

        if result.errors:
            print(f"\n  ❌ Errors: {'; '.join(result.errors)}", file=sys.stderr)
            sys.exit(1)


def _run_doctor() -> None:
    """Print capability report for this machine."""
    lines = ["  Capabilities on this machine:\n"]

    # ffmpeg
    import shutil
    lines.append(f"  {'✅' if shutil.which('ffmpeg') else '❌'}  ffmpeg — video/audio conversion")

    # numpy
    try:
        import numpy
        lines.append(f"  ✅  numpy {numpy.__version__} — acoustic analysis (always-on)")
    except ImportError:
        lines.append("  ❌  numpy — install: pip install numpy")

    # librosa
    try:
        import librosa
        lines.append(f"  ✅  librosa {librosa.__version__} — improved audio analysis")
    except ImportError:
        lines.append("  ⚠️   librosa not installed — pip install librosa  (recommended)")

    # DeepFace
    try:
        from deepface import DeepFace  # noqa
        lines.append("  ✅  DeepFace — face emotion analysis for video")
    except ImportError:
        lines.append("  ⚠️   DeepFace not installed — pip install deepface  (for video face analysis)")

    # PyFEAT
    try:
        from feat import Detector  # noqa
        lines.append("  ✅  Py-FEAT — rich face analysis (action units)")
    except ImportError:
        lines.append("  ⚠️   Py-FEAT not installed — pip install feat  (optional)")

    # transformers
    try:
        import transformers
        lines.append(f"  ✅  transformers {transformers.__version__} — speech emotion model")
    except ImportError:
        lines.append("  ⚠️   transformers not installed — pip install transformers  (speech emotion)")

    # pyannote
    try:
        import pyannote  # noqa
        lines.append("  ✅  pyannote.audio — meeting diarization")
    except ImportError:
        lines.append("  ⚠️   pyannote.audio not installed — pip install pyannote.audio  (meeting mode)")

    # matplotlib
    try:
        import matplotlib
        lines.append(f"  ✅  matplotlib {matplotlib.__version__} — PNG chart output")
    except ImportError:
        lines.append("  ⚠️   matplotlib not installed — pip install matplotlib  (for PNG visuals)")

    # FastAPI
    try:
        import fastapi
        lines.append(f"  ✅  FastAPI {fastapi.__version__} — REST API server")
    except ImportError:
        lines.append("  ⚠️   FastAPI not installed — pip install fastapi uvicorn  (for serve mode)")

    lines.append("")
    lines.append("  Core acoustic analysis (works NOW with no optional packages)")
    lines.append("  Add packages above to unlock more capabilities.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
