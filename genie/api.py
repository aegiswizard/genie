"""
Genie 🧞‍♀️ — REST API
FastAPI server. POST a file path or upload a file, get a GenieResult JSON back.

Endpoints:
  POST /analyze          — analyze uploaded file
  POST /analyze/path     — analyze file by local path
  GET  /doctor           — capability check
  GET  /health           — health check
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


def create_app():
    try:
        from fastapi import FastAPI, File, Form, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    from genie.pipeline import run
    from genie.config   import GenieConfig

    app = FastAPI(
        title="Genie 🧞‍♀️ — State Sketch Engine",
        description=(
            "Best-effort human state sketch from voice and video clips. "
            "MIT License — github.com/aegiswizard/genie"
        ),
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "genie", "version": "1.0.0"}

    @app.get("/doctor")
    def doctor():
        caps = {}
        import shutil
        caps["ffmpeg"]        = bool(shutil.which("ffmpeg"))
        for pkg in ["librosa","deepface","feat","pyannote","transformers","matplotlib"]:
            try:
                __import__(pkg)
                caps[pkg] = True
            except ImportError:
                caps[pkg] = False
        return {"capabilities": caps, "disclaimer": "Core acoustic analysis always available."}

    @app.post("/analyze")
    async def analyze_upload(
        file: UploadFile = File(...),
        mode: str        = Form(default="auto"),
        output: str      = Form(default="agent_json"),
    ):
        """Upload a file and get a state sketch."""
        suffix = os.path.splitext(file.filename or "upload")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            cfg    = GenieConfig(mode=mode)
            result = run(tmp_path, config=cfg, mode=mode)
            return JSONResponse(result.to_dict())
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @app.post("/analyze/path")
    def analyze_path(
        path: str = Form(...),
        mode: str = Form(default="auto"),
    ):
        """Analyze a file by local path (for local agent use)."""
        cfg    = GenieConfig(mode=mode)
        result = run(path, config=cfg, mode=mode)
        return JSONResponse(result.to_dict())

    return app
