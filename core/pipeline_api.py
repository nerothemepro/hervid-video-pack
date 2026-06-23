#!/usr/bin/env python3
"""HerVid Pipeline API (v1).

A thin FastAPI wrapper around generate_ltx_video_sequence.py. Its whole reason to
exist is to make the *mechanical* choices deterministic so a caller (an LLM, n8n,
or curl) can only supply creative content — never pick the wrong tool, style, or
keyframe engine. style=realistic, keyframe_engine=auto(->flux) and
continuity=independent are FORCED here; that single rule is what kills the
"animation prompt rendered as anime samurai" class of bug.

The single RTX 3090 can run exactly one render at a time, so jobs are processed
by one background worker pulling from a queue. State is in-process, so run uvicorn
with a SINGLE worker (the default below).

Config (env vars, defaults match the current install):
  HVP_SCRIPT_PATH   path to generate_ltx_video_sequence.py
  HVP_ENV_FILE      media-pipeline.env passed through to the script
  HVP_OUTPUT_DIR    where final mp4s are written
  HVP_COMFY_URL     ComfyUI base url (used only by /health)
  HVP_PYTHON_BIN    python used to run the script (must have its deps)
  HVP_HOST / HVP_PORT
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
SCRIPT_PATH = os.environ.get(
    "HVP_SCRIPT_PATH",
    "/workspace/projects/media-pipeline/generate_ltx_video_sequence.py",
)
KEYFRAME_SCRIPT = os.environ.get(
    "HVP_KEYFRAME_SCRIPT",
    "/workspace/projects/media-pipeline/generate_video.py",
)
ENV_FILE = os.environ.get("HVP_ENV_FILE", "/opt/data/hermes/media-pipeline.env")
OUTPUT_DIR = os.environ.get("HVP_OUTPUT_DIR", "/opt/data/hermes/generated-videos")
COMFY_URL = os.environ.get("HVP_COMFY_URL", "http://host.docker.internal:8188")
PYTHON_BIN = os.environ.get("HVP_PYTHON_BIN", "python3")
HOST = os.environ.get("HVP_HOST", "0.0.0.0")
PORT = int(os.environ.get("HVP_PORT", "8500"))

# A whole sequence (up to ~20 shots) can take well over an hour on a 3090.
SUBPROCESS_TIMEOUT = int(os.environ.get("HVP_SUBPROCESS_TIMEOUT", "10800"))  # 3h
STDERR_TAIL_LINES = 60


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class SequenceRequest(BaseModel):
    """Creative input only. Mechanical params are forced server-side."""

    prompt: str = Field(..., min_length=3, description="Overall video brief / story")
    mode: str = Field("quality", pattern="^(test|standard|quality)$")
    total_duration_seconds: int = Field(30, ge=6, le=90)
    shot_duration_seconds: int = Field(3, ge=1, le=5)
    animation: str = Field("auto", pattern="^(auto|on|off)$")
    character_note: str = Field("", description="Shared character anchor for every shot")
    seed: Optional[int] = Field(None, ge=0, le=2147483647)
    # Optional creative-but-safe overrides
    steps: Optional[int] = Field(None, ge=1, le=30)
    skip_face_restore: bool = Field(False, description="Disable CodeFormer face restore (set for animal content)")
    keyframe_only: bool = Field(False, description="Generate just the Flux keyframe image; skip LTX I2V render")


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    queue_position: Optional[int] = None
    final_video_path: Optional[str] = None
    image_path: Optional[str] = None
    manifest_path: Optional[str] = None
    runtime_seconds: Optional[float] = None
    errors: list[str] = []
    warnings: list[str] = []
    request: Optional[dict[str, Any]] = None
    stderr_tail: list[str] = []


# --------------------------------------------------------------------------- #
# Job store + single worker                                                    #
# --------------------------------------------------------------------------- #
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_queue: "Queue[str]" = Queue()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _build_command(req: SequenceRequest) -> list[str]:
    """Build the CLI. Mechanical params are FORCED, not taken from the caller."""
    cmd = [
        PYTHON_BIN,
        SCRIPT_PATH,
        "--prompt", req.prompt,
        "--mode", req.mode,
        "--total-duration-seconds", str(req.total_duration_seconds),
        "--shot-duration-seconds", str(req.shot_duration_seconds),
        "--animation", req.animation,
        "--output-dir", OUTPUT_DIR,
        # --- FORCED mechanical params (caller cannot override) ---
        "--style", "realistic",          # flux keyframe engine; never animagine
        "--continuity", "independent",   # fresh keyframe/shot, no last-frame drift
    ]
    if req.character_note:
        cmd += ["--character-note", req.character_note]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if req.steps is not None:
        cmd += ["--steps", str(req.steps)]
    if Path(ENV_FILE).exists():
        cmd += ["--env-file", ENV_FILE]
    return cmd


def _build_keyframe_command(req: SequenceRequest) -> list[str]:
    """Build a generate_video.py --keyframe-only command for preview generation."""
    # Resolution and steps vary by mode: test is fast/low-res, quality is full-res.
    if req.mode == "test":
        kw, kh, flux_steps = 768, 512, 8
    else:
        kw, kh, flux_steps = 1152, 768, 28
    full_prompt = req.prompt
    if req.character_note:
        full_prompt = f"{full_prompt}, {req.character_note}"
    cmd = [
        PYTHON_BIN,
        KEYFRAME_SCRIPT,
        "--prompt", full_prompt,
        "--keyframe-only",
        "--keyframe-engine", "flux",
        "--keyframe-frame-mode", "single_scene",
        "--keyframe-width", str(kw),
        "--keyframe-height", str(kh),
        "--flux-steps", str(flux_steps),
    ]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if Path(ENV_FILE).exists():
        cmd += ["--env-file", ENV_FILE]
    return cmd


def _run_job(job_id: str) -> None:
    req = SequenceRequest(**_jobs[job_id]["request"])
    is_keyframe = req.keyframe_only
    cmd = _build_keyframe_command(req) if is_keyframe else _build_command(req)
    _set(job_id, status=JobStatus.running.value, started_at=_now(), queue_position=None)
    started = time.time()
    tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
    # Build subprocess env — override LTX_FACE_RESTORE for animal content
    proc_env = None
    if req.skip_face_restore:
        import os as _os
        proc_env = {**_os.environ, "LTX_FACE_RESTORE": "0"}
    script_dir = str(Path(KEYFRAME_SCRIPT if is_keyframe else SCRIPT_PATH).parent)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir,
            env=proc_env,
        )
        # Drain stderr live so we keep a useful tail for debugging.
        def _drain() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                tail.append(line.rstrip())
        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        try:
            stdout, _ = proc.communicate(timeout=SUBPROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            _set(
                job_id,
                status=JobStatus.failed.value,
                finished_at=_now(),
                runtime_seconds=round(time.time() - started, 1),
                errors=[f"render timed out after {SUBPROCESS_TIMEOUT}s"],
                stderr_tail=list(tail),
            )
            return
        t.join(timeout=2)

        payload: dict[str, Any] = {}
        if stdout and stdout.strip():
            try:
                payload = json.loads(stdout.strip())
            except json.JSONDecodeError:
                payload = {"status": "error", "errors": ["could not parse pipeline JSON output"]}

        ok = proc.returncode == 0 and payload.get("status") == "completed"
        _set(
            job_id,
            status=JobStatus.completed.value if ok else JobStatus.failed.value,
            finished_at=_now(),
            runtime_seconds=payload.get("runtime_seconds") or round(time.time() - started, 1),
            final_video_path=payload.get("final_video_path"),
            image_path=payload.get("image_path") or payload.get("start_keyframe_path"),
            manifest_path=payload.get("manifest_path"),
            errors=payload.get("errors") or ([] if ok else [f"exit code {proc.returncode}"]),
            warnings=payload.get("warnings") or [],
            stderr_tail=list(tail),
        )
    except Exception as exc:  # noqa: BLE001 - surface any launch error to the caller
        _set(
            job_id,
            status=JobStatus.failed.value,
            finished_at=_now(),
            runtime_seconds=round(time.time() - started, 1),
            errors=[f"{type(exc).__name__}: {exc}"],
            stderr_tail=list(tail),
        )


def _worker() -> None:
    while True:
        job_id = _queue.get()
        try:
            if _jobs.get(job_id, {}).get("status") == JobStatus.queued.value:
                _run_job(job_id)
        finally:
            _queue.task_done()
            _recompute_queue_positions()


def _recompute_queue_positions() -> None:
    with _jobs_lock:
        pending = [j for j in _jobs.values() if j["status"] == JobStatus.queued.value]
        pending.sort(key=lambda j: j["created_at"])
        for pos, j in enumerate(pending, start=1):
            j["queue_position"] = pos


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(title="HerVid Pipeline API", version="1.0.0")
_worker_thread = threading.Thread(target=_worker, daemon=True)


@app.on_event("startup")
def _startup() -> None:
    if not _worker_thread.is_alive():
        _worker_thread.start()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hervid-pipeline-api",
        "version": "1.0.0",
        "script": SCRIPT_PATH,
        "output_dir": OUTPUT_DIR,
        "endpoints": ["/health", "/generate-sequence", "/job/{id}", "/jobs"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    script_ok = Path(SCRIPT_PATH).exists()
    comfy_ok = False
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=8) as r:
            comfy_ok = r.status == 200
    except Exception:
        comfy_ok = False
    running = any(j["status"] == JobStatus.running.value for j in _jobs.values())
    return {
        "ok": script_ok and comfy_ok,
        "script_present": script_ok,
        "comfyui_reachable": comfy_ok,
        "busy": running,
        "queued": _queue.qsize(),
    }


@app.post("/generate-sequence", response_model=JobInfo)
def generate_sequence(req: SequenceRequest, validate_only: bool = False) -> JobInfo:
    if not Path(SCRIPT_PATH).exists():
        raise HTTPException(500, f"pipeline script not found: {SCRIPT_PATH}")

    if validate_only:
        # Fast smoke test: run the script's own --validate-only synchronously.
        cmd = _build_command(req) + ["--validate-only"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              cwd=str(Path(SCRIPT_PATH).parent))
        try:
            payload = json.loads(proc.stdout.strip()) if proc.stdout.strip() else {}
        except json.JSONDecodeError:
            payload = {"status": "error", "errors": ["validate-only produced no JSON"]}
        ok = proc.returncode == 0
        return JobInfo(
            id="validate-only",
            status=JobStatus.completed if ok else JobStatus.failed,
            created_at=_now(),
            finished_at=_now(),
            errors=payload.get("errors") or ([] if ok else [proc.stderr[-400:]]),
            request=req.model_dump(),
        )

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.queued.value,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "queue_position": None,
            "final_video_path": None,
            "image_path": None,
            "manifest_path": None,
            "runtime_seconds": None,
            "errors": [],
            "warnings": [],
            "request": req.model_dump(),
            "stderr_tail": [],
        }
    _queue.put(job_id)
    _recompute_queue_positions()
    return JobInfo(**_jobs[job_id])


@app.get("/job/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> JobInfo:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"job not found: {job_id}")
    return JobInfo(**job)


@app.get("/jobs", response_model=list[JobInfo])
def list_jobs() -> list[JobInfo]:
    with _jobs_lock:
        return [JobInfo(**j) for j in sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, workers=1)
