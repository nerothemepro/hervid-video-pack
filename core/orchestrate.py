#!/usr/bin/env python3
"""HerVid Orchestrator — deterministic 4-step video pipeline.

  Brief (any language)
    ↓ step 1 — LM Studio: brief → creative JSON
  {prompt, character_note, total_duration_seconds, animation}
    ↓ step 2 — Pipeline API /generate-sequence → job_id
    ↓ step 3 — poll /job/{id} → completed + final_video_path
    ↓ step 4 — Telegram sendVideo (if TELEGRAM_BOT_TOKEN + chat_id)
  ✅ done

Two modes:
  Service  — `python orchestrate.py` → FastAPI on port 8501
             POST /generate-video {"brief": "...", "chat_id": 123}
             GET  /pipeline-job/{id}
  CLI      — `python orchestrate.py "your brief"` → blocks until done

Config (env):
  LM_STUDIO_BASE_URL   http://host.docker.internal:1234/v1
  LM_MODEL             gemma-4-12b-qat
  HVP_API_URL          http://localhost:8500
  TELEGRAM_BOT_TOKEN   (optional) send result to Telegram
  HVP_POLL_INTERVAL    30  (seconds between polls)
  HVP_MAX_WAIT         7200 (hard timeout in seconds)
  HVO_HOST / HVO_PORT  0.0.0.0 / 8501
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from typing import Any, Callable, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
LM_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
LM_MODEL = os.environ.get("LM_MODEL", "google/gemma-4-12b-qat")
# LM Studio management API (not /v1 — separate port/path)
LM_MGMT = LM_URL.replace("/v1", "")  # http://host.docker.internal:1234
HVP_URL = os.environ.get("HVP_API_URL", "http://localhost:8500")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
POLL_INTERVAL = int(os.environ.get("HVP_POLL_INTERVAL", "30"))
MAX_WAIT = int(os.environ.get("HVP_MAX_WAIT", "7200"))
HOST = os.environ.get("HVO_HOST", "0.0.0.0")
PORT = int(os.environ.get("HVO_PORT", "8501"))

_LM_SYSTEM = """\
You are a video production assistant. Given a user brief in any language, extract creative elements and return ONLY a valid JSON object with exactly these fields:
- "prompt": vivid English description of what happens in the video (1-3 sentences, present tense, specific actions and setting). Focus on the scene, action, and environment first. Example for "fox running in forest": "A small red fox sprints through a vibrant autumn forest, leaping over fallen leaves, sunlight filtering through the canopy."
- "character_note": English description of the main character or subject appearance for visual consistency. Include species, colors, textures, and distinctive features. Example: "small red fox, pointed black ears, amber eyes, white muzzle, fluffy tail with white tip"
- "total_duration_seconds": integer 6-30 (default 10 for test/short requests, 20-30 for full videos)
- "animation": "on" if 3D-animated/cartoon/Pixar-style content, "off" if live-action/realistic/documentary, "auto" if uncertain

Rules: output ONLY the JSON object, no explanation, no markdown fences."""


# --------------------------------------------------------------------------- #
# LM Studio VRAM management (single 3090: LM + render cannot coexist)         #
# --------------------------------------------------------------------------- #
def _lm_load() -> None:
    """Ensure LM_MODEL is loaded in LM Studio. No-op if already loaded."""
    try:
        info = requests.get(f"{LM_MGMT}/api/v1/models", timeout=10).json()
        for m in info.get("models", []):
            # Match on key suffix to handle "google/gemma-4-12b-qat" vs "gemma-4-12b-qat"
            key = m.get("key", "")
            if (key == LM_MODEL or key.endswith("/" + LM_MODEL) or LM_MODEL.endswith("/" + key)) \
                    and m.get("loaded_instances"):
                return  # already loaded
        requests.post(f"{LM_MGMT}/api/v1/models/load", json={"model": LM_MODEL}, timeout=120).raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not load LM Studio model {LM_MODEL!r}: {e}") from e


def _lm_eject() -> None:
    """Unload all LM Studio models to free VRAM before render."""
    try:
        info = requests.get(f"{LM_MGMT}/api/v1/models", timeout=10).json()
        for m in info.get("models", []):
            for inst in m.get("loaded_instances", []):
                iid = inst.get("instance_id")
                if iid:
                    requests.post(f"{LM_MGMT}/api/v1/models/unload", json={"instance_id": iid}, timeout=30)
    except Exception:
        pass  # best-effort; render will proceed anyway


# --------------------------------------------------------------------------- #
# Step 1 — LM Studio: brief → creative JSON                                   #
# --------------------------------------------------------------------------- #
def step1_parse_brief(brief: str, retries: int = 3) -> dict[str, Any]:
    payload = {
        "model": LM_MODEL,
        "messages": [
            {"role": "system", "content": _LM_SYSTEM},
            {"role": "user", "content": brief},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,  # gemma-4-12b: ~800 for thinking phase + ~400 for JSON output
    }
    last_err: Exception | None = None
    text = ""
    # Backoff schedule: 15s, 30s, 60s, 90s — long enough for Hermes curator to release LM Studio
    _backoffs = [15, 30, 60, 90]
    for attempt in range(retries):
        try:
            resp = requests.post(f"{LM_URL}/chat/completions", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()

            # LM Studio returns error dict (e.g. model busy with another caller) instead of choices
            if "error" in data:
                raise ValueError(f"LM Studio returned error: {data['error']}")
            if not data.get("choices"):
                raise ValueError(f"LM Studio response missing 'choices' — raw: {str(data)[:200]}")

            msg = data["choices"][0]["message"]
            # gemma-4-12b-qat: answer in 'content', thinking in 'reasoning_content'
            text = (msg.get("content") or "").strip()
            if not text:
                # Fallback: reasoning_content sometimes carries the final output
                text = (msg.get("reasoning_content") or "").strip()
            if not text:
                raise ValueError("LM Studio returned empty content (token budget exhausted in thinking phase?)")

            # Strip markdown fences if model added them
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
                text = text.rsplit("```", 1)[0].strip()

            parsed = json.loads(text)
            if "prompt" not in parsed:
                raise ValueError("missing 'prompt' field")
            parsed.setdefault("character_note", "")
            parsed.setdefault("total_duration_seconds", 30)
            parsed.setdefault("animation", "auto")
            parsed["total_duration_seconds"] = max(6, min(60, int(parsed["total_duration_seconds"])))
            return parsed

        except (requests.RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            if attempt < retries - 1:
                wait = _backoffs[min(attempt, len(_backoffs) - 1)]
                print(f"[orchestrate] step1 attempt {attempt+1}/{retries} failed: {e} — retrying in {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"step1_parse_brief failed after {retries} attempts. "
        f"Last error: {last_err}. Last text: {text!r}"
    )


# --------------------------------------------------------------------------- #
# Step 2 — Pipeline API: submit job                                            #
# --------------------------------------------------------------------------- #
_FACE_DETAIL_SUFFIX = ", sharp detailed face, in-focus features"

# CodeFormer is trained on human faces and produces artifacts (color blotches, muzzle
# distortion) when applied to animal content. Skip it when character_note is animal.
_ANIMAL_KEYWORDS = {
    "fox", "bear", "wolf", "dog", "cat", "tiger", "lion", "elephant", "deer",
    "rabbit", "bird", "eagle", "owl", "horse", "panda", "raccoon", "squirrel",
}

def _is_animal_content(character_note: str) -> bool:
    note_lower = character_note.lower()
    return any(kw in note_lower for kw in _ANIMAL_KEYWORDS)


def step2_submit_job(creative: dict[str, Any], mode: str = "quality") -> str:
    prompt = creative["prompt"]
    character_note = creative.get("character_note", "")
    # Append face-detail tokens if not already present
    if "sharp" not in prompt.lower() and "detail" not in prompt.lower():
        prompt = prompt.rstrip(". ") + _FACE_DETAIL_SUFFIX
    body = {
        "prompt": prompt,
        "character_note": character_note,
        "total_duration_seconds": creative.get("total_duration_seconds", 30),
        "animation": creative.get("animation", "auto"),
        "mode": mode,
    }
    # CodeFormer creates muzzle/snout artifacts on animal faces (not trained for animals).
    # Signal pipeline_api to disable face restore for this job.
    if _is_animal_content(character_note):
        body["skip_face_restore"] = True
    resp = requests.post(f"{HVP_URL}/generate-sequence", json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


# --------------------------------------------------------------------------- #
# Step 3 — Poll until done                                                     #
# --------------------------------------------------------------------------- #
def step3_poll(job_id: str, on_status: Callable[[str], None] | None = None) -> dict[str, Any]:
    deadline = time.time() + MAX_WAIT
    last = None
    while time.time() < deadline:
        resp = requests.get(f"{HVP_URL}/job/{job_id}", timeout=15)
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]
        if status != last:
            if on_status:
                on_status(status)
            last = status
        if status in ("completed", "failed"):
            return job
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"job {job_id} did not finish within {MAX_WAIT}s")


# --------------------------------------------------------------------------- #
# Step 4 — Telegram send (optional)                                            #
# --------------------------------------------------------------------------- #
def step4_telegram(chat_id: str | int, video_path: str, caption: str = "") -> None:
    if not TG_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendVideo"
    with open(video_path, "rb") as f:
        resp = requests.post(
            url,
            data={"chat_id": str(chat_id), "caption": caption[:1024]},
            files={"video": (os.path.basename(video_path), f, "video/mp4")},
            timeout=120,
        )
    resp.raise_for_status()
    if not resp.json().get("ok"):
        raise RuntimeError(f"Telegram sendVideo failed: {resp.json()}")


# --------------------------------------------------------------------------- #
# Full pipeline                                                                #
# --------------------------------------------------------------------------- #
def run(
    brief: str,
    chat_id: str | int | None = None,
    on_progress: Callable[[str], None] | None = None,
    mode: str = "quality",
) -> dict[str, Any]:
    log = on_progress or (lambda m: print(f"[orchestrate] {m}", file=sys.stderr, flush=True))

    log("step 1/4 — loading LM Studio model...")
    _lm_load()
    log("  parsing brief...")
    creative = step1_parse_brief(brief)
    log(f"  prompt: {creative['prompt'][:100]}")
    log(f"  character: {creative['character_note'][:80]}")
    log(f"  duration: {creative['total_duration_seconds']}s  animation: {creative['animation']}")

    log("step 2/4 — ejecting LM Studio model to free VRAM for render...")
    _lm_eject()
    log("  submitting to pipeline API...")
    job_id = step2_submit_job(creative, mode=mode)
    log(f"  job_id: {job_id}")

    log("step 3/4 — rendering (15-60 min on RTX 3090)...")
    job = step3_poll(job_id, on_status=lambda s: log(f"  render status → {s}"))

    if job["status"] != "completed":
        errors = job.get("errors") or job.get("stderr_tail", [])[-3:]
        raise RuntimeError(f"render failed — errors: {errors}")

    video_path = job["final_video_path"]
    log(f"  done: {video_path}  ({job.get('runtime_seconds', '?')}s)")

    if chat_id and TG_TOKEN:
        log("step 4/4 — sending to Telegram...")
        caption = f"🎬 {creative['prompt'][:200]}"
        step4_telegram(chat_id, video_path, caption=caption)
        log("  sent!")
    else:
        log("step 4/4 — skipped (no chat_id or TELEGRAM_BOT_TOKEN)")

    return {**job, "_creative": creative}


# --------------------------------------------------------------------------- #
# Alternate entry point: skip LM, use pre-computed creative (from preview)    #
# --------------------------------------------------------------------------- #
def run_from_creative(
    creative: dict[str, Any],
    chat_id: str | int | None = None,
    on_progress: Callable[[str], None] | None = None,
    mode: str = "quality",
) -> dict[str, Any]:
    """Like run() but skips the LM step — creative JSON is already parsed."""
    log = on_progress or (lambda m: print(f"[orchestrate] {m}", file=sys.stderr, flush=True))
    log("step 2/4 — submitting to pipeline API (keyframe pre-approved)...")
    _lm_eject()  # LM may have been loaded earlier; free VRAM before render
    job_id = step2_submit_job(creative, mode=mode)
    log(f"  job_id: {job_id}")
    log("step 3/4 — rendering...")
    job = step3_poll(job_id, on_status=lambda s: log(f"  render status → {s}"))
    if job["status"] != "completed":
        errors = job.get("errors") or job.get("stderr_tail", [])[-3:]
        raise RuntimeError(f"render failed — errors: {errors}")
    video_path = job["final_video_path"]
    log(f"  done: {video_path}  ({job.get('runtime_seconds', '?')}s)")
    if chat_id and TG_TOKEN:
        log("step 4/4 — sending to Telegram...")
        step4_telegram(chat_id, video_path, caption=f"🎬 {creative['prompt'][:200]}")
        log("  sent!")
    return {**job, "_creative": creative}


# --------------------------------------------------------------------------- #
# Service API                                                                  #
# --------------------------------------------------------------------------- #
_pipeline_jobs: dict[str, dict[str, Any]] = {}
_pj_lock = threading.Lock()

# Preview store: {preview_id → {creative, mode}} — persists across turns so
# the AI can reference a keyframe by ID after the user approves it.
_preview_store: dict[str, dict[str, Any]] = {}
_preview_lock = threading.Lock()


class VideoRequest(BaseModel):
    brief: str
    chat_id: Optional[str] = None
    mode: str = "quality"  # test | standard | quality
    preview_id: Optional[str] = None  # if set, skip LM step and use stored creative


class PipelineJobInfo(BaseModel):
    id: str
    status: str
    created_at: str
    finished_at: Optional[str] = None
    final_video_path: Optional[str] = None
    runtime_seconds: Optional[float] = None
    errors: list[str] = []
    creative: Optional[dict[str, Any]] = None


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _run_pipeline(pj_id: str, brief: str, chat_id: str | None, mode: str = "quality",
                  preview_id: str | None = None) -> None:
    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)
        with _pj_lock:
            _pipeline_jobs[pj_id]["logs"] = logs

    try:
        if preview_id:
            with _preview_lock:
                stored = _preview_store.get(preview_id)
            if not stored:
                raise RuntimeError(f"preview_id '{preview_id}' not found or expired")
            creative = stored["creative"]
            mode = stored.get("mode", mode)
            result = run_from_creative(creative, chat_id=chat_id, on_progress=_log, mode=mode)
        else:
            result = run(brief, chat_id=chat_id, on_progress=_log, mode=mode)
        with _pj_lock:
            _pipeline_jobs[pj_id].update({
                "status": "completed",
                "finished_at": _now(),
                "final_video_path": result.get("final_video_path"),
                "runtime_seconds": result.get("runtime_seconds"),
                "creative": result.get("_creative"),
                "errors": [],
                "logs": logs,
            })
    except Exception as exc:
        with _pj_lock:
            _pipeline_jobs[pj_id].update({
                "status": "failed",
                "finished_at": _now(),
                "errors": [str(exc)],
                "logs": logs,
            })


app = FastAPI(title="HerVid Orchestrator", version="1.0.0")

import traceback as _traceback
from fastapi import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse

@app.exception_handler(Exception)
async def _debug_exception_handler(request: _Request, exc: Exception):
    tb = _traceback.format_exc()
    print(f"[orchestrate/ERROR] {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr, flush=True)
    return _JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__, "traceback": tb})


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hervid-orchestrator",
        "version": "1.0.0",
        "pipeline_api": HVP_URL,
        "lm_studio": LM_URL,
        "lm_model": LM_MODEL,
        "endpoints": ["/generate-video", "/generate-preview", "/pipeline-job/{id}", "/pipeline-jobs", "/health"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    lm_ok = False
    try:
        r = requests.get(f"{LM_URL}/models", timeout=5)
        lm_ok = r.status_code == 200
    except Exception:
        pass
    hvp_ok = False
    hvp_busy = False
    try:
        r = requests.get(f"{HVP_URL}/health", timeout=5)
        d = r.json()
        hvp_ok = d.get("ok", False)
        hvp_busy = d.get("busy", False)
    except Exception:
        pass
    return {
        "ok": lm_ok and hvp_ok,
        "lm_studio": lm_ok,
        "pipeline_api": hvp_ok,
        "pipeline_busy": hvp_busy,
    }


@app.post("/generate-video")
def generate_video(req: VideoRequest) -> dict[str, Any]:
    pj_id = str(uuid.uuid4())
    with _pj_lock:
        _pipeline_jobs[pj_id] = {
            "id": pj_id,
            "status": "running",
            "created_at": _now(),
            "finished_at": None,
            "final_video_path": None,
            "runtime_seconds": None,
            "creative": None,
            "errors": [],
            "logs": [],
        }
    t = threading.Thread(
        target=_run_pipeline,
        args=(pj_id, req.brief, req.chat_id, req.mode, req.preview_id),
        daemon=True,
    )
    t.start()
    return {"id": pj_id, "status": "running"}


class PreviewRequest(BaseModel):
    brief: str
    mode: str = "quality"  # test | standard | quality


@app.post("/generate-preview")
def generate_preview(req: PreviewRequest) -> dict[str, Any]:
    """Step 1+keyframe: parse brief → creative JSON → render Flux keyframe → return image."""
    log = lambda m: print(f"[orchestrate/preview] {m}", file=sys.stderr, flush=True)

    log("loading LM model...")
    _lm_load()
    try:
        creative = step1_parse_brief(req.brief)
    finally:
        _lm_eject()

    log(f"parsed: {creative['prompt'][:80]}")

    # Submit a keyframe-only job to pipeline_api
    prompt = creative["prompt"]
    if "sharp" not in prompt.lower() and "detail" not in prompt.lower():
        prompt = prompt.rstrip(". ") + _FACE_DETAIL_SUFFIX

    body = {
        "prompt": prompt,
        "character_note": creative.get("character_note", ""),
        "animation": creative.get("animation", "auto"),
        "mode": "test" if req.mode == "test" else "quality",
        "keyframe_only": True,
        "skip_face_restore": _is_animal_content(creative.get("character_note", "")),
    }
    resp = requests.post(f"{HVP_URL}/generate-sequence", json=body, timeout=30)
    resp.raise_for_status()
    kf_job_id = resp.json()["id"]
    log(f"keyframe job: {kf_job_id}")

    # Poll until done (keyframe is fast: ~30-90s)
    deadline = time.time() + 300
    while time.time() < deadline:
        r = requests.get(f"{HVP_URL}/job/{kf_job_id}", timeout=15)
        r.raise_for_status()
        job = r.json()
        if job["status"] == "completed":
            image_path = job.get("image_path")
            if not image_path:
                raise RuntimeError(f"keyframe job completed but no image_path: {job}")
            # Store creative so generate-video can reuse it
            preview_id = str(uuid.uuid4())
            with _preview_lock:
                _preview_store[preview_id] = {"creative": creative, "mode": req.mode}
            log(f"keyframe done: {image_path} → preview_id={preview_id}")
            return {
                "preview_id": preview_id,
                "image_path": image_path,
                "prompt": creative["prompt"],
                "character_note": creative.get("character_note", ""),
                "mode": req.mode,
            }
        if job["status"] == "failed":
            errs = job.get("errors") or job.get("stderr_tail", [])[-3:]
            raise RuntimeError(f"keyframe render failed: {errs}")
        time.sleep(10)

    raise TimeoutError(f"keyframe job {kf_job_id} did not complete within 300s")


@app.get("/pipeline-job/{pj_id}")
def get_pipeline_job(pj_id: str) -> dict[str, Any]:
    job = _pipeline_jobs.get(pj_id)
    if not job:
        raise HTTPException(404, f"pipeline job not found: {pj_id}")
    return job


@app.get("/pipeline-jobs")
def list_pipeline_jobs() -> list[dict[str, Any]]:
    with _pj_lock:
        return sorted(_pipeline_jobs.values(), key=lambda j: j["created_at"], reverse=True)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HerVid Orchestrator")
    sub = parser.add_subparsers(dest="cmd")

    # python orchestrate.py run "brief" [--chat-id 123]
    run_p = sub.add_parser("run", help="Run full pipeline (blocks until done)")
    run_p.add_argument("brief")
    run_p.add_argument("--chat-id", default=None)

    # python orchestrate.py parse "brief"
    parse_p = sub.add_parser("parse", help="Only step 1: parse brief → creative JSON")
    parse_p.add_argument("brief")

    # python orchestrate.py serve
    sub.add_parser("serve", help="Start FastAPI service on port 8501")

    # Positional fallback: python orchestrate.py "brief" (shorthand for run)
    parser.add_argument("brief_fallback", nargs="?", default=None)

    args = parser.parse_args()

    if args.cmd == "parse" or (args.cmd is None and args.brief_fallback):
        brief = args.brief if args.cmd == "parse" else args.brief_fallback
        creative = step1_parse_brief(brief)
        print(json.dumps(creative, ensure_ascii=False, indent=2))

    elif args.cmd == "run":
        result = run(args.brief, chat_id=args.chat_id)
        print(json.dumps({
            "status": result["status"],
            "video": result.get("final_video_path"),
            "runtime_s": result.get("runtime_seconds"),
            "creative": result.get("_creative"),
        }, ensure_ascii=False, indent=2))

    elif args.cmd == "serve" or args.cmd is None:
        import uvicorn
        uvicorn.run(app, host=HOST, port=PORT, workers=1)
