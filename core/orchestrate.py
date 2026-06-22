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
LM_MODEL = os.environ.get("LM_MODEL", "gemma-4-12b-qat")
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
- "prompt": vivid English description of what happens in the video (1-3 sentences, present tense, specific actions and setting)
- "character_note": English description of the main character or subject appearance for visual consistency across all shots (e.g. "orange-brown bear cub, red bib, round eyes, 3D Pixar style")
- "total_duration_seconds": integer 6-60, estimated duration (default 30)
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
            if m.get("key") == LM_MODEL and m.get("loaded_instances"):
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
        "max_tokens": 800,  # gemma-4-12b needs headroom for reasoning before JSON output
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        resp = requests.post(f"{LM_URL}/chat/completions", json=payload, timeout=90)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if model added them
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(text)
            if "prompt" not in parsed:
                raise ValueError("missing 'prompt' field")
            parsed.setdefault("character_note", "")
            parsed.setdefault("total_duration_seconds", 30)
            parsed.setdefault("animation", "auto")
            parsed["total_duration_seconds"] = max(6, min(60, int(parsed["total_duration_seconds"])))
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_err = e
    raise RuntimeError(
        f"LM Studio returned unparseable JSON after {retries} attempts. "
        f"Last error: {last_err}. Last text: {text!r}"
    )


# --------------------------------------------------------------------------- #
# Step 2 — Pipeline API: submit job                                            #
# --------------------------------------------------------------------------- #
def step2_submit_job(creative: dict[str, Any]) -> str:
    body = {
        "prompt": creative["prompt"],
        "character_note": creative.get("character_note", ""),
        "total_duration_seconds": creative.get("total_duration_seconds", 30),
        "animation": creative.get("animation", "auto"),
        "mode": "quality",
    }
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
    job_id = step2_submit_job(creative)
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
# Service API                                                                  #
# --------------------------------------------------------------------------- #
_pipeline_jobs: dict[str, dict[str, Any]] = {}
_pj_lock = threading.Lock()


class VideoRequest(BaseModel):
    brief: str
    chat_id: Optional[str] = None


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


def _run_pipeline(pj_id: str, brief: str, chat_id: str | None) -> None:
    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)
        with _pj_lock:
            _pipeline_jobs[pj_id]["logs"] = logs

    try:
        result = run(brief, chat_id=chat_id, on_progress=_log)
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


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hervid-orchestrator",
        "version": "1.0.0",
        "pipeline_api": HVP_URL,
        "lm_studio": LM_URL,
        "lm_model": LM_MODEL,
        "endpoints": ["/generate-video", "/pipeline-job/{id}", "/pipeline-jobs", "/health"],
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
    t = threading.Thread(target=_run_pipeline, args=(pj_id, req.brief, req.chat_id), daemon=True)
    t.start()
    return {"id": pj_id, "status": "running"}


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
