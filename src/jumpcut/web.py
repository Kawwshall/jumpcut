"""Local web GUI for jumpcut.

Runs a small FastAPI app: drop in a Loom (file or URL), watch it transcribe /
analyze / render in the browser, then preview original vs. edited side by side.

    uv run jumpcut-web

Single-user, local-only tool — job state lives in memory, no auth, no queueing
beyond "one job at a time per id". Not meant to be exposed to the network.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import render as render_mod
from .models import CutReason, EditPlan
from .pipeline import PipelineOptions, has_anthropic, run_pipeline

ROOT = Path(".jumpcut/web").resolve()
JOBS_DIR = ROOT / "jobs"


@dataclass
class Job:
    id: str
    stage: str = "queued"
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    output_name: str | None = None
    srt_name: str | None = None
    original_name: str | None = None
    plan_stats: dict[str, Any] | None = None
    caption_mode: str | None = None
    clips: list[dict[str, Any]] = field(default_factory=list)


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()

app = FastAPI(title="jumpcut")


def _plan_stats(plan: EditPlan) -> dict[str, Any]:
    counts: dict[str, dict[str, float | int]] = {}
    for c in plan.cuts:
        d = counts.setdefault(c.reason.value, {"count": 0, "seconds": 0.0})
        d["count"] += 1
        d["seconds"] += c.duration
    return {
        "by_reason": counts,
        "original_duration": plan.original_duration,
        "final_duration": plan.final_duration,
        "removed_duration": plan.removed_duration,
        "percent_shorter": (
            plan.removed_duration / plan.original_duration * 100
            if plan.original_duration else 0
        ),
    }


def _run_job(job_id: str, sources: list[str], opts: PipelineOptions) -> None:
    job = JOBS[job_id]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def on_stage(stage: str, **info):
        job.stage = stage
        if stage == "transcribing":
            job.detail = {"done": info["done"], "total": info["total"]}
        elif stage == "resolving":
            job.detail = {"index": info["index"], "total": info["total"]}
        elif stage == "resolved":
            job.clips.append({
                "index": info["index"], "filename": info["filename"],
                "duration": info["duration"],
            })
        elif stage == "concatenating":
            job.detail = {"count": info["count"]}
        elif stage == "source_ready":
            src_path: Path = info["path"]
            local = job_dir / f"original{src_path.suffix}"
            if not local.exists():
                try:
                    local.symlink_to(src_path)
                except OSError:
                    shutil.copy(src_path, local)
            job.original_name = local.name
        elif stage == "planned":
            job.plan_stats = _plan_stats(info["plan"])
        elif stage == "rendered":
            job.caption_mode = info["mode"]

    try:
        result = run_pipeline(
            sources, output=job_dir / "output.mp4", opts=opts,
            workdir=job_dir, on_stage=on_stage,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.error = str(exc)
        job.stage = "error"
        return

    job.output_name = result.output.name
    if result.srt:
        srt_dest = job_dir / result.srt.name
        if result.srt != srt_dest:
            shutil.copy(result.srt, srt_dest)
        job.srt_name = srt_dest.name
    job.stage = "done"


@app.on_event("startup")
def _startup():
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(JOBS_DIR)), name="media")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/capabilities")
def capabilities():
    return {
        "can_burn_captions": render_mod.can_burn_captions(),
        "has_anthropic": has_anthropic(),
    }


def _form_bool(form, name: str, default: bool) -> bool:
    v = form.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _form_float(form, name: str, default: float) -> float:
    v = form.get(name)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


@app.post("/api/jobs")
async def create_job(request: Request):
    """Accepts an ordered queue of clips: a JSON `items` field describing the
    sequence (each `{"kind": "file", "id": "clip_0"}` or `{"kind": "url",
    "value": "..."}`), with the actual file blobs attached under field names
    matching each file item's `id`. Clips are stitched together in that order.
    """
    form = await request.form()

    items_raw = form.get("items")
    if not items_raw:
        raise HTTPException(400, "No clips provided.")
    try:
        items = json.loads(items_raw)
    except json.JSONDecodeError:
        raise HTTPException(400, "Malformed clip list.")
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "No clips provided.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    sources: list[str] = []
    for i, item in enumerate(items):
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind == "file":
            upload = form.get(item.get("id", ""))
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(400, f"Missing uploaded file for clip {i + 1}.")
            suffix = Path(getattr(upload, "filename", "") or "clip.mp4").suffix or ".mp4"
            local_path = job_dir / f"clip_{i:02d}{suffix}"
            local_path.write_bytes(await upload.read())
            sources.append(str(local_path))
        elif kind == "url":
            value = str(item.get("value", "")).strip()
            if not value:
                raise HTTPException(400, f"Empty URL for clip {i + 1}.")
            sources.append(value)
        else:
            raise HTTPException(400, f"Unrecognized clip entry at position {i + 1}.")

    opts = PipelineOptions(
        silence=_form_bool(form, "silence", True),
        fillers=_form_bool(form, "fillers", True),
        captions=_form_bool(form, "captions", True),
        retakes=_form_bool(form, "retakes", False),
        min_gap=_form_float(form, "min_gap", 0.6),
        aggressive_fillers=_form_bool(form, "aggressive_fillers", False),
        whisper_model=str(form.get("whisper_model") or "base"),
        keep_srt=True,
    )

    JOBS[job_id] = Job(id=job_id)
    thread = threading.Thread(target=_run_job, args=(job_id, sources, opts), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return {
        "id": job.id,
        "stage": job.stage,
        "detail": job.detail,
        "error": job.error,
        "plan_stats": job.plan_stats,
        "caption_mode": job.caption_mode,
        "clips": job.clips,
        "original_url": f"/media/{job.id}/{job.original_name}" if job.original_name else None,
        "output_url": f"/media/{job.id}/{job.output_name}" if job.output_name else None,
        "srt_url": f"/media/{job.id}/{job.srt_name}" if job.srt_name else None,
    }


def main():
    uvicorn.run("jumpcut.web:app", host="127.0.0.1", port=8756, reload=False)


if __name__ == "__main__":
    main()
