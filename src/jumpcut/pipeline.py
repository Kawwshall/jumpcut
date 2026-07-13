"""The end-to-end pipeline, shared by the CLI and the web GUI.

ingest -> transcribe -> analyze -> render, reported through a single
`on_stage(stage, **info)` callback so any frontend (rich terminal UI, web
progress bar) can drive off the same sequence of events.

Stages emitted, in order:
  resolving         -> {"index": int, "total": int}       (once per source clip)
  resolved          -> {"index": int, "total": int, "filename": str}
  concatenating     -> {"count": int}                      (only if >1 source)
  concatenated      -> {"filename": str}                   (only if >1 source)
  source_ready      -> {"filename": str, "path": Path}     (the video that gets transcribed)
  transcribing      -> {"done": float, "total": float}     (repeated)
  transcribed       -> {"words": int, "duration": float}
  analyzing         -> {}
  planned           -> {"plan": EditPlan}
  nothing_left      -> {}                                 (terminal, error)
  rendering         -> {}
  rendered          -> {"mode": "burned"|"soft"|"none"}
  done              -> {"output": Path, "srt": Path | None}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import analyze, ingest, render as render_mod, transcribe
from .transcribe import probe_duration
from .captions import build_srt
from .models import EditPlan

StageCallback = Callable[..., None]


@dataclass
class PipelineOptions:
    silence: bool = True
    fillers: bool = True
    captions: bool = True
    retakes: bool = False
    min_gap: float = 0.6
    aggressive_fillers: bool = False
    whisper_model: str = "base"
    keep_srt: bool = False
    retake_model: str | None = None


@dataclass
class PipelineResult:
    source: Path
    output: Path
    srt: Path | None
    plan: EditPlan
    caption_mode: str


def has_anthropic() -> bool:
    import os
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def run_pipeline(
    sources: str | list[str],
    output: Path | None,
    opts: PipelineOptions,
    workdir: Path,
    on_stage: StageCallback | None = None,
    dry_run: bool = False,
) -> PipelineResult | EditPlan:
    """Run the full pipeline. `sources` is one video/URL, or a list of several
    to be stitched together (in order) into a single timeline before editing.
    Returns a PipelineResult, or (if dry_run) just the EditPlan without
    rendering anything.

    Wrapped in `stay_awake()` so a lid-close or idle-sleep mid-job can't turn
    a few-minute render into an hour-plus one — the system staying asleep
    doesn't make the CPU work slower, it just stops it from running at all.
    """
    from . import ffbin
    with ffbin.stay_awake():
        return _run_pipeline_impl(sources, output, opts, workdir, on_stage, dry_run)


def _run_pipeline_impl(
    sources: str | list[str],
    output: Path | None,
    opts: PipelineOptions,
    workdir: Path,
    on_stage: StageCallback | None,
    dry_run: bool,
) -> PipelineResult | EditPlan:

    def emit(stage: str, **info):
        if on_stage:
            on_stage(stage, **info)

    workdir.mkdir(parents=True, exist_ok=True)

    src_list = [sources] if isinstance(sources, str) else list(sources)
    if not src_list:
        raise RuntimeError("No source clips provided.")

    resolved: list[Path] = []
    for i, src in enumerate(src_list):
        emit("resolving", index=i, total=len(src_list))
        p = ingest.resolve_source(src, workdir / "downloads")
        resolved.append(p)
        # Surface each clip's actual duration right away — a URL that only
        # partially downloads (e.g. a broken/short grab) is invisible once
        # it's buried inside a much longer combined video, so catch it here.
        dur = probe_duration(p)
        emit("resolved", index=i, total=len(src_list), filename=p.name, duration=dur)

    if len(resolved) > 1:
        emit("concatenating", count=len(resolved))
        # Keep the inter-clip gap comfortably above the silence threshold so the
        # seam between clips is always cut, however the user's slider is set.
        seam_gap = max(0.5, opts.min_gap + 0.2)
        video = ingest.concat_sources(resolved, workdir, gap=seam_gap)
        emit("concatenated", filename=video.name)
    else:
        video = resolved[0]
    emit("source_ready", filename=video.name, path=video)

    def _tick(done, total):
        emit("transcribing", done=done, total=total)

    tx = transcribe.transcribe(video, model_size=opts.whisper_model, on_progress=_tick)
    emit("transcribed", words=len(tx.words), duration=tx.duration)

    lexicon = analyze.DEFAULT_FILLERS if opts.aggressive_fillers else analyze.SAFE_FILLERS
    retakes = opts.retakes
    if retakes and not has_anthropic():
        retakes = False

    emit("analyzing")
    plan = analyze.build_plan(
        tx, source=str(video),
        cut_silence=opts.silence, cut_fillers=opts.fillers, cut_retakes=retakes,
        min_gap=opts.min_gap, filler_lexicon=lexicon, retake_model=opts.retake_model,
    )
    emit("planned", plan=plan)

    if dry_run:
        return plan
    if not plan.keep:
        emit("nothing_left")
        raise RuntimeError("Nothing left after cuts. Loosen your settings.")

    out = output or video.with_name(f"{video.stem}.jumpcut.mp4")
    srt_path = None
    if opts.captions or opts.keep_srt:
        srt_text = build_srt(plan)
        srt_path = workdir / f"{video.stem}.srt"
        srt_path.write_text(srt_text, encoding="utf-8")

    emit("rendering")
    mode = render_mod.render(
        plan, source=video, out_path=out,
        srt_path=srt_path if opts.captions else None,
    )
    emit("rendered", mode=mode)

    final_srt = None
    if opts.keep_srt and srt_path:
        final_srt = out.with_suffix(".srt")
        final_srt.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")

    emit("done", output=out, srt=final_srt)
    return PipelineResult(source=video, output=out, srt=final_srt, plan=plan, caption_mode=mode)
