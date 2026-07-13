"""Render an EditPlan to a finished video with ffmpeg.

Strategy: a single filter_complex that trims each keep-segment from the source,
concatenates them, and — if there are captions — burns them in.

Caption burn strategy, in preference order:
  1. libass (`subtitles` filter) — burns the whole .srt in ONE filter pass,
     regardless of how many caption lines there are. This is what real video
     editors use. Requires an ffmpeg build with libass (see ffbin.py: we look
     for Homebrew's keg-only `ffmpeg-full`, which has it, since the default
     `ffmpeg` formula often doesn't).
  2. Per-line PNG overlays (Pillow-rendered) composited with ffmpeg's `overlay`
     filter — works on any ffmpeg build (overlay is a core filter), but each
     caption line adds one more filter to a serial chain, so this only stays
     fast for short clips. It's the fallback when libass isn't available.
  3. Soft mov_text track — last resort; many players (especially browsers)
     won't render this at all, so we only fall here if even `overlay` is
     missing (exceedingly rare).
"""

from __future__ import annotations

import functools
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import ffbin
from .captions import CaptionLine, caption_lines
from .models import EditPlan

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Caption line count above which the per-line overlay fallback is skipped
# entirely (too slow to be worth it) rather than left to grind for an hour.
_MAX_OVERLAY_LINES = 60


@functools.cache
def _has_filter(name: str) -> bool:
    out = subprocess.run(
        [ffbin.ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def _burn_strategy() -> str:
    """Which caption burn-in method is available: "libass", "overlay", or "none"."""
    if _has_filter("subtitles"):
        return "libass"
    if _has_filter("overlay"):
        return "overlay"
    return "none"


def can_burn_captions() -> bool:
    return _burn_strategy() != "none"


@functools.cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _probe_dims(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        [ffbin.ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _render_caption_png(text: str, out_path: Path, font_size: int, pad: int = 16) -> None:
    """Render one caption line as a tightly-cropped transparent PNG: white
    text with a black outline, TikTok/YouTube-Shorts style."""
    font = _font(font_size)
    scratch = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(scratch)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=max(2, font_size // 14))
    w = (bbox[2] - bbox[0]) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad - bbox[0]
    y = pad - bbox[1]
    d.text(
        (x, y), text, font=font, fill=(255, 255, 255, 255),
        stroke_width=max(2, font_size // 14), stroke_fill=(0, 0, 0, 255),
    )
    img.save(out_path)


def _escape_sub_path(path: Path) -> str:
    p = str(path)
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render(
    plan: EditPlan,
    source: Path,
    out_path: Path,
    srt_path: Path | None = None,
    crf: int = 23,
    preset: str = "ultrafast",
) -> str:
    """Render the kept segments of `source` to `out_path`.

    `ultrafast` + a slightly higher crf is the default: for talking-head
    screen recordings the quality difference against slower presets is not
    perceptible, but the encode is roughly 5-10x faster — measured at ~27x
    realtime on a 20-minute 720p source, vs. several minutes with `medium`.
    Speed is the point here, not squeezing out the last few % of file size.

    Returns which caption mode was used: "burned", "soft", or "none".
    """
    if not plan.keep:
        raise ValueError("Nothing left to render — the plan removed everything.")

    strategy = _burn_strategy() if srt_path is not None else "none"
    lines: list[CaptionLine] = []
    if strategy == "overlay":
        lines = caption_lines(plan)
        if len(lines) > _MAX_OVERLAY_LINES:
            # Chaining this many overlay filters would take forever (each
            # adds a serial stage ffmpeg must evaluate for the whole video).
            # Falling back to a soft track is faster than a multi-hour render.
            strategy = "soft"
    soft = strategy == "soft"
    burn = strategy in ("libass", "overlay")

    filters = []
    for i, seg in enumerate(plan.keep):
        filters.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        filters.append(
            f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(plan.keep)))
    n = len(plan.keep)
    filters.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vc][ac]")

    video_label = "[vc]"
    extra_inputs: list[str] = []
    png_tmpdir = None

    if strategy == "libass":
        filters.append(
            f"[vc]subtitles={_escape_sub_path(srt_path)}"
            f":force_style='FontSize=18,Bold=1,Outline=1,Shadow=0'[vsub]"
        )
        video_label = "[vsub]"

    elif strategy == "overlay" and lines:
        width, height = _probe_dims(source)
        font_size = max(20, round(height * 0.045))
        png_tmpdir = tempfile.TemporaryDirectory(prefix="jumpcut-caps-")
        png_dir = Path(png_tmpdir.name)

        for i, line in enumerate(lines):
            png_path = png_dir / f"cap_{i:05d}.png"
            _render_caption_png(line.text, png_path, font_size=font_size)
            extra_inputs += ["-i", str(png_path)]

        margin = max(24, round(height * 0.06))
        prev = video_label
        for i, line in enumerate(lines):
            img_idx = i + 1  # input 0 is the source video
            out_label = f"[ov{i}]"
            filters.append(
                f"{prev}[{img_idx}:v]overlay="
                f"x=(main_w-overlay_w)/2:y=main_h-overlay_h-{margin}:"
                f"enable='between(t,{line.start:.3f},{line.end:.3f})'{out_label}"
            )
            prev = out_label
        video_label = prev

    cmd = [ffbin.ffmpeg(), "-y", "-i", str(source), *extra_inputs]
    if soft:
        cmd += ["-i", str(srt_path)]

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", video_label,
        "-map", "[ac]",
    ]
    if soft:
        soft_idx = 1 + len(extra_inputs) // 2
        cmd += ["-map", f"{soft_idx}:0", "-c:s", "mov_text"]

    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if png_tmpdir is not None:
            png_tmpdir.cleanup()

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg render failed:\n{tail}")

    return "burned" if burn else "soft" if soft else "none"
