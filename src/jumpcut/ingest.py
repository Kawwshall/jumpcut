"""Turn whatever the user gives us (a path or a Loom URL) into a local video file,
and stitch multiple clips into one timeline when there's more than one."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import ffbin

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source.strip()))


def resolve_source(source: str, workdir: Path) -> Path:
    """Return a local Path to the video, downloading first if `source` is a URL."""
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        return path
    return _download(source, workdir)


def _download(url: str, workdir: Path) -> Path:
    """Download a Loom (or any yt-dlp-supported) URL into workdir."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yt-dlp is required to download URLs. Install with: uv sync"
        ) from exc

    workdir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(workdir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "mp4/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tmpl,
        "--print",
        "after_move:filepath",
        "--no-simulate",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed:\n{result.stderr.strip()}")
    # last non-empty stdout line is the final filepath
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Download reported success but produced no file path.")
    return Path(lines[-1]).resolve()


def concat_sources(
    paths: list[Path],
    workdir: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    out_name: str = "combined.mp4",
    gap: float = 0.5,
) -> Path:
    """Stitch multiple clips (in order) into one continuous video.

    Clips can differ in resolution, fps, codec, or aspect ratio (e.g. a mix of
    screen recordings and phone footage) — each is scaled/padded to a common
    canvas and its audio resampled to a common format before concatenation, so
    the ffmpeg `concat` filter (which requires matching stream parameters)
    always has something uniform to work with.

    A short black+silent `gap` is inserted between clips. Back-to-back clips
    otherwise have no natural pause at the seam, so without this the silence
    detector won't cut there and captions can bridge across two unrelated
    clips into one nonsensical line. `gap` should be >= the pipeline's
    min_gap silence threshold so the seam always gets treated as a real cut.
    """
    if len(paths) < 2:
        raise ValueError("concat_sources needs at least two clips.")

    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / out_name

    inputs: list[str] = []
    filters: list[str] = []
    concat_inputs: list[str] = []
    n_gaps = len(paths) - 1
    next_input_idx = 0

    for i, p in enumerate(paths):
        clip_idx = next_input_idx
        next_input_idx += 1
        inputs += ["-i", str(p)]
        filters.append(
            f"[{clip_idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]"
        )
        filters.append(f"[{clip_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")

        if i < n_gaps:
            gi = f"g{i}"
            color_idx = next_input_idx
            null_idx = next_input_idx + 1
            next_input_idx += 2
            inputs += ["-f", "lavfi", "-t", f"{gap:.3f}",
                       "-i", f"color=c=black:s={width}x{height}:r={fps}"]
            inputs += ["-f", "lavfi", "-t", f"{gap:.3f}",
                       "-i", "anullsrc=r=48000:cl=stereo"]
            filters.append(f"[{color_idx}:v]format=yuv420p,setsar=1[{gi}v]")
            filters.append(f"[{null_idx}:a]anull[{gi}a]")
            concat_inputs.append(f"[{gi}v][{gi}a]")

    total_segments = len(paths) + n_gaps
    filters.append(f"{''.join(concat_inputs)}concat=n={total_segments}:v=1:a=1[vout][aout]")

    cmd = [
        ffbin.ffmpeg(), "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"Failed to combine clips:\n{tail}")
    return out
