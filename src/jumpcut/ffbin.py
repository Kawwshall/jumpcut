"""Locate the best available ffmpeg/ffprobe binaries.

Homebrew's default `ffmpeg` formula ships without libass/libfreetype — no
burned-in subtitle support, and (per render.py's caption strategy) that used
to mean falling back to per-line image overlays chained one-by-one, which
doesn't scale past a couple dozen caption lines. `ffmpeg-full` ships both
libass and libfreetype and has a precompiled bottle, but it's keg-only (not
symlinked onto PATH) so a plain `ffmpeg` call won't find it. We look for it
explicitly and prefer it; if it's not installed, we fall back to whatever
`ffmpeg`/`ffprobe` is on PATH.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

_FULL_PREFIX_CANDIDATES = [
    Path("/opt/homebrew/opt/ffmpeg-full/bin"),  # Apple Silicon Homebrew
    Path("/usr/local/opt/ffmpeg-full/bin"),  # Intel Homebrew
]


@functools.cache
def ffmpeg() -> str:
    for prefix in _FULL_PREFIX_CANDIDATES:
        candidate = prefix / "ffmpeg"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg") or "ffmpeg"


@functools.cache
def ffprobe() -> str:
    for prefix in _FULL_PREFIX_CANDIDATES:
        candidate = prefix / "ffprobe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffprobe") or "ffprobe"


@contextmanager
def stay_awake():
    """Prevent macOS from sleeping (or App Nap throttling this process) for
    as long as the block runs. A transcribe+render job that would normally
    take a few minutes can otherwise stretch to over an hour if the lid
    closes or the machine goes idle mid-job — the CPU work itself doesn't
    slow down, it just stops running while the system sleeps. `caffeinate -w
    <our pid>` ties the assertion to our own lifetime, so it cleans up
    automatically even if we crash. No-op on non-macOS or if caffeinate is
    missing.
    """
    proc = None
    if sys.platform == "darwin":
        caffeinate = shutil.which("caffeinate")
        if caffeinate:
            proc = subprocess.Popen(
                [caffeinate, "-i", "-s", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
