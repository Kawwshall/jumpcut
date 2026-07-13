"""Transcription: video -> word-level timeline via faster-whisper.

We extract a 16 kHz mono WAV with ffmpeg (whisper's native input), then run
faster-whisper with word timestamps on. Everything downstream keys off this.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from . import ffbin
from .models import Transcript, Word


def probe_duration(video: Path) -> float:
    """Total media duration in seconds, via ffprobe."""
    cmd = [
        ffbin.ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return float(out) if out else 0.0


def _extract_audio(video: Path, wav_path: Path) -> None:
    cmd = [
        ffbin.ffmpeg(),
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def transcribe(
    video: Path,
    model_size: str = "base",
    language: str | None = None,
    on_progress=None,
) -> Transcript:
    """Return a word-level Transcript for `video`.

    `model_size`: tiny/base/small/medium/large-v3. Bigger = more accurate, slower.
    `on_progress`: optional callback(seconds_done, total_seconds) for a progress bar.
    """
    import os

    from faster_whisper import WhisperModel

    duration = probe_duration(video)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        _extract_audio(video, wav)

        # int8 on CPU is the sweet spot on Apple Silicon: fast, low memory.
        # Use every core for decode, and greedy search (beam_size=1) instead
        # of the library default (5) — for cutting silence/fillers we need
        # accurate word timestamps, not competition-grade transcription
        # accuracy, and greedy decoding is noticeably faster.
        model = WhisperModel(
            model_size, device="cpu", compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
        segments, info = model.transcribe(
            str(wav),
            language=language,
            word_timestamps=True,
            vad_filter=True,  # skip long non-speech chunks up front
            beam_size=1,
            condition_on_previous_text=False,
        )

        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
            if on_progress:
                on_progress(min(seg.end, duration), duration)

    return Transcript(
        words=words,
        language=getattr(info, "language", language or "en"),
        duration=duration or (words[-1].end if words else 0.0),
    )
