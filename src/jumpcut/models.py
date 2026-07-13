"""Core data types shared across the pipeline.

The whole tool is built on one idea: a video is a list of *words on a timeline*.
Every edit decision is expressed as a `Cut` (a time range to remove) with a
reason. We invert the cuts into the ranges we keep, then hand those to ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CutReason(str, Enum):
    SILENCE = "silence"
    FILLER = "filler"
    RETAKE = "retake"


@dataclass
class Word:
    """A single transcribed word with its span on the audio timeline (seconds)."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Cut:
    """A time range to remove from the source, and why."""

    start: float
    end: float
    reason: CutReason
    detail: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """A contiguous range we KEEP in the final render."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class EditPlan:
    """The full result of analysis: what to cut, what to keep, and the stats."""

    source: str
    transcript: Transcript
    cuts: list[Cut]
    keep: list[Segment]

    @property
    def original_duration(self) -> float:
        return self.transcript.duration

    @property
    def final_duration(self) -> float:
        return sum(s.duration for s in self.keep)

    @property
    def removed_duration(self) -> float:
        return self.original_duration - self.final_duration
