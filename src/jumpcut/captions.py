"""Build captions on the OUTPUT timeline.

After cutting, every surviving word sits at a new timestamp. We remap each kept
word by subtracting the total duration removed before it, then group words into
readable caption lines. `caption_lines()` is the single source of truth used both
for the .srt sidecar and for the burned-in PNG overlay captions (see render.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import EditPlan, Segment, Word


@dataclass
class CaptionLine:
    text: str
    start: float
    end: float


def _remap_words(plan: EditPlan) -> list[Word]:
    """Return kept words with start/end translated to the output timeline."""
    keep = plan.keep
    out: list[Word] = []
    # cumulative output-time at the start of each keep segment
    offsets: list[float] = []
    acc = 0.0
    for s in keep:
        offsets.append(acc)
        acc += s.duration

    def seg_index(t: float) -> int | None:
        for i, s in enumerate(keep):
            if s.start - 1e-6 <= t <= s.end + 1e-6:
                return i
        return None

    for w in plan.transcript.words:
        # word must survive: its midpoint falls inside a keep segment
        mid = (w.start + w.end) / 2
        i = seg_index(mid)
        if i is None:
            continue
        seg: Segment = keep[i]
        new_start = offsets[i] + (max(w.start, seg.start) - seg.start)
        new_end = offsets[i] + (min(w.end, seg.end) - seg.start)
        if new_end > new_start:
            out.append(Word(text=w.text, start=new_start, end=new_end))
    return out


def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def caption_lines(plan: EditPlan, max_chars: int = 42, max_gap: float = 0.8) -> list[CaptionLine]:
    """Group remapped words into readable caption lines on the output timeline."""
    words = _remap_words(plan)
    if not words:
        return []

    groups: list[list[Word]] = [[words[0]]]
    for w in words[1:]:
        cur = groups[-1]
        cur_text = " ".join(x.text for x in cur)
        gap = w.start - cur[-1].end
        if len(cur_text) + 1 + len(w.text) > max_chars or gap > max_gap:
            groups.append([w])
        else:
            cur.append(w)

    return [
        CaptionLine(
            text=" ".join(w.text for w in g).strip(),
            start=g[0].start,
            end=g[-1].end,
        )
        for g in groups
    ]


def build_srt(plan: EditPlan, max_chars: int = 42, max_gap: float = 0.8) -> str:
    """Render caption lines as .srt text."""
    lines = caption_lines(plan, max_chars=max_chars, max_gap=max_gap)
    blocks = [
        f"{idx}\n{_fmt_ts(line.start)} --> {_fmt_ts(line.end)}\n{line.text}\n"
        for idx, line in enumerate(lines, 1)
    ]
    return "\n".join(blocks)
