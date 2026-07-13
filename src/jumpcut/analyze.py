"""The editorial brain: turn a Transcript into an EditPlan.

Three detectors each emit `Cut`s (ranges to remove):
  - silence:  gaps between words longer than a threshold
  - filler:   words matching a filler lexicon (um, uh, like, ...)
  - retake:   false starts / repeated sentences (optional, LLM-driven)

We then merge overlapping cuts and invert them into the `keep` segments,
padded slightly so speech isn't clipped at the edges.
"""

from __future__ import annotations

import re

from .models import Cut, CutReason, EditPlan, Segment, Transcript, Word

# Common English fillers. Matched on the bare word (punctuation/case stripped).
DEFAULT_FILLERS = {
    "um", "uh", "umm", "uhh", "erm", "er", "ah", "hmm",
    "like", "basically", "literally", "actually", "honestly",
    "right", "okay", "so", "well", "yeah",
}
# The riskier fillers (like/so/right/well/okay/yeah/actually...) are only cut
# when they're clearly standalone crutch words, not carrying meaning. To stay
# safe by default we only auto-cut the unambiguous non-words.
SAFE_FILLERS = {"um", "uh", "umm", "uhh", "erm", "er", "ah", "hmm"}

_NORM_RE = re.compile(r"[^a-z']")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", text.lower())


def detect_silence(
    words: list[Word],
    total_duration: float,
    min_gap: float = 0.6,
    lead_in: float = 0.15,
) -> list[Cut]:
    """Cut gaps between words longer than `min_gap`, keeping a small lead-in
    so the next word doesn't start abruptly. Also trims head/tail dead air."""
    cuts: list[Cut] = []
    if not words:
        return cuts

    # leading dead air
    if words[0].start > min_gap:
        cuts.append(Cut(0.0, words[0].start - lead_in, CutReason.SILENCE, "lead"))

    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        if gap > min_gap:
            cuts.append(
                Cut(prev.end + lead_in, nxt.start - lead_in, CutReason.SILENCE,
                    f"{gap:.1f}s pause")
            )

    # trailing dead air
    tail = total_duration - words[-1].end
    if tail > min_gap:
        cuts.append(Cut(words[-1].end + lead_in, total_duration, CutReason.SILENCE, "tail"))

    return [c for c in cuts if c.duration > 0.05]


def detect_fillers(words: list[Word], lexicon: set[str] | None = None) -> list[Cut]:
    """Cut individual words that match the filler lexicon."""
    lex = lexicon if lexicon is not None else SAFE_FILLERS
    cuts: list[Cut] = []
    for w in words:
        if _norm(w.text) in lex:
            cuts.append(Cut(w.start, w.end, CutReason.FILLER, w.text))
    return cuts


def detect_retakes(transcript: Transcript, model: str | None = None) -> list[Cut]:
    """Use an LLM to find false starts / retakes ('let me redo that', repeated
    sentences) and return the time ranges of the *discarded* attempts.

    Requires the `anthropic` extra and ANTHROPIC_API_KEY. Returns [] if either
    is missing, so the pipeline degrades gracefully.
    """
    import json
    import os

    if not transcript.words or not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []

    # Number the words so the model can reference exact indices -> timestamps.
    numbered = " ".join(f"[{i}]{w.text}" for i, w in enumerate(transcript.words))
    prompt = (
        "You are a video editor cleaning up a talking-head screen recording. "
        "The transcript below has each word tagged with an index like [12]word. "
        "Find spans that are FALSE STARTS or RETAKES: the speaker flubbed a line "
        "and restarted, said 'let me redo that', or repeated the same sentence. "
        "For each such span, return the index range of the BAD attempt to delete "
        "(keep the good/final take). Be conservative — only clear retakes.\n\n"
        "Return ONLY JSON: {\"cuts\": [{\"start_idx\": int, \"end_idx\": int, "
        "\"reason\": str}]}\n\n"
        f"Transcript:\n{numbered}"
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    cuts: list[Cut] = []
    n = len(transcript.words)
    for item in data.get("cuts", []):
        try:
            i, j = int(item["start_idx"]), int(item["end_idx"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= i <= j < n:
            cuts.append(
                Cut(transcript.words[i].start, transcript.words[j].end,
                    CutReason.RETAKE, str(item.get("reason", ""))[:80])
            )
    return cuts


def _merge_cuts(cuts: list[Cut]) -> list[Cut]:
    """Merge overlapping/adjacent cuts into disjoint ranges (reason of first)."""
    if not cuts:
        return []
    ordered = sorted(cuts, key=lambda c: c.start)
    merged = [ordered[0]]
    for c in ordered[1:]:
        last = merged[-1]
        if c.start <= last.end + 0.01:
            last.end = max(last.end, c.end)
        else:
            merged.append(c)
    return merged


def _invert(cuts: list[Cut], total: float, min_keep: float = 0.05) -> list[Segment]:
    """Turn removal ranges into the complementary keep segments."""
    keep: list[Segment] = []
    cursor = 0.0
    for c in cuts:
        if c.start > cursor:
            keep.append(Segment(cursor, min(c.start, total)))
        cursor = max(cursor, c.end)
    if cursor < total:
        keep.append(Segment(cursor, total))
    return [s for s in keep if s.duration >= min_keep]


def build_plan(
    transcript: Transcript,
    source: str,
    cut_silence: bool = True,
    cut_fillers: bool = True,
    cut_retakes: bool = False,
    min_gap: float = 0.6,
    filler_lexicon: set[str] | None = None,
    retake_model: str | None = None,
) -> EditPlan:
    cuts: list[Cut] = []
    if cut_silence:
        cuts += detect_silence(transcript.words, transcript.duration, min_gap=min_gap)
    if cut_fillers:
        cuts += detect_fillers(transcript.words, filler_lexicon)
    if cut_retakes:
        cuts += detect_retakes(transcript, model=retake_model)

    merged = _merge_cuts(cuts)
    keep = _invert(merged, transcript.duration)
    return EditPlan(source=source, transcript=transcript, cuts=merged, keep=keep)
