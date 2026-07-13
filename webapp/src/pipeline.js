// Client-side port of jumpcut's core pipeline (see src/jumpcut/analyze.py and
// captions.py for the original). Pure functions operating on plain
// {text, start, end} word objects — no server, no dependencies beyond what's
// loaded in index.html (ffmpeg.wasm + transformers.js).

export const SAFE_FILLERS = new Set(["um", "uh", "umm", "uhh", "erm", "er", "ah", "hmm"]);
export const AGGRESSIVE_FILLERS = new Set([
  ...SAFE_FILLERS,
  "like", "basically", "literally", "actually", "honestly",
  "right", "okay", "so", "well", "yeah",
]);

const _normRe = /[^a-z']/g;
function norm(text) {
  return text.toLowerCase().replace(_normRe, "");
}

/** Gaps between words longer than minGap become cuts, plus head/tail dead air. */
export function detectSilence(words, totalDuration, minGap = 0.6, leadIn = 0.15) {
  const cuts = [];
  if (!words.length) return cuts;

  if (words[0].start > minGap) {
    cuts.push({ start: 0, end: words[0].start - leadIn, reason: "silence", detail: "lead" });
  }
  for (let i = 0; i < words.length - 1; i++) {
    const gap = words[i + 1].start - words[i].end;
    if (gap > minGap) {
      cuts.push({
        start: words[i].end + leadIn, end: words[i + 1].start - leadIn,
        reason: "silence", detail: `${gap.toFixed(1)}s pause`,
      });
    }
  }
  const tail = totalDuration - words[words.length - 1].end;
  if (tail > minGap) {
    cuts.push({ start: words[words.length - 1].end + leadIn, end: totalDuration, reason: "silence", detail: "tail" });
  }
  return cuts.filter(c => c.end - c.start > 0.05);
}

export function detectFillers(words, lexicon = SAFE_FILLERS) {
  const cuts = [];
  for (const w of words) {
    if (lexicon.has(norm(w.text))) {
      cuts.push({ start: w.start, end: w.end, reason: "filler", detail: w.text });
    }
  }
  return cuts;
}

function mergeCuts(cuts) {
  if (!cuts.length) return [];
  const ordered = [...cuts].sort((a, b) => a.start - b.start);
  const merged = [{ ...ordered[0] }];
  for (const c of ordered.slice(1)) {
    const last = merged[merged.length - 1];
    if (c.start <= last.end + 0.01) {
      last.end = Math.max(last.end, c.end);
    } else {
      merged.push({ ...c });
    }
  }
  return merged;
}

function invertToKeep(cuts, total, minKeep = 0.05) {
  const keep = [];
  let cursor = 0;
  for (const c of cuts) {
    if (c.start > cursor) keep.push({ start: cursor, end: Math.min(c.start, total) });
    cursor = Math.max(cursor, c.end);
  }
  if (cursor < total) keep.push({ start: cursor, end: total });
  return keep.filter(s => s.end - s.start >= minKeep);
}

/** words: [{text,start,end}], returns {cuts, keep, originalDuration, finalDuration} */
export function buildPlan(words, totalDuration, opts) {
  let cuts = [];
  if (opts.silence) cuts = cuts.concat(detectSilence(words, totalDuration, opts.minGap ?? 0.6));
  if (opts.fillers) cuts = cuts.concat(detectFillers(words, opts.aggressiveFillers ? AGGRESSIVE_FILLERS : SAFE_FILLERS));

  const merged = mergeCuts(cuts);
  const keep = invertToKeep(merged, totalDuration);
  const removed = merged.reduce((sum, c) => sum + (c.end - c.start), 0);
  return {
    cuts: merged,
    keep,
    originalDuration: totalDuration,
    finalDuration: keep.reduce((s, k) => s + (k.end - k.start), 0),
    removedDuration: Math.min(removed, totalDuration),
  };
}

/** Remap words onto the post-cut output timeline, then group into caption lines. */
export function captionLines(words, keep, maxChars = 42, maxGap = 0.8) {
  const offsets = [];
  let acc = 0;
  for (const seg of keep) {
    offsets.push(acc);
    acc += seg.end - seg.start;
  }

  function segIndex(t) {
    for (let i = 0; i < keep.length; i++) {
      if (keep[i].start - 1e-6 <= t && t <= keep[i].end + 1e-6) return i;
    }
    return null;
  }

  const remapped = [];
  for (const w of words) {
    const mid = (w.start + w.end) / 2;
    const i = segIndex(mid);
    if (i === null) continue;
    const seg = keep[i];
    const newStart = offsets[i] + Math.max(w.start, seg.start) - seg.start;
    const newEnd = offsets[i] + Math.min(w.end, seg.end) - seg.start;
    if (newEnd > newStart) remapped.push({ text: w.text, start: newStart, end: newEnd });
  }
  if (!remapped.length) return [];

  const groups = [[remapped[0]]];
  for (const w of remapped.slice(1)) {
    const cur = groups[groups.length - 1];
    const curText = cur.map(x => x.text).join(" ");
    const gap = w.start - cur[cur.length - 1].end;
    if (curText.length + 1 + w.text.length > maxChars || gap > maxGap) {
      groups.push([w]);
    } else {
      cur.push(w);
    }
  }
  return groups.map(g => ({
    text: g.map(w => w.text).join(" ").trim(),
    start: g[0].start,
    end: g[g.length - 1].end,
  }));
}
