# jumpcut

AI auto-editor for **Loom videos**. Point it at one or more files or Loom share
URLs and it stitches them together, cuts the dead air, strips filler words,
optionally removes retakes, and burns in captions — all driven off a single
word-level transcript.

```
raw video(s) ──▶ stitch (if >1) ──▶ transcribe (whisper) ──▶ analyze ──▶ render (ffmpeg)
                                        word timeline          cut plan     final .mp4
```

Three ways to use it: a **CLI** for scripting, a local **web GUI** (drag &
drop, live progress, before/after preview), and a **browser-native webapp**
(`webapp/` — whisper + ffmpeg run entirely client-side via WebAssembly, no
server, no upload — see `webapp/README.md`).

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` on PATH.

```bash
uv sync                 # core
uv sync --extra retakes # + LLM retake detection (needs ANTHROPIC_API_KEY)
```

**Captions need libass.** Homebrew's default `ffmpeg` formula often ships
without it, which silently degrades burned-in captions to an unreadable
soft track. Install the full build (safe — it's keg-only, won't replace your
existing `ffmpeg`):

```bash
brew install ffmpeg-full
```

jumpcut auto-detects it (see `src/jumpcut/ffbin.py`) and prefers it over
whatever's on PATH when present.

## Use — CLI

```bash
# local file
uv run jumpcut path/to/loom.mp4

# multiple clips — stitched together in the order given
uv run jumpcut intro.mp4 demo.mp4 outro.mp4 -o full.mp4

# a Loom share URL (downloaded first)
uv run jumpcut https://www.loom.com/share/xxxxxxxx

# see the plan without rendering
uv run jumpcut loom.mp4 --dry-run

# tune it
uv run jumpcut loom.mp4 \
    --min-gap 0.8 \            # only cut pauses longer than 0.8s
    --aggressive-fillers \     # also cut like/so/actually/...
    --retakes \                # LLM removes false starts (needs API key)
    --model small \            # bigger whisper = more accurate, slower
    -o edited.mp4
```

Flags: `--no-silence`, `--no-fillers`, `--no-captions`, `--keep-srt`.

Try it on the bundled sample:

```bash
uv run jumpcut examples/sample_loom.mp4 --dry-run
```

## Use — Web GUI

```bash
uv run jumpcut-web
# → http://127.0.0.1:8756
```

Drag in clip(s) (or paste Loom URLs), reorder them by dragging the queue rows,
toggle which edits to apply, hit **Edit video**. You get a live stage-by-stage
progress bar and a before/after comparison with download links when it's done.
Local-only, single-user, no auth — don't expose it to the network as-is.

## How it works

Everything keys off a **word-level transcript** (faster-whisper, greedy
decode, all CPU cores):

- **silence** — gaps between words longer than `--min-gap` become cuts (plus
  head/tail dead air), keeping a small lead-in so speech isn't clipped.
- **fillers** — words matching a lexicon are cut. Default is conservative
  (`um`, `uh`, ...); `--aggressive-fillers` adds the riskier ones (`like`,
  `so`, `actually`...).
- **retakes** — an LLM scans the sentence list for false starts / repeated
  takes and deletes the bad attempt (opt-in, needs the `retakes` extra).

Cuts are merged and inverted into keep-segments, which ffmpeg trims and
concatenates in one re-encode (`preset=ultrafast` — for talking-head content
the quality difference against slower presets isn't perceptible, but it's
5-10x faster). Captions are remapped onto the output timeline and burned in
via libass (`subtitles` filter) when available, falling back to per-line
image overlays, then to a soft track as a last resort.

**Multiple clips** are stitched before analysis: each is normalized to a
common resolution/fps/audio format, with a short black+silent gap inserted
between them so the silence detector reliably treats the seam as a real cut
(otherwise captions can bridge two unrelated clips into one nonsensical line).

## Status

v1 — Loom cleanup (silence / fillers / retakes / captions) + multi-clip
stitching + local web GUI. Next: long-form → short-form clip generation
(podcasts / talks → vertical clips).
