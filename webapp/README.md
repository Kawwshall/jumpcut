# jumpcut, browser-native

Same cut-planning logic as the Python CLI (`src/jumpcut/analyze.py` +
`captions.py`, ported to `src/pipeline.js`), but whisper transcription and
ffmpeg rendering both run **entirely client-side** via WebAssembly
(`@huggingface/transformers` + `@ffmpeg/wasm`). No backend, no upload — the
video never leaves the visitor's machine. Static site, deployable anywhere
that serves files.

## Dev

```bash
npm install
npm run dev       # → http://127.0.0.1:5173 (or whatever port Vite picks)
```

## Build

```bash
npm run build     # → dist/
npm run preview   # serve the actual production build locally to sanity-check it
```

## Deploy (Cloudflare Pages)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Cloudflare dashboard → Pages → Create a project → Connect to Git → pick this repo.
3. Build settings:
   - **Build command:** `npm run build`
   - **Build output directory:** `webapp/dist`
   - **Root directory:** `webapp`
4. Deploy. Add your custom domain (e.g. `edit.iamagentkay.com`) under the
   project's **Custom domains** tab — if the domain's already on the same
   Cloudflare account, DNS gets configured automatically.

No environment variables, no secrets, no server to manage — it's a static
site. Whisper model weights are fetched from Hugging Face's CDN on first use
and cached by the browser after that.

## Why this needed a build step

Three genuine, successive incompatibilities showed up trying to load
`@ffmpeg/ffmpeg` straight from a CDN via `<script>` tags (no bundler):

1. **Cross-origin `SecurityError`** — browsers refuse to spawn a classic
   `Worker` from an absolute cross-origin script URL.
2. **A broken `@ffmpeg/util` UMD build** — it references a bare `exports`
   identifier without a `typeof` guard, so it throws `ReferenceError:
   exports is not defined` the instant it runs as a plain script (confirmed
   directly, not a guess).
3. **A `require()` call inside the UMD worker chunk** — even after
   blob-ifying the worker script to dodge #1, the chunk itself does a
   Node-style `require('blob:...')` to load the wasm core, which no browser
   `require()` shim understands.

Vite (or any real bundler) resolves `@ffmpeg/ffmpeg`'s worker as a proper
same-origin module, sidestepping all three structurally instead of patching
around each one. `ffmpeg-core.wasm` itself is still fetched from a CDN at
runtime rather than bundled into `dist/` — it's ~32MB, over Cloudflare
Pages' 25MB-per-file limit, and there was never a reason to self-host it
(only the worker script had the cross-origin/module problem).

## Known limitations vs. the Python version

- **English-only, `tiny` model** (`Xenova/whisper-tiny.en`) for now — smaller
  download, faster in-browser inference. Swapping in `base`/multilingual
  models is just changing `WHISPER_MODEL` in `src/app.js`, at the cost of a
  bigger download and slower transcription on the visitor's own hardware.
- **No retake detection.** That feature calls the Anthropic API — an API key
  can't safely live in client-side JS, so it's not available here.
- **Caption overlay cap** (`MAX_OVERLAY_LINES = 60` in `src/app.js`): each
  caption line adds one more `ffmpeg` overlay filter stage; past ~60 lines
  it's slower than it's worth in-browser, so captions are skipped rather than
  grinding for minutes on a long video.
