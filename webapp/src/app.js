// jumpcut, browser-native: everything runs on the visitor's own machine.
// No server, no upload — whisper transcription and ffmpeg rendering both
// happen client-side via WebAssembly. See pipeline.js for the ported
// cut-planning logic (same math as src/jumpcut/analyze.py + captions.py).

// Real npm packages, bundled by Vite — this is the whole point of the build
// step. Raw CDN <script>/ESM-import usage hit three successive, genuine
// incompatibilities (documented in git history if curious: a cross-origin
// Worker SecurityError, a broken @ffmpeg/util UMD bundle, and a Node-style
// require() inside the UMD worker chunk that can't resolve blob: URLs).
// Vite bundles @ffmpeg/ffmpeg's worker as same-origin, sidestepping all of
// that structurally instead of patching around it.
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import { pipeline as hfPipeline } from "@huggingface/transformers";
import { buildPlan, captionLines } from "./pipeline.js";

// ffmpeg-core.wasm is ~32MB — bundling it into our own dist/ (via Vite's
// `?url` imports) exceeds Cloudflare Pages' 25MB-per-file limit. It's fetched
// from a CDN at runtime instead, same as the model weights already are;
// Vite's bundling was only needed to fix the *worker script* (a same-origin
// requirement, not a size one) — the core/wasm binaries never had that
// problem, so there's no reason to self-host them.
const FFMPEG_CORE_BASE = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm";

const WHISPER_MODEL = "Xenova/whisper-tiny.en"; // small download, English only — good for a PoC
const MAX_OVERLAY_LINES = 60; // mirrors render.py's cap; browser CPU is more limited than a server

const el = (id) => document.getElementById(id);
const fileInput = el("fileInput");
const drop = el("drop");
const submitBtn = el("submitBtn");
const statusBox = el("statusBox");
const logBox = el("logBox");
const results = el("results");
const outVideo = el("outVideo");
const dlLink = el("dlLink");
const statsBlock = el("statsBlock");

let chosenFile = null;
let ffmpeg = null;
let transcriber = null;

function log(msg) {
  console.log("[jumpcut]", msg); // also visible via devtools/CDP console, independent of DOM state
  const line = document.createElement("div");
  line.textContent = msg;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function setStatus(msg) {
  statusBox.textContent = msg;
}

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(f) {
  chosenFile = f;
  drop.querySelector(".big").textContent = `✓ ${f.name}`;
  submitBtn.disabled = false;
}

async function getFFmpeg() {
  if (ffmpeg) return ffmpeg;
  log("Loading ffmpeg.wasm (single-threaded core, ~30MB, cached after first load)...");
  ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => {
    // ffmpeg's own stderr is very chatty; only surface it for debugging.
    if (message.includes("Error") || message.includes("error")) log("[ffmpeg] " + message);
  });
  // Vite bundles @ffmpeg/ffmpeg's own worker as a proper same-origin module
  // worker automatically — no classWorkerURL override needed, no blob-URL
  // gymnastics. core/wasm still come from the CDN (see FFMPEG_CORE_BASE
  // above) — that part of the original approach always worked fine; only
  // the worker script needed the bundler.
  await ffmpeg.load({
    coreURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`, "application/wasm"),
  });
  log("ffmpeg.wasm ready.");
  return ffmpeg;
}

async function getTranscriber() {
  if (transcriber) return transcriber;
  log(`Loading Whisper (${WHISPER_MODEL}) — first run downloads the model, then it's cached.`);
  transcriber = await hfPipeline("automatic-speech-recognition", WHISPER_MODEL, {
    // The default (quantized int8) export of this model has a broken node
    // in onnxruntime-web — "Missing required scale" on a DequantizeLinear
    // node, confirmed directly. fp32 is a larger download but is the
    // unquantized artifact, sidestepping that broken quantization path.
    dtype: "fp32",
    progress_callback: (p) => {
      if (p.status === "progress") {
        setStatus(`Downloading model: ${p.file} (${Math.round(p.progress || 0)}%)`);
      }
    },
  });
  log("Whisper ready.");
  return transcriber;
}

async function extractAudioFloat32(ff, inputName) {
  await ff.exec(["-i", inputName, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "audio.wav"]);
  const wavBytes = await ff.readFile("audio.wav");
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const arrayBuf = wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength);
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuf);
  return audioBuffer.getChannelData(0);
}

async function probeDuration(ff, inputName) {
  // ffmpeg.wasm has no ffprobe; read duration out of -i's own stderr banner.
  let duration = 0;
  const onLog = ({ message }) => {
    const m = message.match(/Duration:\s*(\d+):(\d+):(\d+\.\d+)/);
    if (m) duration = (+m[1]) * 3600 + (+m[2]) * 60 + parseFloat(m[3]);
  };
  ff.on("log", onLog);
  try {
    await ff.exec(["-i", inputName]);
  } catch {
    // ffmpeg exits non-zero with no output file specified — expected, we
    // only wanted the stderr banner.
  }
  ff.off("log", onLog);
  return duration;
}

function renderCaptionPNG(text, fontSizePx) {
  const canvas = document.createElement("canvas");
  let ctx = canvas.getContext("2d");
  ctx.font = `bold ${fontSizePx}px Arial, sans-serif`;
  const metrics = ctx.measureText(text);
  const pad = Math.round(fontSizePx * 0.4);
  canvas.width = Math.ceil(metrics.width) + pad * 2;
  canvas.height = Math.ceil(fontSizePx * 1.3) + pad * 2;

  // Resizing the canvas resets its context state, so re-acquire + re-set font.
  ctx = canvas.getContext("2d");
  ctx.font = `bold ${fontSizePx}px Arial, sans-serif`;
  ctx.textBaseline = "top";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(3, Math.round(fontSizePx / 7));
  ctx.strokeStyle = "black";
  ctx.fillStyle = "white";
  ctx.strokeText(text, pad, pad);
  ctx.fillText(text, pad, pad);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      blob.arrayBuffer().then((buf) => resolve(new Uint8Array(buf)));
    }, "image/png");
  });
}

async function runPipeline(file, opts) {
  const ff = await getFFmpeg();

  setStatus("Loading video into ffmpeg...");
  const inputName = "input" + (file.name.match(/\.\w+$/)?.[0] || ".mp4");
  await ff.writeFile(inputName, await fetchFile(file));

  setStatus("Reading duration...");
  const duration = await probeDuration(ff, inputName);
  log(`Duration: ${duration.toFixed(1)}s`);

  setStatus("Extracting audio for transcription...");
  const audioData = await extractAudioFloat32(ff, inputName);

  setStatus("Transcribing (Whisper, in-browser — this is the slow part)...");
  const asr = await getTranscriber();
  const result = await asr(audioData, {
    return_timestamps: "word",
    chunk_length_s: 30,
    stride_length_s: 5,
  });

  const words = (result.chunks || [])
    .filter((c) => c.timestamp && c.timestamp[0] != null && c.timestamp[1] != null)
    .map((c) => ({ text: c.text.trim(), start: c.timestamp[0], end: c.timestamp[1] }));
  log(`Transcribed ${words.length} words.`);
  if (!words.length) throw new Error("No words transcribed — try a clip with clearer speech.");

  setStatus("Planning cuts...");
  const plan = buildPlan(words, duration || words[words.length - 1].end, opts);
  log(`Plan: ${plan.cuts.length} cuts, ${plan.originalDuration.toFixed(1)}s -> ${plan.finalDuration.toFixed(1)}s`);
  if (!plan.keep.length) throw new Error("Nothing left after cuts — loosen the settings.");

  setStatus("Rendering...");
  const filters = [];
  plan.keep.forEach((seg, i) => {
    filters.push(`[0:v]trim=start=${seg.start.toFixed(3)}:end=${seg.end.toFixed(3)},setpts=PTS-STARTPTS[v${i}]`);
    filters.push(`[0:a]atrim=start=${seg.start.toFixed(3)}:end=${seg.end.toFixed(3)},asetpts=PTS-STARTPTS[a${i}]`);
  });
  const concatInputs = plan.keep.map((_, i) => `[v${i}][a${i}]`).join("");
  filters.push(`${concatInputs}concat=n=${plan.keep.length}:v=1:a=1[vc][ac]`);

  let videoLabel = "[vc]";
  const extraArgs = [];

  if (opts.captions) {
    const lines = captionLines(words, plan.keep);
    if (lines.length && lines.length <= MAX_OVERLAY_LINES) {
      log(`Burning in ${lines.length} caption lines...`);
      const fontSize = 28; // fixed for the PoC; could scale with output height
      const margin = 40;
      for (let i = 0; i < lines.length; i++) {
        const png = await renderCaptionPNG(lines[i].text, fontSize);
        await ff.writeFile(`cap_${i}.png`, png);
        extraArgs.push("-i", `cap_${i}.png`);
      }
      lines.forEach((line, i) => {
        const idx = i + 1; // input 0 is the source video
        const outLabel = `[ov${i}]`;
        filters.push(
          `${videoLabel}[${idx}:v]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h-${margin}:` +
          `enable='between(t,${line.start.toFixed(3)},${line.end.toFixed(3)})'${outLabel}`
        );
        videoLabel = outLabel;
      });
    } else if (lines.length) {
      log(`${lines.length} caption lines is too many for in-browser overlay (cap ${MAX_OVERLAY_LINES}) — skipping captions.`);
    }
  }

  const args = ["-i", inputName, ...extraArgs, "-filter_complex", filters.join(";"),
    "-map", videoLabel, "-map", "[ac]",
    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
    "-c:a", "aac", "-b:a", "160k",
    "output.mp4"];
  await ff.exec(args);

  setStatus("Done!");
  const data = await ff.readFile("output.mp4");
  const blob = new Blob([data.buffer], { type: "video/mp4" });
  return { blob, plan };
}

submitBtn.addEventListener("click", async () => {
  if (!chosenFile) return;
  submitBtn.disabled = true;
  results.style.display = "none";
  logBox.innerHTML = "";
  logBox.style.display = "block";

  const opts = {
    silence: el("optSilence").checked,
    fillers: el("optFillers").checked,
    aggressiveFillers: el("optAggressive").checked,
    captions: el("optCaptions").checked,
    minGap: parseFloat(el("optMinGap").value),
  };

  try {
    const { blob, plan } = await runPipeline(chosenFile, opts);
    const url = URL.createObjectURL(blob);
    outVideo.src = url;
    dlLink.href = url;
    dlLink.download = "edited.mp4";
    statsBlock.textContent =
      `${plan.originalDuration.toFixed(0)}s -> ${plan.finalDuration.toFixed(0)}s ` +
      `(-${plan.removedDuration.toFixed(0)}s, ${((plan.removedDuration / plan.originalDuration) * 100).toFixed(0)}% shorter)`;
    results.style.display = "block";
    setStatus("Done — see below.");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
    log("ERROR: " + (e.stack || e.message));
  } finally {
    submitBtn.disabled = false;
  }
});

// Test-only hook: lets automated/headless testing drive the pipeline without
// a real OS file picker. Harmless to leave — this is a client-side app with
// no privileged access to expose.
window.__jumpcutRunPipeline = runPipeline;

window.addEventListener("error", (e) => console.error("[jumpcut] window.onerror:", e.message, e.error?.stack));
window.addEventListener("unhandledrejection", (e) => console.error("[jumpcut] unhandledrejection:", e.reason?.stack || e.reason));
console.log("[jumpcut] app.js module loaded at", Date.now());
