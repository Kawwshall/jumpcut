import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    // @ffmpeg/ffmpeg spawns its own Worker and loads wasm dynamically at
    // runtime — Vite's dependency pre-bundling step breaks that if it tries
    // to pre-optimize these packages, so they're excluded here (standard
    // recommendation from ffmpeg.wasm's own Vite integration notes).
    exclude: ["@ffmpeg/ffmpeg", "@ffmpeg/util", "@ffmpeg/core"],
  },
});
