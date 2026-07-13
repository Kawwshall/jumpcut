# Deploying jumpcut

## What you get today

One Fly.io machine running the exact same app you've been testing locally
(`jumpcut-web`), with:

- HTTP Basic Auth gating every route (including video downloads) once you set
  `JUMPCUT_AUTH_USER`/`JUMPCUT_AUTH_PASSWORD` — unset locally, required in prod.
- A persistent volume at `/data` so job files survive a machine restart
  (though not a full redeploy — see **Known limits** below).
- `ffmpeg` from Debian's package (has libass — captions burn in a single fast
  pass, no `ffmpeg-full` workaround needed like on macOS Homebrew).
- The `base` whisper model baked into the image at build time (no cold
  download on first request).

This is intentionally the *simple* version — good for a handful of people you
know, not a public product. See **When to upgrade** for what changes and why.

## First deploy

```bash
# 1. Install the CLI (if you don't have it)
curl -L https://fly.io/install.sh | sh

# 2. Log in (opens a browser)
fly auth login

# 3. From the repo root — this reads fly.toml. It'll ask to confirm the app
#    name/region; say yes, don't let it overwrite fly.toml.
fly launch --no-deploy

# 4. Create the persistent volume (name must match fly.toml's [[mounts]])
fly volumes create jumpcut_data --size 10 --region sjc

# 5. Set secrets — these never touch the repo or fly.toml
fly secrets set JUMPCUT_AUTH_USER=yourteam JUMPCUT_AUTH_PASSWORD="pick-something-real"
# optional, only if you want LLM retake detection:
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

# 6. Ship it
fly deploy
```

You'll get a URL like `https://jumpcut.fly.dev`. Log in with the
`JUMPCUT_AUTH_USER`/`PASSWORD` you set — the browser will prompt for HTTP
Basic Auth credentials the first time.

## Known limits (why this is "beta," not "production")

- **Job history resets on redeploy.** `JOBS` is an in-memory Python dict.
  Redeploying restarts the process, so in-flight/completed job records
  disappear (the actual video *files* on the volume survive, but the app
  won't know about them anymore). Fine for "upload, wait, download" usage;
  not fine if people expect to come back next week and see old jobs.
- **One machine's worth of CPU, shared.** If two people kick off big
  transcriptions at the same second, they compete for the same 2 vCPUs and
  both get slower. Whisper + ffmpeg are the only CPU-heavy parts — see
  `README.md`'s "How it works" for where.
- **No per-user isolation.** Everyone behind the shared password sees the
  same job list conceptually (jobs are looked up by opaque ID, so in practice
  people only see their own if they don't share links — but there's no real
  user account model).

None of these matter until you actually have concurrent beta users hitting
it, which is exactly why they were deferred rather than built speculatively.

## When to upgrade (and to what)

If/when the above limits actually bite:

1. **Job persistence + concurrency** → move the `JOBS` dict to Postgres (Fly
   Postgres or Neon's free tier are both fine at this scale). Turns "job
   history resets on redeploy" into a solved problem, and lets you add a
   proper job queue (SKIP LOCKED pattern, or Redis+RQ) so jobs don't just run
   in a background thread on the same machine as the API.
2. **Storage at scale** → move uploaded/rendered video off the Fly volume and
   onto Cloudflare R2 (S3-compatible API, zero egress fees — matters once
   people are repeatedly downloading finished videos). Presigned upload/
   download URLs mean the app server never proxies the actual video bytes.
3. **Real compute isolation** → split into an `api` process group (small,
   always-on, just enqueues jobs) and a `worker` process group (bigger,
   autostart/autostop so you only pay while a job is actually processing).
   Fly's `fly.toml` supports multiple `[[vm]]`/process-group blocks for
   exactly this.
4. **Real auth** → swap the shared HTTP Basic Auth password for per-user
   accounts (magic-link email is the least-friction option that avoids
   storing passwords at all).

Each of these is a genuine, scoped piece of work — don't build all four
because you *might* need them; build the one you're actually blocked on.
