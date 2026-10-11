# Open review findings (2026-10-10)

From a three-part code review before the first push of local mode. Not yet fixed.

## Medium
- **Run slot race** (`src/evolora/web/app.py` `start_run`/`_stream_run`): `_active_runs` is
  incremented when streaming starts, so two quick POSTs can both pass the check (two trainings
  on one GPU locally; bypasses `WEB_MAX_CONCURRENT_RUNS` hosted). Increment in `start_run`,
  release once (idempotent) from the stream's `finally` and a `BackgroundTask`.
- **Chat doesn't hold the GPU** (`/api/chat`): count chat in the same slot as runs.
- **Hosted approval/cancel** (`/api/runs/{id}/approval|cancel`): add `_require_local()`;
  hosted runs should cancel only by client disconnect. Consider hiding goals in hosted
  `/api/runs`.
- **Orphaned children on hard kill** (`local_backend.py`, `supervisor.py`): closing the agent
  window kills uvicorn without cleanup; put children in a Windows Job Object
  (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
- **Installer update while running** (`EvoLoRA-Setup.cmd`): `Wait-Process` after
  `Stop-Process`, rename `app` -> `app.old`, move new tree in, delete old best-effort.
- **Launcher with non-ASCII user names**: write `%LOCALAPPDATA%`/`%USERPROFILE%` into
  `EvoLoRA.cmd` instead of expanded paths (ASCII encoding turns them into `?`).
- **Startup failures that crash instead of falling back** (`LocalResources.acquire`): catch
  `OSError`/`subprocess` errors from `_free_port`/`Popen` and return a warning.
- **Frontend startup race** (`app.js`): hosted `loadStatus()` can land after `useAgent()`;
  await `initConnection()` first.
- **Revoked pairing leaves the site stuck** (`app.js`): if `hello.paired` is false or an
  API call returns 401, clear the token and show Connect.
- **Pair page quick-click** (`pair.js`): enable Allow only after ~1 s visible+focused; send
  `X-Frame-Options: DENY` / `frame-ancestors 'none'` from the agent.

## Low
- Re-show the approval prompt if answering fails (`answerGate`).
- Middleware: decode headers as latin-1; return JSON 500 with CORS on unhandled errors.
- Remove `"testserver"` from the agent's allowed hosts outside tests.
- Rate limiter: prefer `x-vercel-forwarded-for` / `x-real-ip`.
- Guard `json.loads` on worker event lines; set `PYTHONUTF8=1` for the worker.
- Strip outer quotes from `LOCAL_LLM_EXTRA_ARGS` tokens on Windows.
- GGUF resume: verify final size/ETag before renaming `.part`.
- llama.cpp extraction: extract to a temp dir, then rename.
- Installer: pass its own path via an environment variable (apostrophes in paths).
