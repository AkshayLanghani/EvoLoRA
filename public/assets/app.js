// EvoLoRA site: streams real runs from /api/runs and drives the guardrail cards.

const $ = (selector) => document.querySelector(selector);

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch {
    // Non-JSON error body: fall through to the status line.
  }
  return `Request failed (${response.status})`;
}

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const fmt = (n, digits = 3) => (typeof n === "number" ? n.toFixed(digits) : "-");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ---------------------------------------------------------------------------
// Mascot: the TUI's terminal cat, same frames and moods
// ---------------------------------------------------------------------------

const MOODS = {
  idle: [["(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^.^=)", "(=^-^=)"], "purring", false],
  think: [["(=o.o=)?", "(=o.o=) ", "(=O.o=)?", "(=o.O=)?"], "thinking", true],
  run: [["(=^.^=)", "(=^o^=)"], "chasing", true],
  look: [["(=O.O=)", "(=o.o=)", "(=O.O=)", "(=-.-=)"], "watching", true],
  happy: [["(=^w^=)", "(=^o^=)"], "yay!", false],
  sad: [["(=;_;=)", "(=T_T=)"], "mrrp", false],
};

const mascot = {
  mood: "idle",
  pos: 0,
  dir: 1,
  frame: 0,
  track: 14,
  node: $("#mascot"),
  set(mood) {
    this.mood = MOODS[mood] ? mood : "idle";
    this.render();
  },
  tick() {
    this.frame += 1;
    const scampers = MOODS[this.mood][2] && !reducedMotion.matches;
    if (scampers) {
      this.pos += this.dir;
      if (this.pos >= this.track) [this.pos, this.dir] = [this.track, -1];
      else if (this.pos <= 0) [this.pos, this.dir] = [0, 1];
    }
    this.render();
  },
  render() {
    const [frames, caption] = MOODS[this.mood];
    const face = frames[this.frame % frames.length];
    this.node.textContent = `${" ".repeat(this.pos)}${face}`.padEnd(this.track + 10) + caption;
  },
};
mascot.render();
setInterval(() => mascot.tick(), 180);

// ---------------------------------------------------------------------------
// Event presentation
// ---------------------------------------------------------------------------

// event kind -> [state label, mascot mood, log label, tone]
const KINDS = {
  run_started: ["STARTING", "run", "start", "good"],
  status_changed: ["RUNNING", "run", "status", ""],
  eval_set_locked: ["LOCKED", "think", "lock", "info"],
  baseline_complete: ["BASELINE", "look", "baseline", "info"],
  planning_started: ["PLANNING", "think", "plan", ""],
  plan_received: ["PLANNING", "think", "plan ok", ""],
  agent_fallback_used: ["PLANNING", "think", "fallback", "warn"],
  validation_complete: ["VALIDATE", "run", "validate", ""],
  training_started: ["TRAINING", "run", "train", ""],
  training_complete: ["TRAINED", "happy", "trained", "good"],
  eval_started: ["EVALUATE", "look", "eval", ""],
  eval_complete: ["EVAL", "look", "eval", "info"],
  adaptive_complete: ["EVAL", "look", "adaptive", ""],
  judge_started: ["JUDGE", "look", "judge", ""],
  judge_complete: ["JUDGED", "think", "judged", "info"],
  retrain_decision_received: ["DECIDE", "think", "decide", ""],
  iteration_complete: ["ITERATION", "run", "iter", ""],
  best_updated: ["BEST", "happy", "best", "good"],
  stop_condition_met: ["STOP", "idle", "stop", "warn"],
  run_complete: ["DONE", "happy", "done", "good"],
  run_failed: ["FAILED", "sad", "failed", "bad"],
  run_cancelled: ["CANCELLED", "sad", "cancel", "bad"],
  log: [null, null, "log", ""],
};

const STOP_REASONS = {
  target_score: "Target score reached",
  max_iterations: "Max iterations",
  patience: "No further improvement",
  training_failure: "Repeated training failures",
  eval_hash_changed: "Eval set tampered",
  cancelled: "Cancelled",
  backend_unavailable: "Backend unavailable",
  judge_accepted: "Judge accepted the adapter",
  user_declined_retrain: "User declined retraining",
  server_error: "Server error",
};

function describeHyperparams(hp) {
  if (!hp) return "";
  return ` · lr ${hp.learning_rate} · r ${hp.r} · alpha ${hp.lora_alpha} · ${hp.num_epochs} epochs · batch ${hp.batch_size}`;
}

function eventText(event) {
  const data = event.data || {};
  switch (event.kind) {
    case "validation_complete":
      return event.message + describeHyperparams(data.hyperparams);
    case "judge_complete":
      return data.summary ? `${event.message} (${data.source || "judge"}): ${data.summary}` : event.message;
    case "retrain_decision_received":
      return data.reason ? `${event.message}: ${data.reason}` : event.message;
    case "plan_received":
      return data.rationale ? `${event.message}: ${data.rationale}` : event.message;
    default:
      return event.message;
  }
}

// ---------------------------------------------------------------------------
// Run console
// ---------------------------------------------------------------------------

const ui = {
  form: $("#run-form"),
  goal: $("#goal"),
  baseModel: $("#base-model"),
  samples: $("#samples"),
  start: $("#start-btn"),
  cancel: $("#cancel-btn"),
  error: $("#run-error"),
  state: $("#state"),
  iteration: $("#iteration"),
  metaRun: $("#meta-run"),
  metaHash: $("#meta-hash"),
  metaBest: $("#meta-best"),
  log: $("#log"),
  progress: $("#progress"),
  progressBar: $("#progress-bar"),
  progressLabel: $("#progress-label"),
  lossChart: $("#loss-chart"),
  scoreChart: $("#score-chart"),
  targetValue: $("#target-value"),
  summary: $("#summary"),
};

const run = {
  controller: null,
  startedAt: 0,
  maxIterations: 3,
  iteration: 0,
  target: 0.85,
  baseline: null,
  scores: [],
  losses: [],
  drawnBars: 0,
  drawQueued: false,
};

function setState(label, mood) {
  ui.state.textContent = label;
  ui.state.dataset.tone = ["FAILED", "CANCELLED"].includes(label) ? "danger" : label === "READY" ? "idle" : "";
  if (mood) mascot.set(mood);
}

function appendLog(label, message, tone = "") {
  ui.log.querySelector(".log-empty")?.remove();
  const elapsed = (performance.now() - run.startedAt) / 1000;
  const time = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${(elapsed % 60).toFixed(1).padStart(4, "0")}`;
  const row = el(
    "li",
    { className: tone ? `tone-${tone}` : "" },
    el("span", { className: "t", text: time }),
    el("span", { className: "k", text: label }),
    el("span", { className: "m", text: message }),
  );
  const atBottom = ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight < 40;
  ui.log.append(row);
  if (atBottom) ui.log.scrollTop = ui.log.scrollHeight;
}

function queueDraw() {
  if (run.drawQueued) return;
  run.drawQueued = true;
  requestAnimationFrame(() => {
    run.drawQueued = false;
    drawLoss();
    drawScores();
  });
}

const CHART = { w: 320, h: 140, left: 26, right: 6, top: 8, bottom: 18 };

function chartFrame(node, yMax, yTicks) {
  node.replaceChildren();
  const { w, h, left, right, top, bottom } = CHART;
  const y = (v) => top + (1 - v / yMax) * (h - top - bottom);
  for (const tick of yTicks) {
    node.append(
      svg("line", { class: "grid", x1: left, x2: w - right, y1: y(tick), y2: y(tick) }),
      svg("text", { x: left - 4, y: y(tick) + 3, "text-anchor": "end" }, String(tick)),
    );
  }
  return y;
}

function drawLoss() {
  const node = ui.lossChart;
  const peak = Math.max(2, ...run.losses.flatMap((s) => s.points.map((p) => p.loss)));
  const yMax = Math.ceil(peak * 2) / 2;
  const y = chartFrame(node, yMax, [0, yMax / 2, yMax]);
  const { w, left, right } = CHART;
  if (!run.losses.length) {
    node.append(svg("text", { class: "empty", x: w / 2, y: 70, "text-anchor": "middle" }, "Waiting for training"));
    return;
  }
  run.losses.forEach((series, index) => {
    if (!series.points.length) return;
    const x = (step) => left + ((step - 1) / Math.max(1, series.total - 1)) * (w - left - right);
    const d = series.points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join("");
    const isPast = index < run.losses.length - 1;
    node.append(svg("path", { class: `loss-line${isPast ? " is-past" : ""}`, d }));
  });
}

function drawScores() {
  const node = ui.scoreChart;
  const y = chartFrame(node, 1, [0, 0.5, 1]);
  const { w, h, left, right, bottom } = CHART;
  const bars = [];
  if (run.baseline !== null) bars.push({ label: "base", value: run.baseline, baseline: true });
  run.scores.forEach((s) => bars.push({ label: `it ${s.iteration}`, value: s.score }));
  const slots = Math.max(run.maxIterations + 1, bars.length);
  const slot = (w - left - right) / slots;
  bars.forEach((bar, i) => {
    const x = left + i * slot + slot * 0.2;
    const barW = slot * 0.6;
    const top = y(Math.max(0, Math.min(1, bar.value)));
    // The chart is redrawn on every training step; only animate bars that just appeared.
    const fresh = i >= run.drawnBars ? " is-new" : "";
    node.append(
      svg("rect", { class: `bar${bar.baseline ? " is-baseline" : ""}${fresh}`, x, y: top, width: barW, height: Math.max(1, y(0) - top), rx: 3 }),
      svg("text", { class: `bar-value${fresh}`, x: x + barW / 2, y: top - 4, "text-anchor": "middle" }, bar.value.toFixed(2)),
      svg("text", { x: x + barW / 2, y: h - bottom + 12, "text-anchor": "middle" }, bar.label),
    );
  });
  run.drawnBars = bars.length;
  node.append(svg("line", { class: "target", x1: left, x2: w - right, y1: y(run.target), y2: y(run.target) }));
  if (!bars.length) {
    node.append(svg("text", { class: "empty", x: w / 2, y: 70, "text-anchor": "middle" }, "Waiting for a baseline"));
  }
}

function resetConsole() {
  run.startedAt = performance.now();
  run.baseline = null;
  run.scores = [];
  run.losses = [];
  run.drawnBars = 0;
  ui.log.replaceChildren();
  ui.summary.hidden = true;
  ui.summary.replaceChildren();
  ui.progress.hidden = true;
  ui.iteration.textContent = "";
  ui.metaRun.textContent = "-";
  ui.metaHash.textContent = "-";
  ui.metaBest.textContent = "-";
  queueDraw();
}

function renderSummary(data) {
  const items = [
    ["Baseline", fmt(data.baseline_score)],
    ["Best", data.best_iteration ? `${fmt(data.best_score)} (it ${data.best_iteration})` : fmt(data.best_score)],
    ["Stopped on", STOP_REASONS[data.stop_reason] || data.stop_reason || "-"],
    ["Adapter", `adapters/${data.adapter_name}`],
  ];
  ui.summary.replaceChildren(
    ...items.map(([dt, dd]) => el("div", {}, el("dt", { text: dt }), el("dd", { text: dd }))),
  );
  ui.summary.hidden = false;
}

function handleEvent(event) {
  const [state, mood, label, tone] = KINDS[event.kind] || [null, null, event.kind, ""];
  const data = event.data || {};
  if (state) setState(state, mood);

  switch (event.kind) {
    case "run_started":
      ui.metaRun.textContent = event.run_id.slice(0, 8);
      break;
    case "eval_set_locked":
      ui.metaHash.textContent = `${data.hash.slice(0, 16)}…`;
      ui.metaHash.title = data.hash;
      break;
    case "baseline_complete":
      run.baseline = data.score;
      ui.metaBest.textContent = fmt(data.score);
      queueDraw();
      break;
    case "planning_started":
      run.iteration = data.iteration;
      ui.iteration.textContent = `iteration ${data.iteration} of ${run.maxIterations}`;
      break;
    case "training_started":
      run.losses.push({ total: 1, points: [] });
      ui.progress.hidden = false;
      ui.progressBar.value = 0;
      ui.progressLabel.textContent = `Training on ${data.backend || "backend"}`;
      break;
    case "training_progress": {
      const series = run.losses.at(-1);
      if (series && typeof data.loss === "number") {
        series.total = data.total_steps || series.total;
        series.points.push({ step: data.step, loss: data.loss });
        ui.progressBar.value = data.step / series.total;
        ui.progressLabel.textContent = `Step ${data.step}/${series.total} · loss ${data.loss.toFixed(3)}`;
        queueDraw();
      }
      return; // too frequent for the log
    }
    case "training_complete":
      ui.progress.hidden = true;
      break;
    case "eval_complete":
      run.scores.push({ iteration: run.iteration, score: data.score });
      queueDraw();
      break;
    case "best_updated":
      ui.metaBest.textContent = `${fmt(data.score)} (it ${data.iteration})`;
      break;
    case "run_summary":
      renderSummary(data);
      return;
    default:
      break;
  }
  if (event.kind === "status_changed" && /baseline/i.test(event.message)) setState("BASELINE", "look");
  appendLog(label, eventText(event), tone);
}

async function readNdjson(response, onEvent) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function setRunning(running) {
  $(".terminal").classList.toggle("is-running", running);
  ui.start.disabled = running;
  ui.cancel.disabled = !running;
  for (const field of [ui.goal, ui.baseModel, ui.samples]) field.disabled = running;
}

async function startRun(event) {
  event.preventDefault();
  ui.error.hidden = true;
  const goal = ui.goal.value.trim();
  if (!goal) {
    ui.error.textContent = "Describe a goal first.";
    ui.error.hidden = false;
    ui.goal.focus();
    return;
  }

  resetConsole();
  setRunning(true);
  setState("STARTING", "run");
  run.controller = new AbortController();
  let finished = false;

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        goal,
        base_model_id: ui.baseModel.value,
        training_samples: Number(ui.samples.value) || 10,
      }),
      signal: run.controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    await readNdjson(response, (ev) => {
      if (["run_complete", "run_failed", "run_cancelled"].includes(ev.kind)) finished = true;
      handleEvent(ev);
    });
    if (!finished) {
      setState("FAILED", "sad");
      appendLog("failed", "The stream ended before the run finished (the server may have hit its time limit).", "bad");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      setState("CANCELLED", "sad");
      appendLog("cancel", "Run cancelled from the browser", "bad");
    } else {
      setState("READY", "sad");
      ui.error.textContent = error.message;
      ui.error.hidden = false;
      appendLog("error", error.message, "bad");
    }
  } finally {
    ui.progress.hidden = true;
    run.controller = null;
    setRunning(false);
    loadHistory();
  }
}

ui.form.addEventListener("submit", startRun);
ui.cancel.addEventListener("click", () => run.controller?.abort());

const presets = document.querySelectorAll(".presets .chip");
function syncPresets() {
  for (const chip of presets) chip.setAttribute("aria-pressed", String(chip.dataset.goal === ui.goal.value.trim()));
}
for (const chip of presets) {
  chip.addEventListener("click", () => {
    ui.goal.value = chip.dataset.goal;
    syncPresets();
  });
}
ui.goal.addEventListener("input", syncPresets);
syncPresets();

// ---------------------------------------------------------------------------
// Service status and history
// ---------------------------------------------------------------------------

function service(name, value, live) {
  return el("li", { className: `service ${live ? "is-live" : "is-fallback"}` }, `${name} `, el("strong", { text: value }));
}

async function loadStatus() {
  const list = $("#services");
  try {
    const s = await api("/api/status");
    const remote = s.training_backend !== "mock";
    list.replaceChildren(
      service("Planner", s.planner === "minimax" ? s.planner_model : "heuristic fallback", s.planner === "minimax"),
      service("Judge", s.judge === "digitalocean" ? s.judge_model : "heuristic fallback", s.judge === "digitalocean"),
      service("Training", remote ? `${s.training_backend} GPU` : "mock backend", remote),
      service("History", s.persistence === "mongodb" ? "MongoDB" : "in-memory", s.persistence === "mongodb"),
    );
    ui.baseModel.replaceChildren(...s.base_models.map((m) => el("option", { value: m.id, text: m.label })));
    ui.baseModel.value = s.default_base_model;
    ui.samples.max = String(s.limits.max_training_samples);
    run.maxIterations = s.loop.max_iterations;
    run.target = s.loop.target_score;
    ui.targetValue.textContent = s.loop.target_score.toFixed(2);
    $("#history-note").textContent =
      s.persistence === "mongodb"
        ? "Stored in MongoDB."
        : "Without MONGODB_URI, history lives in server memory and resets when the function restarts.";
  } catch (error) {
    list.replaceChildren(el("li", { className: "service", text: `API unreachable: ${error.message}` }));
  }
  queueDraw();
}

async function loadHistory() {
  const body = $("#history-body");
  try {
    const { runs } = await api("/api/runs?limit=10");
    if (!runs.length) return;
    body.replaceChildren(
      ...runs.map((r) =>
        el(
          "tr",
          {},
          el("td", { text: r.goal || "(demo task)" }),
          el("td", { text: fmt(r.baseline_score) }),
          el("td", { text: fmt(r.best_score) }),
          el("td", { text: STOP_REASONS[r.stop_reason] || r.stop_reason || r.status }),
          el("td", {}, el("code", { text: r.adapter_name })),
        ),
      ),
    );
  } catch {
    // History is optional; keep the placeholder row.
  }
}

// ---------------------------------------------------------------------------
// Guardrail cards
// ---------------------------------------------------------------------------

function verdict(ok, text, detail) {
  const node = el("div", { className: `verdict${ok ? "" : " bad"}` }, text);
  if (detail) node.append(el("code", { text: detail }));
  return node;
}

function showError(target, error) {
  target.replaceChildren(verdict(false, error.message));
}

// Hyperparameter snapping
const snapForm = $("#snap-form");
const snapResult = $("#snap-result");
snapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const args = {};
  for (const input of snapForm.querySelectorAll("input")) {
    if (input.value !== "") args[input.name] = Number(input.value);
  }
  try {
    const { snapped } = await api("/api/tools/snap", { method: "POST", body: { args } });
    const mult = snapped.lora_alpha / snapped.r;
    const rows = [
      ["learning_rate", args.learning_rate, snapped.learning_rate],
      ["lora_rank_r", args.lora_rank_r, snapped.r],
      ["alpha multiplier", args.lora_alpha_multiplier, mult],
      ["num_train_epochs", args.num_train_epochs, snapped.num_epochs],
      ["batch_size", args.per_device_train_batch_size, snapped.batch_size],
    ];
    const table = el(
      "table",
      {},
      el("thead", {}, el("tr", {}, el("th", { text: "Field" }), el("th", { text: "Proposed" }), el("th", { text: "Trained with" }))),
      el(
        "tbody",
        {},
        ...rows.map(([name, proposed, used]) =>
          el(
            "tr",
            {},
            el("td", {}, el("code", { text: name })),
            el("td", { text: proposed === undefined ? "(missing)" : String(proposed) }),
            el("td", { className: proposed === used ? "" : "changed", text: String(used) }),
          ),
        ),
      ),
    );
    snapResult.replaceChildren(table, el("p", { className: "note", text: `lora_alpha = ${snapped.r} × ${mult} = ${snapped.lora_alpha}` }));
  } catch (error) {
    showError(snapResult, error);
  }
});

// Locked eval set, objective scorer
const lockEditor = $("#lock-editor");
const lockResult = $("#lock-result");
const scoreSample = $("#score-sample");
const scorePrompt = $("#score-prompt");
const scoreResult = $("#score-result");
let evalSet = null;

async function verifyLock() {
  try {
    const result = await api("/api/tools/verify-evals", { method: "POST", body: { samples_json: lockEditor.value } });
    lockResult.replaceChildren(
      result.match
        ? verdict(true, "Hash matches. The run would continue.", result.computed_hash)
        : verdict(false, "Hash mismatch. The run halts with eval_hash_changed.", result.computed_hash),
    );
  } catch (error) {
    showError(lockResult, error);
  }
}

async function loadEvalSet() {
  try {
    evalSet = await api("/api/tools/eval-set");
    lockEditor.value = evalSet.samples_json;
    lockResult.replaceChildren(verdict(true, "Locked hash", evalSet.hash));
    scoreSample.replaceChildren(...evalSet.samples.map((s) => el("option", { value: s.sample_id, text: s.sample_id })));
    scorePrompt.textContent = evalSet.samples[0].prompt;
  } catch (error) {
    showError(lockResult, error);
  }
}

$("#lock-verify").addEventListener("click", verifyLock);
$("#lock-reset").addEventListener("click", () => {
  if (!evalSet) return;
  lockEditor.value = evalSet.samples_json;
  verifyLock();
});
$("#lock-tamper").addEventListener("click", () => {
  lockEditor.value = lockEditor.value.replace('"top_customer_total": 550.0', '"top_customer_total": 450.0');
  verifyLock();
});

scoreSample.addEventListener("change", () => {
  const sample = evalSet?.samples.find((s) => s.sample_id === scoreSample.value);
  if (sample) scorePrompt.textContent = sample.prompt;
});

$("#score-run").addEventListener("click", async () => {
  try {
    const result = await api("/api/tools/score", {
      method: "POST",
      body: { sample_id: scoreSample.value, response: $("#score-response").value },
    });
    const checks = result.details?.checks;
    const nodes = [
      verdict(result.passed, `Score ${fmt(result.score, 2)}${result.passed ? ", all checks pass" : ""}`, result.details?.error),
    ];
    if (checks) {
      nodes.push(
        el(
          "ul",
          { className: "checks" },
          ...Object.entries(checks).map(([name, ok]) =>
            el("li", {}, el("span", { className: ok ? "ok" : "no", text: ok ? "pass" : "fail" }), name),
          ),
        ),
      );
    }
    scoreResult.replaceChildren(...nodes);
  } catch (error) {
    showError(scoreResult, error);
  }
});

// Adapter naming
const nameGoal = $("#name-goal");
const nameResult = $("#name-result");
const nameRunId = crypto.getRandomValues(new Uint8Array(4)).reduce((s, b) => s + b.toString(16).padStart(2, "0"), "");
let nameTimer = 0;

async function updateName() {
  try {
    const { name } = await api("/api/tools/adapter-name", { method: "POST", body: { goal: nameGoal.value, run_id: nameRunId } });
    nameResult.textContent = name;
  } catch (error) {
    nameResult.textContent = error.message;
  }
}
nameGoal.addEventListener("input", () => {
  clearTimeout(nameTimer);
  nameTimer = setTimeout(updateName, 250);
});

// ---------------------------------------------------------------------------
// Copy buttons
// ---------------------------------------------------------------------------

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Press Ctrl+C";
    }
    setTimeout(() => (button.textContent = "Copy"), 1600);
  });
}

// ---------------------------------------------------------------------------
// Motion: staggered scroll reveals and the hero loop
// ---------------------------------------------------------------------------

const REVEAL_GROUPS = [
  ".hero .eyebrow, .hero h1, .hero .lede, .hero .cta-row, .hero .loop-visual",
  ".facts > div",
  ".section-head",
  ".services, .runner > *, .history",
  ".cards > .card",
  ".lanes",
  ".stack-grid > *, .stack-line",
  ".commands > .command, #local .note",
];

function setUpReveals() {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const node = entry.target;
        // Drop the reveal styles once shown so hover transitions apply again.
        node.addEventListener(
          "transitionend",
          () => {
            node.classList.remove("reveal", "is-visible");
            node.style.removeProperty("--delay");
          },
          { once: true },
        );
        node.classList.add("is-visible");
        observer.unobserve(node);
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
  );
  for (const group of REVEAL_GROUPS) {
    document.querySelectorAll(group).forEach((node, index) => {
      node.classList.add("reveal");
      node.style.setProperty("--delay", `${Math.min(index, 6) * 90}ms`);
      observer.observe(node);
    });
  }
}

const loopSvg = $("#loop-svg");
function syncLoopMotion() {
  if (reducedMotion.matches) loopSvg.pauseAnimations();
  else loopSvg.unpauseAnimations();
}
reducedMotion.addEventListener("change", syncLoopMotion);
syncLoopMotion();

if ("IntersectionObserver" in window && !reducedMotion.matches) setUpReveals();

// ---------------------------------------------------------------------------

setState("READY", "idle");
loadStatus();
loadHistory();
loadEvalSet();
updateName();
