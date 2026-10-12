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

// Where API calls go: this site's own backend, or the EvoLoRA agent on the visitor's
// computer once they've connected it (Bearer token from the local pairing page).
const AGENT = "http://127.0.0.1:8765";
const TOKEN_KEY = "evolora-agent-token";
const conn = { base: "", token: "" };
const servedByAgent = ["127.0.0.1", "localhost"].includes(location.hostname);

function apiUrl(path) {
  return conn.base + path;
}

function apiHeaders(json) {
  const headers = json ? { "content-type": "application/json" } : {};
  if (conn.token) headers.authorization = `Bearer ${conn.token}`;
  return headers;
}

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(apiUrl(path), {
    method,
    headers: apiHeaders(Boolean(body)),
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
  eval_approval_required: ["APPROVE", "think", "approve", "warn"],
  user_approval_required: ["APPROVE", "think", "approve", "warn"],
  user_approval_received: ["APPROVED", "run", "answer", ""],
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
  autonomous: $("#autonomous"),
  gate: $("#gate"),
  gateText: $("#gate-text"),
  gateEvals: $("#gate-evals"),
};

const run = {
  controller: null,
  runId: "",
  mode: "hosted",
  startedAt: 0,
  maxIterations: 3,
  iteration: 0,
  target: 0.85,
  baseline: null,
  scores: [],
  losses: [],
  drawnBars: 0,
  drawQueued: false,
  ready: !servedByAgent,
  running: false,
  followLog: true,
  blocker: "setup",
};

function setState(label, mood) {
  ui.state.textContent = label;
  ui.state.dataset.tone = ["FAILED", "CANCELLED"].includes(label) ? "danger" : label === "READY" ? "idle" : "";
  if (mood) mascot.set(mood);
  const stage = { READY: -1, STARTING: 0, LOCKED: 0, APPROVE: 0, BASELINE: 1,
    PLANNING: 2, VALIDATE: 2, TRAINING: 3, TRAINED: 3, EVALUATE: 4,
    EVAL: 4, JUDGE: 5, JUDGED: 5, DECIDE: 5, ITERATION: 5, BEST: 5, DONE: 6 }[label];
  if (stage !== undefined) {
    for (const item of document.querySelectorAll("[data-stage]")) {
      const index = Number(item.dataset.stage);
      item.classList.toggle("is-done", index < stage);
      if (index === stage) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    }
  }
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
  if (atBottom && run.followLog) ui.log.scrollTop = ui.log.scrollHeight;
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
  $("#meta-improvement").textContent = "-";
  $("#elapsed").textContent = "00:00";
  $("#eval-diagnostics").hidden = true;
  latestResourceSettings = {};
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
    ["Adapter", data.adapter_name ? `adapters/${data.adapter_name}` : "Base model retained"],
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
      run.runId = event.run_id;
      ui.metaRun.textContent = event.run_id.slice(0, 8);
      break;
    case "eval_approval_required":
    case "user_approval_required":
      showGate(event.message, data.evals);
      break;
    case "user_approval_received":
    case "run_complete":
    case "run_failed":
    case "run_cancelled":
      hideGate();
      break;
    case "eval_set_locked":
      ui.metaHash.textContent = `${data.hash.slice(0, 16)}…`;
      ui.metaHash.title = data.hash;
      break;
    case "baseline_complete":
      run.baseline = data.score;
      ui.metaBest.textContent = fmt(data.score);
      $("#meta-improvement").textContent = "+0.000";
      renderDiagnostics(data.sample_results);
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
    case "resource_status":
      renderResources(data.resources, data.settings);
      return;
    case "training_complete":
      ui.progress.hidden = true;
      break;
    case "eval_complete":
      run.scores.push({ iteration: run.iteration, score: data.score });
      queueDraw();
      renderDiagnostics(data.sample_results);
      break;
    case "best_updated":
      ui.metaBest.textContent = `${fmt(data.score)} (it ${data.iteration})`;
      $("#meta-improvement").textContent = `+${fmt(data.score - (run.baseline || 0))}`;
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

// ---------------------------------------------------------------------------
// Approval prompts (local agent only)
// ---------------------------------------------------------------------------

function showGate(message, evals) {
  ui.gateText.textContent = message;
  ui.gateEvals.replaceChildren(
    ...(evals || []).slice(0, 20).map((e) => el("li", { text: `${e.input}  →  ${e.expected}` })),
  );
  ui.gate.hidden = false;
  $("#gate-yes").focus();
}

function hideGate() {
  ui.gate.hidden = true;
}

async function answerGate(approved) {
  if (!run.runId) return;
  const buttons = [$("#gate-yes"), $("#gate-no")];
  buttons.forEach((button) => { button.disabled = true; });
  try {
    await api(`/api/runs/${encodeURIComponent(run.runId)}/approval`, { method: "POST", body: { approved } });
    hideGate();
  } catch (error) {
    appendLog("error", `Could not send the answer: ${error.message}`, "bad");
    if (run.running) $("#gate-yes").focus();
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

$("#gate-yes").addEventListener("click", () => answerGate(true));
$("#gate-no").addEventListener("click", () => answerGate(false));

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
  run.running = running;
  $(".terminal").classList.toggle("is-running", running);
  ui.start.disabled = running || !run.ready;
  ui.start.textContent = running ? "Run in progress…" : run.ready ? "Start run" : run.blocker === "offline" ? "Agent offline" : "Finish setup first";
  ui.cancel.disabled = !running;
  for (const field of [ui.goal, ui.baseModel, ui.samples, ui.autonomous]) field.disabled = running;
  for (const field of [$("#chat-send"), $("#chat-model"), $("#chat-prompt")]) field.disabled = running;
}

async function startRun(event) {
  event.preventDefault();
  ui.error.hidden = true;
  if (!run.ready) {
    ui.error.textContent = "Finish installing the local planner and training tools before starting.";
    ui.error.hidden = false;
    $("#setup")?.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  const goal = ui.goal.value.trim();
  const count = Number(ui.samples.value);
  if (!Number.isInteger(count) || count < Number(ui.samples.min) || count > Number(ui.samples.max)) {
    ui.error.textContent = `Choose a whole number of examples between ${ui.samples.min} and ${ui.samples.max}.`;
    ui.error.hidden = false;
    ui.samples.focus();
    return;
  }
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
    const response = await fetch(apiUrl("/api/runs"), {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({
        goal,
        base_model_id: ui.baseModel.value,
        training_samples: count,
        autonomous: run.mode !== "local" || ui.autonomous.checked,
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
      setState("FAILED", "sad");
      ui.error.textContent = error.message;
      ui.error.hidden = false;
      appendLog("error", error.message, "bad");
    }
  } finally {
    ui.progress.hidden = true;
    hideGate();
    run.controller = null;
    run.runId = "";
    setRunning(false);
    loadHistory();
    loadAdapters();
  }
}

ui.form.addEventListener("submit", startRun);
ui.cancel.addEventListener("click", async () => {
  const controller = run.controller;
  const runId = run.runId;
  if (!controller || !run.running) return;
  ui.cancel.disabled = true;
  try {
    if (runId) await api(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
    if (run.controller === controller && run.running) controller.abort();
  } catch (error) {
    if (run.controller !== controller || !run.running) return;
    ui.cancel.disabled = false;
    ui.error.textContent = `Could not confirm cancellation: ${error.message}`;
    ui.error.hidden = false;
  }
});

const presets = document.querySelectorAll(".presets .chip");
function syncPresets() {
  for (const chip of presets) chip.setAttribute("aria-pressed", String(chip.dataset.goal === ui.goal.value.trim()));
  $("#goal-count").textContent = `${ui.goal.value.length} / ${ui.goal.maxLength}`;
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
    const blocker = s.mode === "local" && (s.local_llm_issue || s.local_training_issue);
    run.ready = !blocker;
    run.blocker = "setup";
    $("#readiness").dataset.ready = String(run.ready);
    $("#readiness").textContent = blocker ? "Setup needed" : s.training_backend === "mock" ? "Demo mode" : "Ready to train";
    setRunning(run.running);
    const local = s.mode === "local";
    const remote = s.training_backend !== "mock";
    const llmLabel = (kind, model) => (kind === "heuristic" ? "heuristic fallback" : local ? `local · ${model}` : model);
    const history = { sqlite: "SQLite", mongodb: "MongoDB", memory: "in-memory" }[s.persistence];
    list.replaceChildren(
      service("Planner", llmLabel(s.planner, s.planner_model), s.planner !== "heuristic"),
      service("Judge", llmLabel(s.judge, s.judge_model), s.judge !== "heuristic"),
      service("Training", remote ? `${s.training_backend} GPU` : "mock backend", remote),
      service("History", history, s.persistence !== "memory"),
    );
    run.mode = s.mode;
    baseModels = s.base_models;
    $("#mode-badge").hidden = !local;
    setChatLocked(!local);
    $("#cloud-models-note").hidden = local;
    if (local) loadAdapters();
    $("#autonomous-field").hidden = !s.approvals;
    if (local) {
      $("#mode-note").textContent = blocker
        ? "Install the local AI components above to enable real training."
        : "Running on this computer. Turn off Autonomous to approve the eval set and decide whether to keep training.";
      $("#run-title").textContent = "Run the loop on this computer";
    }
    const selectedModel = ui.baseModel.value;
    ui.baseModel.replaceChildren(...s.base_models.map((m) => el("option", { value: m.id, text: m.label })));
    ui.baseModel.value = s.base_models.some((m) => m.id === selectedModel) ? selectedModel : s.default_base_model;
    ui.samples.max = String(s.limits.max_training_samples);
    syncPresets();
    run.maxIterations = s.loop.max_iterations;
    run.target = s.loop.target_score;
    ui.targetValue.textContent = s.loop.target_score.toFixed(2);
    $("#history-note").textContent = {
      sqlite: "Stored in a local SQLite file.",
      mongodb: "Stored in MongoDB.",
      memory: "Without MONGODB_URI, history lives in server memory and resets when the function restarts.",
    }[s.persistence];
  } catch (error) {
    run.ready = false;
    run.blocker = "offline";
    $("#readiness").dataset.ready = "false";
    $("#readiness").textContent = "Agent offline";
    setRunning(run.running);
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
// Connect this site to the visitor's EvoLoRA agent, and one-click setup
// ---------------------------------------------------------------------------

const connectState = $("#connect-state");
const connectError = $("#connect-error");
let setupTimer = 0;
let pairingPopup = null;

function storage(action, value) {
  try {
    if (action === "get") return localStorage.getItem(TOKEN_KEY) || "";
    if (action === "set") localStorage.setItem(TOKEN_KEY, value);
    if (action === "clear") localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage blocked (private mode): stay connected for this tab only.
  }
  return "";
}

function setConnectState(state, text) {
  connectState.dataset.state = state;
  connectState.textContent = text;
}

async function agentHello() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(`${AGENT}/api/agent/hello`, { signal: controller.signal });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function useAgent(token) {
  conn.base = AGENT;
  conn.token = token;
  setConnectState("connected", "Connected to your computer");
  $("#connect-intro").hidden = true;
  $("#download-btn").hidden = true;
  $("#connect-btn").hidden = true;
  $("#disconnect-btn").hidden = false;
  connectError.hidden = true;
  await loadStatus();
  loadHistory();
  loadSetup();
}

function disconnect() {
  storage("clear");
  conn.base = "";
  conn.token = "";
  location.reload();
}

$("#connect-btn").addEventListener("click", async () => {
  connectError.hidden = true;
  // Open the popup inside the click (popup blockers), then check the agent is there.
  pairingPopup = window.open(
    `${AGENT}/pair.html?origin=${encodeURIComponent(location.origin)}`,
    "evolora-pair",
    "width=480,height=560",
  );
  const popup = pairingPopup;
  if (!popup) {
    connectError.textContent = "Allow popups for this site, then press Connect again.";
    connectError.hidden = false;
    return;
  }
  const hello = await agentHello();
  if (!hello) {
    popup?.close();
    setConnectState("none", "EvoLoRA isn't running on this computer");
    connectError.textContent = "Start EvoLoRA from the Start menu (or install it with the download above), then press Connect again.";
    connectError.hidden = false;
    return;
  }
  setConnectState("found", "Found EvoLoRA: approve the request in the popup");
});

window.addEventListener("message", (event) => {
  if (event.origin !== AGENT || event.source !== pairingPopup || !pairingPopup || !event.data) return;
  if (event.data.type === "evolora-paired" && typeof event.data.token === "string" && event.data.token) {
    pairingPopup = null;
    storage("set", event.data.token);
    useAgent(event.data.token);
  } else if (event.data.type === "evolora-pair-denied") {
    pairingPopup = null;
    setConnectState("none", "Connection declined");
  }
});

$("#disconnect-btn").addEventListener("click", disconnect);

function renderSetup(s) {
  $("#setup").hidden = false;
  const hw = s.hardware;
  $("#setup-hardware").textContent =
    `Detected: ${hw.gpus.join(", ") || "no GPU found"} · ${hw.ram_gb} GB RAM`;
  $("#setup-list").replaceChildren(
    ...s.components.map((c) =>
      el(
        "li",
        {},
        el("span", { text: `${c.title} (${c.size})` }),
        el("span", { className: c.installed ? "ok" : "todo", text: c.installed ? "Installed" : "Not installed" }),
      ),
    ),
  );
  const job = s.job;
  const running = job?.status === "running";
  const allInstalled = s.components.every((c) => c.installed);
  const install = $("#setup-install");
  install.disabled = running || allInstalled;
  install.textContent = allInstalled ? "Everything is installed" : running ? "Installing…" : "Install everything";
  const progress = $("#setup-progress");
  progress.hidden = !running;
  if (running && job.progress != null) progress.value = job.progress;
  else progress.removeAttribute("value");
  const log = $("#setup-log");
  log.hidden = !job;
  if (job) {
    log.textContent = job.log.join("\n");
    log.scrollTop = log.scrollHeight;
  }
  clearTimeout(setupTimer);
  if (running) setupTimer = setTimeout(loadSetup, 1000);
  else if (job?.status === "done") loadStatus();
}

async function loadSetup() {
  try {
    renderSetup(await api("/api/setup"));
  } catch (error) {
    connectError.textContent = error.message;
    connectError.hidden = false;
  }
}

$("#setup-install").addEventListener("click", async () => {
  try {
    await api("/api/setup/install", { method: "POST", body: {} });
    loadSetup();
  } catch (error) {
    connectError.textContent = error.message;
    connectError.hidden = false;
  }
});

async function initConnection() {
  const local = servedByAgent && (await api("/api/status").catch(() => ({}))).mode === "local";
  if (local) {
    // This page comes from the agent itself: no pairing needed, just setup.
    $("#connect-title").textContent = "This computer";
    $("#connect-intro").hidden = true;
    $("#connect-actions").hidden = true;
    setConnectState("connected", "Running locally");
    loadSetup();
    return;
  }
  const token = storage("get");
  if (token && (await agentHello())) {
    await useAgent(token);
  }
}

// ---------------------------------------------------------------------------
// Chat with trained adapters (local agent)
// ---------------------------------------------------------------------------

const chatModel = $("#chat-model");
const chatAnswer = $("#chat-answer");
let baseModels = [];

// Adapters exist only on the visitor's own machine, so the hosted page shows the
// playground as a locked preview until a local agent is connected.
function setChatLocked(locked) {
  $("#chat-locked").hidden = !locked;
  $("#chat").dataset.locked = String(locked);
  for (const control of $("#chat-form").elements) control.disabled = locked;
  if (!locked) $("#chat-send").disabled = run.running;
}
setChatLocked(true);

async function loadAdapters() {
  if (run.mode !== "local") return;
  try {
    const { adapters } = await api("/api/adapters");
    chatModel.replaceChildren(
      ...adapters.map((a) => el("option", { value: `adapter:${a.name}`, text: `${a.name} (${a.base_model.split("/").pop()})` })),
      ...baseModels.map((m) => el("option", { value: `base:${m.id}`, text: `${m.label}, untrained` })),
    );
  } catch {
    // Chat is optional; leave the selector as it is.
  }
}

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const [kind, ...rest] = chatModel.value.split(":");
  const id = rest.join(":");
  const send = $("#chat-send");
  send.disabled = true;
  chatAnswer.hidden = false;
  chatAnswer.textContent = "Loading the model and generating…";
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: {
        prompt: $("#chat-prompt").value,
        adapter: kind === "adapter" ? id : null,
        base_model_id: kind === "base" ? id : null,
      },
    });
    chatAnswer.replaceChildren(
      result.answer || "(empty answer)",
      el("span", { className: "meta", text: `${result.adapter || result.base_model} · ${result.seconds}s${result.goal_applied ? " · training instruction added" : ""}` }),
    );
  } catch (error) {
    chatAnswer.textContent = error.message;
  } finally {
    send.disabled = run.running;
  }
});

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
  ".hero-center h1, .hero-sub, .hero-lede, .hero-cta, .anchor-player",
  ".facts > div",
  ".section-head",
  ".services, .runner > *, .history, .connect-panel",
  ".cards > .card",
  ".lanes, .panel-table",
  ".stack-grid > *, .stack-line",
  ".commands > .command, #local .note, .changelog-card",
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

if ("IntersectionObserver" in window && !reducedMotion.matches) setUpReveals();

function syncMenuClock() {
  const node = $("#menu-clock");
  if (!node) return;
  const now = new Date();
  node.textContent = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  node.dateTime = now.toISOString();
}
syncMenuClock();
setInterval(syncMenuClock, 30_000);

(function initHeroVideo() {
  const video = $("#hero-video");
  const play = $("#anchor-play");
  const frame = $("#anchor-frame");
  if (!video || !play || !frame) return;
  // The poster stays visible until play; native controls appear only once it is playing.
  const showPlaying = () => {
    frame.classList.add("is-playing");
    video.controls = true;
    play.hidden = true;
  };
  play.addEventListener("click", () => {
    video.play().catch(() => {
      play.textContent = "video unavailable";
      play.disabled = true;
    });
  });
  video.addEventListener("play", showPlaying);
  video.addEventListener("ended", () => {
    frame.classList.remove("is-playing");
    video.controls = false;
    play.hidden = false;
  });
})();

const loopSvg = $("#loop-svg");
function syncLoopMotion() {
  if (reducedMotion.matches) loopSvg.pauseAnimations();
  else loopSvg.unpauseAnimations();
}
reducedMotion.addEventListener("change", syncLoopMotion);
syncLoopMotion();

// ---------------------------------------------------------------------------

setState("READY", "idle");
let latestDeviceResources = {};
let latestResourceSettings = {};
function renderResources(resources, settings = {}) {
  if (resources.gpu_total_bytes) latestDeviceResources = resources;
  if (settings.batch_size) latestResourceSettings = settings;
  if (settings.inference_batch) latestResourceSettings = settings;
  const info = { ...latestDeviceResources, ...resources };
  const gib = (n) => typeof n === "number" ? `${(n / 1024 ** 3).toFixed(1)} GiB` : "unavailable";
  $("#resource-panel").hidden = false;
  $("#ram-value").textContent = `${gib(info.ram_available_bytes)} free / ${gib(info.ram_total_bytes)}`;
  $("#ram-meter").hidden = !info.ram_total_bytes;
  $("#ram-meter").value = info.ram_total_bytes ? Math.min(1, info.ram_available_bytes / info.ram_total_bytes) : 0;
  $("#cpu-value").textContent = `${info.cpu_threads || "?"} threads${typeof info.cpu_utilization_percent === "number" ? ` · ${info.cpu_utilization_percent.toFixed(0)}% busy` : ""}`;
  let fresh = false;
  if (info.gpu_total_bytes) {
    fresh = Date.now() / 1000 - latestDeviceResources.sampled_at < 15;
    $("#gpu-value").textContent = `${gib(info.gpu_free_bytes)} free / ${gib(info.gpu_total_bytes)}${fresh ? "" : " (last report)"}`;
    $("#gpu-meter").hidden = !fresh;
    $("#gpu-meter").value = Math.min(1, info.gpu_free_bytes / info.gpu_total_bytes);
  }
  $("#batch-value").textContent = latestResourceSettings.batch_size
    ? `${latestResourceSettings.batch_size} per step · ${latestResourceSettings.grad_accum} accumulation`
    : latestResourceSettings.inference_batch ? `${latestResourceSettings.inference_batch} prompts per batch` : "Scales to available memory";
  const pressure = typeof info.ram_available_bytes === "number" && info.ram_available_bytes < 2 * 1024 ** 3;
  $("#resource-panel").dataset.pressure = pressure ? "high" : "normal";
  const node = $("#resource-status");
  node.hidden = false;
  node.textContent = pressure ? "Memory is tight. Execution will use smaller batches; closing other apps can help."
    : "Batches adapt to available memory while keeping training and evaluation content intact.";
}
async function loadResources() {
  if (!servedByAgent && !conn.base) return;
  try {
    const info = await api("/api/resources");
    if (info.available) renderResources(info);
    if (run.blocker === "offline") loadStatus();
  } catch {
    $("#resource-status").textContent = "Agent offline. Resource readings are paused.";
    run.ready = false;
    run.blocker = "offline";
    $("#readiness").dataset.ready = "false";
    $("#readiness").textContent = "Agent offline";
    setRunning(run.running);
  }
}
function renderDiagnostics(results) {
  if (!results?.length) return;
  $("#eval-diagnostics").hidden = false;
  $("#eval-diagnostic-list").replaceChildren(...results.slice(0, 10).map((sample) => {
    const details = sample.details || {};
    const failed = Object.entries(details.checks || {}).filter(([, passed]) => !passed).map(([key]) => key);
    const reason = details.error || details.reason || (failed.length ? `Check: ${failed.join(", ")}` : "All checks passed");
    return el("article", {}, el("strong", { text: `${sample.sample_id} · ${fmt(sample.score)} · ${sample.passed ? "Passed" : "Needs improvement"}` }),
      el("p", { text: reason }), el("pre", { text: `Prompt: ${details.input || "See approved eval set"}\n\nModel output: ${details.actual || "(empty)"}` }));
  }));
}
$("#log-follow").addEventListener("click", (event) => {
  run.followLog = !run.followLog;
  event.currentTarget.setAttribute("aria-pressed", String(run.followLog));
  event.currentTarget.textContent = run.followLog ? "Follow live" : "Scroll paused";
  if (run.followLog) ui.log.scrollTop = ui.log.scrollHeight;
});
$("#log-copy").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(ui.log.innerText);
    button.textContent = "Copied";
  } catch { button.textContent = "Select log to copy"; }
  setTimeout(() => { button.textContent = "Copy log"; }, 2000);
});
setInterval(() => {
  if (!run.running) return;
  const seconds = Math.floor((performance.now() - run.startedAt) / 1000);
  $("#elapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}, 1000);
loadResources();
setInterval(loadResources, 5000);
loadStatus();
loadHistory();
loadEvalSet();
updateName();
initConnection();
