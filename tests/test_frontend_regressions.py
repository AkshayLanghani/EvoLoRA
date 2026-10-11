"""Exercise browser lifecycle helpers without requiring a graphical browser."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path("public/assets/app.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed for frontend checks")


def run_script(script: str) -> None:
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fails", [False, True])
def test_approval_remains_retryable_until_acknowledged(fails: bool) -> None:
    helper = SOURCE.split("async function answerGate(approved) {", 1)[1].split(
        '\n$("#gate-yes").addEventListener', 1
    )[0]
    run_script(
        "const assert = require('node:assert/strict');\n"
        "const buttons = { '#gate-yes': {disabled: false, focus() {}}, "
        "'#gate-no': {disabled: false} };\n"
        "const $ = (id) => buttons[id];\n"
        "const run = {runId: 'run-id', running: true};\n"
        "let hidden = false; let errors = 0;\n"
        "function hideGate() {hidden = true;}\n"
        "function appendLog() {errors++;}\n"
        f"const fails = {str(fails).lower()};\n"
        "async function api() {\n"
        " assert.equal(hidden, false);\n"
        " assert.equal(buttons['#gate-yes'].disabled, true);\n"
        " if (fails) throw new Error('network disconnected');\n"
        "}\n"
        "async function answerGate(approved) {" + helper + "\n"
        "answerGate(true).then(() => {\n"
        " assert.equal(hidden, !fails);\n"
        " assert.equal(errors, fails ? 1 : 0);\n"
        " assert.equal(buttons['#gate-yes'].disabled, false);\n"
        " assert.equal(buttons['#gate-no'].disabled, false);\n"
        "}).catch((error) => {console.error(error); process.exitCode = 1;});"
    )


def test_ready_clears_previous_pipeline_stage() -> None:
    helper = SOURCE.split("function setState(label, mood) {", 1)[1].split(
        "\nfunction appendLog", 1
    )[0]
    run_script(
        "const assert = require('node:assert/strict');\n"
        "const ui = {state: {dataset: {}}};\n"
        "const items = Array.from({length: 6}, (_, i) => ({\n"
        " dataset: {stage: String(i)}, done: true, current: true,\n"
        " classList: {toggle(name, value) {items[i].done = value;}},\n"
        " setAttribute() {this.current = true;},\n"
        " removeAttribute() {this.current = false;}\n"
        "}));\n"
        "const document = {querySelectorAll() {return items;}};\n"
        "function setState(label, mood) {" + helper + "\n"
        "setState('READY');\n"
        "assert.ok(items.every((item) => !item.done && !item.current));"
    )


@pytest.mark.parametrize("finished_while_waiting", [False, True])
def test_cancel_only_aborts_the_original_active_stream(finished_while_waiting: bool) -> None:
    helper = SOURCE.split('ui.cancel.addEventListener("click", async () => {', 1)[1].split(
        "\nconst presets", 1
    )[0]
    run_script(
        "const assert = require('node:assert/strict');\n"
        "let aborts = 0;\n"
        "const controller = {abort() {aborts++;}};\n"
        "const run = {controller, runId: 'run/id', running: true};\n"
        "const ui = {cancel: {}, error: {}};\n"
        f"const finished = {str(finished_while_waiting).lower()};\n"
        "async function api(url) {\n"
        " assert.equal(url, '/api/runs/run%2Fid/cancel');\n"
        " if (finished) {run.controller = null; run.running = false;}\n"
        "}\n"
        "let cancel; const listener = {addEventListener(kind, callback) {cancel = callback;}};\n"
        "listener.addEventListener('click', async () => {" + helper + "\n"
        "cancel().then(() => {assert.equal(aborts, finished ? 0 : 1);})\n"
        ".catch((error) => {console.error(error); process.exitCode = 1;});"
    )
