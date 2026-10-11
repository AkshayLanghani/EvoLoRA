"""Memory budgets constrain scale-up and preserve effective training batches."""

from types import SimpleNamespace

import pytest

from evolora.local.resources import GIB, device_resources, microbatch_limit, system_resources


@pytest.mark.parametrize("ram,gpu,expected", [
    (12, 8, 4), (4, 8, 2), (12, 2, 1), (2, 8, 1), (12, None, 4),
])
def test_shared_gpu_and_ram_budgets_both_limit_microbatch(ram, gpu, expected):
    resources = {"ram_available_bytes": ram * GIB,
                 "gpu_free_bytes": None if gpu is None else gpu * GIB}
    assert microbatch_limit(4, 4, resources) == expected


def test_scale_up_and_throttle_follow_changes_in_available_memory():
    assert [microbatch_limit(4, 4, {"ram_available_bytes": free * GIB})
            for free in (3, 4, 10, 3)] == [1, 2, 4, 1]


def test_cap_and_effective_batch_are_never_exceeded_or_changed():
    plenty = {"ram_available_bytes": 40 * GIB}
    assert microbatch_limit(4, 3, plenty) == 2  # 3 would change the effective batch
    assert microbatch_limit(2, 32, plenty) == 2
    assert microbatch_limit(4, 4, {}) == 1


def test_unavailable_driver_telemetry_does_not_fail_the_run():
    def unavailable():
        raise RuntimeError("driver telemetry unavailable")

    torch = SimpleNamespace(xpu=SimpleNamespace(mem_get_info=unavailable))
    assert device_resources(torch, "xpu")["gpu_free_bytes"] is None


def test_gpu_telemetry_uses_driver_free_memory():
    torch = SimpleNamespace(xpu=SimpleNamespace(mem_get_info=lambda: (3 * GIB, 8 * GIB)))
    info = device_resources(torch, "xpu")
    assert info["gpu_free_bytes"] == 3 * GIB
    assert info["gpu_total_bytes"] == 8 * GIB


def test_system_telemetry_is_nonnegative_or_explicitly_unknown():
    info = system_resources()
    assert info["cpu_threads"] >= 1
    assert info["sampled_at"] > 0
    for key in ("ram_total_bytes", "ram_available_bytes", "disk_available_bytes"):
        assert info[key] is None or info[key] >= 0


@pytest.mark.parametrize("profile,available", [("local", True), ("cloud", False)])
def test_resource_endpoint_is_local_only(profile, available, monkeypatch):
    from fastapi.testclient import TestClient

    from evolora.config import get_config
    from evolora.web.app import app

    monkeypatch.setenv("EVOLORA_PROFILE", profile)
    get_config.cache_clear()
    try:
        body = TestClient(app).get("/api/resources").json()
        assert body["available"] is available
        assert ("ram_available_bytes" in body) is available
    finally:
        get_config.cache_clear()


@pytest.mark.parametrize("oom", [False, True])
def test_inference_scales_with_live_memory_and_retries_same_prompts(oom, monkeypatch, tmp_path):
    import json
    import sys
    from contextlib import nullcontext

    from evolora.local import resources, worker

    monkeypatch.setitem(sys.modules, "resources", resources)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr(worker, "pick_device", lambda: ("cpu", None))
    monkeypatch.setattr(worker, "release_gpu_memory", lambda: None)
    snapshots = iter([3, 2.5, 12])
    monkeypatch.setattr(worker, "resource_snapshot", lambda *args: {
        "ram_available_bytes": (10 if oom else next(snapshots)) * GIB,
        "gpu_free_bytes": 20 * GIB,
    })

    class Tensor:
        def __init__(self, texts):
            self.texts = texts
            self.shape = (len(texts), 7)

        def __getitem__(self, key):
            assert key == (slice(None), slice(7, None))
            return ["answer:" + text for text in self.texts]

    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, texts, **kwargs):
            return Encoding(input_ids=Tensor(texts))

        def decode(self, tokens, **kwargs):
            return tokens

    class Model:
        generation_config = SimpleNamespace()
        calls = []

        def eval(self):
            pass

        def generate(self, input_ids, **kwargs):
            self.calls.append(len(input_ids.texts))
            if oom and len(self.calls) == 1:
                raise RuntimeError("out of memory")
            return input_ids

    model = Model()
    monkeypatch.setattr(worker, "load_model", lambda *args: (model, Tokenizer()))
    count = 4 if oom else 6
    prompts = [{"sample_id": str(i), "prompt": f"prompt {i}"} for i in range(count)]
    output = tmp_path / "outputs.json"
    worker.generate({"base_model": "test", "prompts": prompts, "output_file": str(output)})
    assert model.calls == ([4, 2, 2] if oom else [2, 1, 3])
    assert json.loads(output.read_text()) == {str(i): f"answer:prompt {i}" for i in range(count)}
