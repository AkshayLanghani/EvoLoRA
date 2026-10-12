"""Tests for the one-click local setup (no downloads)."""

import asyncio
from pathlib import Path

from evolora.config import find_llama_server, get_config
from evolora.local import installer
from evolora.local.installer import (
    LLAMA_CPP_BUILDS,
    LLAMA_CPP_FLAVOR,
    TORCH_BACKEND,
    Hardware,
    SetupManager,
    classify_gpus,
)


def test_gpu_vendor_classification():
    assert classify_gpus(["Intel(R) Arc(TM) 140V GPU (8GB)"]) == "intel"
    assert classify_gpus(["NVIDIA GeForce RTX 4060", "Intel(R) UHD Graphics"]) == "nvidia"
    assert classify_gpus(["AMD Radeon 780M"]) == "other"
    assert classify_gpus([]) == "other"


def test_every_vendor_has_a_build_and_torch_backend():
    for vendor in ("intel", "nvidia", "other"):
        assert LLAMA_CPP_BUILDS[vendor] and LLAMA_CPP_FLAVOR[vendor] and TORCH_BACKEND[vendor]
    assert TORCH_BACKEND["intel"] == "xpu"
    assert any("sycl" in a for a in LLAMA_CPP_BUILDS["intel"])
    assert any("cudart" in a for a in LLAMA_CPP_BUILDS["nvidia"])  # CUDA runtime DLLs


def test_find_llama_server_prefers_gpu_builds(tmp_path):
    for flavor in ("vulkan", "sycl"):
        exe = tmp_path / "llama-b1" / flavor / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
    assert Path(find_llama_server(str(tmp_path))).parent.name == "sycl"
    assert find_llama_server(str(tmp_path / "empty")) == ""


def test_partial_model_download_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    model_dir = tmp_path / "hub" / "models--Org--Model"
    snapshot = model_dir / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert not installer.hf_cache_has("Org/Model")  # config only, no weights

    (snapshot / "model.safetensors").write_bytes(b"0" * 2_000_000)
    assert installer.hf_cache_has("Org/Model")

    (model_dir / "blobs").mkdir()
    (model_dir / "blobs" / "123.incomplete").write_bytes(b"")
    assert not installer.hf_cache_has("Org/Model")  # a download still in progress


def _manager(monkeypatch, calls, fail_on=None):
    monkeypatch.setattr(installer, "detect_hardware", lambda: Hardware(["Intel Arc"], 16.0, "intel"))

    def fake_install(key):
        def install(cfg, hw, report):
            calls.append(key)
            report(f"Downloading {key}: 1 / 2 MB", 0.5)
            report(f"Downloading {key}: 2 / 2 MB", 1.0)
            if key == fail_on:
                raise RuntimeError("network down")
        return install

    fakes = [
        installer.Component(c.key, c.title, c.size, lambda cfg: False, fake_install(c.key))
        for c in installer.COMPONENTS
    ]
    monkeypatch.setattr(installer, "COMPONENTS", fakes)
    return SetupManager(get_config)


async def _wait(manager):
    for _ in range(200):
        if manager.job.status != "running":
            return
        await asyncio.sleep(0.01)


async def test_setup_runs_missing_components_in_order(monkeypatch):
    calls = []
    manager = _manager(monkeypatch, calls)
    status = manager.status()
    assert status["hardware"]["vendor"] == "intel"
    assert [c["installed"] for c in status["components"]] == [False] * 4

    manager.start()
    await _wait(manager)
    assert manager.job.status == "done"
    assert calls == ["llama_server", "planner_model", "training_env", "base_model"]
    # Download progress updates one line instead of flooding the log.
    assert sum(line.startswith("Downloading llama_server") for line in manager.job.log) == 1


async def test_setup_failure_is_reported(monkeypatch):
    calls = []
    manager = _manager(monkeypatch, calls, fail_on="training_env")
    manager.start(["training_env", "base_model"])
    await _wait(manager)
    assert manager.job.status == "failed"
    assert "network down" in manager.job.error
    assert calls == ["training_env"]  # stops at the first failure


async def test_unknown_component_is_rejected(monkeypatch):
    manager = _manager(monkeypatch, [])
    try:
        manager.start(["gpu_drivers"])
    except ValueError as exc:
        assert "gpu_drivers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_apple_silicon_gets_the_metal_build_and_plain_torch():
    assert LLAMA_CPP_FLAVOR["apple"] == "metal"
    assert LLAMA_CPP_BUILDS["apple"] == ["llama-{tag}-bin-macos-arm64.tar.gz"]
    assert TORCH_BACKEND["apple"] == "default"  # the PyPI wheel already has MPS


def test_mac_hardware_detection(monkeypatch):
    values = {"hw.memsize": str(16 * 2**30), "machdep.cpu.brand_string": "Apple M2"}
    monkeypatch.setattr(installer, "_sysctl", lambda name: values[name])
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    hw = installer.detect_mac_hardware()
    assert (hw.vendor, hw.ram_gb, hw.gpus) == ("apple", 16.0, ["Apple M2 GPU"])

    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    assert installer.detect_mac_hardware().vendor == "other"  # Intel Macs are unsupported


def test_find_llama_server_looks_inside_the_nested_macos_archive(tmp_path):
    binary = tmp_path / "llama-b1" / "metal" / "llama-b1" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    found = find_llama_server(str(tmp_path), name="llama-server")
    assert Path(found) == binary.resolve()


def test_tar_archives_are_extracted(tmp_path):
    import tarfile

    source = tmp_path / "llama-b1" / "llama-server"
    source.parent.mkdir()
    source.write_bytes(b"binary")
    archive = tmp_path / "llama-b1-bin-macos-arm64.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source, arcname="llama-b1/llama-server")
    installer._extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "llama-b1" / "llama-server").read_bytes() == b"binary"


def test_vm_stat_is_parsed_into_available_bytes():
    from evolora.local.resources import parse_vm_stat

    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               10000.\n"
        "Pages active:                            200000.\n"
        "Pages inactive:                           30000.\n"
        "Pages speculative:                         2000.\n"
    )
    assert parse_vm_stat(text) == 42000 * 16384
