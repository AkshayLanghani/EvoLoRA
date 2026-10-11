"""Lightweight resource telemetry, also importable by the standalone training worker."""

from __future__ import annotations

import ctypes
import os
import shutil
import time
from pathlib import Path
from threading import Lock

GIB = 1024 ** 3
_cpu_sample: tuple[int, int] | None = None
_cpu_lock = Lock()


def _cpu_utilization() -> float | None:
    """CPU busy percentage between telemetry samples, without a blocking sampling sleep."""
    global _cpu_sample
    if os.name == "nt":
        from ctypes import wintypes

        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user),
        ):
            return None

        def ticks(value):
            return (value.dwHighDateTime << 32) | value.dwLowDateTime

        current = (ticks(idle), ticks(kernel) + ticks(user))
    else:
        values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        current = (sum(values[3:5]), sum(values))
    with _cpu_lock:
        previous, _cpu_sample = _cpu_sample, current
    if previous is None or current[1] <= previous[1]:
        return None
    return round(max(0, min(100, 100 * (1 - (current[0] - previous[0]) /
                                      (current[1] - previous[1])))), 1)


def system_resources() -> dict:
    """Report currently available physical RAM; unknown metrics remain null."""
    result = {"ram_total_bytes": None, "ram_available_bytes": None,
              "cpu_threads": os.cpu_count() or 1, "disk_available_bytes": None,
              "cpu_utilization_percent": None,
              "sampled_at": time.time()}
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_physical", "available_physical", "total_page", "available_page",
                        "total_virtual", "available_virtual", "available_extended",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            result.update(ram_total_bytes=status.total_physical,
                          ram_available_bytes=status.available_physical)
        else:
            values = {line.split(":")[0]: int(line.split()[1]) * 1024
                      for line in Path("/proc/meminfo").read_text().splitlines()}
            result.update(ram_total_bytes=values["MemTotal"],
                          ram_available_bytes=values["MemAvailable"])
        result["disk_available_bytes"] = shutil.disk_usage(Path.cwd()).free
        result["cpu_utilization_percent"] = _cpu_utilization()
    except (OSError, AttributeError, KeyError, ValueError):
        pass  # telemetry must not prevent a run on unsupported hosts
    return result


def device_resources(torch, device: str) -> dict:
    """GPU free memory is driver-reported, not inferred from model parameter count."""
    result = system_resources() | {"device": device, "gpu_free_bytes": None,
                                   "gpu_total_bytes": None}
    if device in {"cuda", "xpu"}:
        try:
            api = getattr(torch, device)
            free, total = api.mem_get_info()
            result.update(gpu_free_bytes=int(free), gpu_total_bytes=int(total))
        except (RuntimeError, AttributeError):
            pass
    return result


def microbatch_limit(effective_batch: int, cap: int, resources: dict,
                     per_example_bytes: int = GIB) -> int:
    """Largest divisor within a conservative live memory budget; OOM remains authoritative.

    Shared Intel GPU memory is constrained by BOTH available system RAM and the
    driver budget. Keep 2 GiB for Windows/apps and 1 GiB for device workspaces.
    """
    budgets = []
    ram = resources.get("ram_available_bytes")
    gpu = resources.get("gpu_free_bytes")
    if ram is not None:
        budgets.append(max(0, ram - 2 * GIB))
    if gpu is not None:
        budgets.append(max(0, gpu - GIB))
    limit = min(cap, effective_batch)
    if not budgets:
        limit = 1  # unknown capacity cannot justify speculative scale-up
    else:
        limit = min(limit, max(1, min(budgets) // max(1, per_example_bytes)))
    return max(n for n in range(1, int(limit) + 1) if effective_batch % n == 0)
