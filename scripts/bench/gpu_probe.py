"""Record the Intel XPU stack and time a synchronized bf16 matrix multiply."""

import json
import platform
import time

import torch


def main() -> None:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_xpu_available": torch.xpu.is_available(),
    }
    if torch.xpu.is_available():
        result["device"] = torch.xpu.get_device_name(0)
        result["device_properties"] = str(torch.xpu.get_device_properties(0))
        try:
            a = torch.randn((2048, 2048), device="xpu", dtype=torch.bfloat16)
            b = torch.randn((2048, 2048), device="xpu", dtype=torch.bfloat16)
            for _ in range(3):
                c = a @ b
            torch.xpu.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                c = a @ b
            torch.xpu.synchronize()
            result["bf16_matmul_ms"] = (time.perf_counter() - start) * 100
            result["bf16_output_dtype"] = str(c.dtype)
            result["bf16_output_mean"] = c.float().mean().item()
            result["peak_allocated_bytes"] = torch.xpu.max_memory_allocated()
        except Exception as exc:
            result["bf16_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
