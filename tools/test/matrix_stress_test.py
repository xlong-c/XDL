"""矩阵乘法压测工具。

直接修改 `CONFIG` 后运行，不使用命令行参数解析库。
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch


CONFIG = {
    "device": None,
    "size": 8192,
    "duration": 30,
    "dtype": "float32",
}


def get_device(device_name: Optional[str] = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stress_test(
    device_name: Optional[str] = None,
    size: int = 8192,
    duration: int = 30,
    dtype: str = "float32",
) -> None:
    device = get_device(device_name)

    dtype_map: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print("==================================================")
    print("Starting Matrix Multiplication Stress Test")
    print(f"Device   : {device}")
    print(f"Matrix   : {size}x{size}")
    print(f"Type     : {dtype}")
    print(f"Duration : {duration} seconds")
    print("==================================================")

    # Memory allocation
    try:
        print("Allocating memory...")
        a = torch.randn(size, size, device=device, dtype=torch_dtype)
        b = torch.randn(size, size, device=device, dtype=torch_dtype)
        print("Memory allocated.")
    except Exception as e:
        print(f"Error allocating memory: {e}")
        return

    start_time = time.time()
    end_time = start_time + duration
    iter_count = 0

    print("Running matrix multiplication loop...")
    try:
        while time.time() < end_time:
            _ = torch.matmul(a, b)

            # Ensure computation is finished for timing accuracy on GPU
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            iter_count += 1

            elapsed = time.time() - start_time
            if iter_count % 5 == 0:
                print(
                    f"Iteration {iter_count:4d} | Elapsed: {elapsed:6.2f}s | Speed: {iter_count / elapsed:.2f} it/s"
                )

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    total_time = time.time() - start_time
    print("==================================================")
    print("Test Completed")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Iterations: {iter_count}")
    print(f"Average Speed: {iter_count / total_time:.2f} iterations/second")

    # Calculate approximate TFLOPS
    # 2 * N^3 operations per matmul
    ops_per_iter = 2 * (size**3)
    total_ops = ops_per_iter * iter_count
    tflops = (total_ops / total_time) / 1e12

    print(f"Approximate Performance: {tflops:.4f} TFLOPS")
    print("==================================================")


if __name__ == "__main__":
    stress_test(
        device_name=CONFIG["device"],
        size=CONFIG["size"],
        duration=CONFIG["duration"],
        dtype=CONFIG["dtype"],
    )
