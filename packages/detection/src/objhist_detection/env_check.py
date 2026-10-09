"""Report the PyTorch, CUDA, and Ultralytics setup used for training and inference.

Run with `uv run python -m objhist_detection.env_check [--require-cuda]`. The
`--require-cuda` flag exits with status 1 when no CUDA device is usable, so a
CPU-only PyTorch build is caught before a long training run silently uses CPU.
"""

import argparse
import platform
import sys
from dataclasses import dataclass

import torch
import ultralytics


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    torch_version: str
    torch_cuda_build: str | None
    ultralytics_version: str
    cuda_available: bool
    device_name: str | None
    device_memory_gib: float | None


def collect_report() -> EnvironmentReport:
    cuda_available = torch.cuda.is_available()
    device_name = None
    device_memory_gib = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        device_name = properties.name
        device_memory_gib = round(properties.total_memory / 2**30, 1)
    return EnvironmentReport(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda,
        ultralytics_version=ultralytics.__version__,
        cuda_available=cuda_available,
        device_name=device_name,
        device_memory_gib=device_memory_gib,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-cuda", action="store_true", help="fail when no CUDA device is usable"
    )
    args = parser.parse_args(argv)

    report = collect_report()
    for field, value in vars(report).items():
        print(f"{field:20} {value if value is not None else 'unavailable'}")

    if args.require_cuda and not report.cuda_available:
        print(
            "CUDA is unavailable. Re-sync with `uv sync --all-packages --extra cu130` "
            "and check the NVIDIA driver with `nvidia-smi`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
