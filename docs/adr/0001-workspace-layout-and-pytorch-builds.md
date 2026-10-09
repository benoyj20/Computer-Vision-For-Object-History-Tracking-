# 0001: Workspace layout and PyTorch builds

Date: 2026-10-08. Status: accepted.

## Context

Two contributors work on separate components (detector and tracker) that meet at one data
contract. Training needs CUDA PyTorch on an RTX 5070 (Blackwell) laptop GPU, while CI and
CPU-only machines need a small CPU build. On Windows, the default PyPI PyTorch wheel is
CPU-only, so a plain install silently trains on CPU.

## Decision

- Use a uv workspace with one package per component under `packages/`, sharing a single
  `uv.lock`. Shared types live in `packages/core`; other packages depend on it, not on each other,
  unless a later decision adds that dependency.
- Use Python 3.12 (`.python-version`) for broad ML wheel support.
- Declare two mutually exclusive root extras: `cu130` resolves `torch` and `torchvision` from the
  PyTorch CUDA 13.0 index, and `cpu` from the CPU index. Contributors pass one explicitly.
  CUDA 13.0 supports Blackwell GPUs and requires NVIDIA driver 580 or newer.
- Use the Ultralytics CLI (`yolo detect train`) for training rather than a wrapper until the
  project needs behavior the CLI lacks.

## Consequences

- `uv.lock` contains PyPI, CPU, and CUDA variants of PyTorch; always sync with an extra.
- CI installs `--extra cpu` and cannot test GPU behavior. GPU checks are local and marked
  `@pytest.mark.gpu`, and PRs that affect GPU paths must report them.
- A contributor with an older driver must update the driver or switch the index (for example to
  `cu128`) in a new decision record.
