# Physical Version Control for Object History Tracking

A fixed camera over a shared kitchen counter takes snapshots over time. This project turns those
snapshots into an object-level history: what is on the counter, where each item is, when it was
added, moved, or removed, and how long it has stayed in one place. A natural-language assistant
answers questions such as "how long has the milk been out?" or "where is the dish soap?" from
that history.

CS5330 Computer Vision final project, Northeastern University, by Justin Gubbens and Benoy Joseph.

## Pipeline

1. **Detect**: a fine-tuned Ultralytics YOLO26 model finds kitchen objects in each snapshot.
2. **Track**: each detection is matched to an existing object by position and appearance, so the
   same object keeps the same identity across frames.
3. **Record history**: added, moved, and removed events are logged with the time interval in which
   each change happened. Snapshots are sparse, so placement times are intervals, not exact times.
4. **Ask**: a Next.js web page where a Claude-powered assistant, behind a FastAPI service,
   answers questions using read-only queries over the SQLite history.
5. **Evaluate**: scripted scenarios with known changes measure event precision and recall.

## Current status

The repository is set up: a uv workspace with one package per component, the shared
detector-to-tracker contract, a GPU environment check, a starter Next.js web app, CI, and
contributor rules. Detection
training, tracking, the assistant, and evaluation are not implemented yet. See the
[project checklist](docs/project-checklist.md) for what exists and what remains.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js 22+, pnpm 10, and Git. GNU Make is optional; every target
in the `Makefile` is a single `uv` command you can run directly. For GPU training, an NVIDIA
driver 580 or newer (check with `nvidia-smi`).

```sh
# NVIDIA GPU (CUDA 13.0 PyTorch build)
uv sync --all-packages --extra cu130      # or: make bootstrap
uv run python -m objhist_detection.env_check --require-cuda   # or: make env-check

# CPU only (macOS, machines without NVIDIA GPUs)
uv sync --all-packages --extra cpu        # or: make bootstrap TORCH=cpu
```

Install the web app's dependencies with `pnpm install`, then start it with `make dev-web`
(<http://localhost:3000>).

uv installs Python 3.12 automatically if needed. Always pass one of the two extras; a plain
`uv sync` falls back to the default PyPI PyTorch build, which is CPU-only on Windows.

Run the checks used by CI (or everything at once with `make check`):

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

Once a dataset is in place (see [data/README.md](data/README.md)), fine-tune YOLO26:

```sh
make train DATA=data/processed/kitchen/data.yaml EPOCHS=100
```

## Repository layout

| Path | Contents | Lead |
| --- | --- | --- |
| `packages/core` | Shared data contracts (`BoundingBox`, `Detection`, `FrameDetections`) | Joint |
| `packages/detection` | Dataset preparation, YOLO26 training and inference | Benoy |
| `packages/tracking` | Identity association and the object history log | Justin |
| `packages/assistant` | Natural-language questions over the history | Joint |
| `packages/evaluation` | Scripted scenarios, event precision and recall | Joint |
| `apps/web` | Next.js web page for asking questions and viewing history | Joint |
| `data/` | Local datasets, captures, and outputs (not committed) | |
| `docs/` | Plan, checklist, data card, evaluation protocol, decisions | |

## Documentation

- [Problem statement and plan](docs/problem-statement-and-plan.md)
- [Project checklist](docs/project-checklist.md)
- [Data card](docs/data-card.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Architecture decisions](docs/adr/README.md)

## Contributing

See [the contributor guide](CONTRIBUTING.md) for setup, branch naming, verification, and PR
review. Coding agents must also follow [AGENTS.md](AGENTS.md); Claude Code skills and agents for
this repository are described in [.claude/skills/README.md](.claude/skills/README.md).
