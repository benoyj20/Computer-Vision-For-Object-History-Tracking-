# Contributing

Read [AGENTS.md](AGENTS.md) for repository-wide engineering and collaboration rules. Use the
[project checklist](docs/project-checklist.md) to distinguish working features from planned work.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22+,
   pnpm 10 (`corepack enable` provides the pinned version), and Git. GNU Make is optional.
2. Clone the repository. `.gitattributes` keeps line endings as LF in the repository, so Windows
   and macOS checkouts produce identical diffs.
3. Copy `.env.example` to `.env` only if you do not already have one, then fill in only the
   credentials you need.
4. Install the workspace with the PyTorch build for your machine:
   - NVIDIA GPU: `uv sync --all-packages --extra cu130`, then confirm the GPU with
     `uv run python -m objhist_detection.env_check --require-cuda`.
   - CPU only: `uv sync --all-packages --extra cpu`.
5. Install the web app with `pnpm install`.
6. Run `make check` (or the commands in the table below) to confirm the install.

Never commit `.env`, credentials, datasets, kitchen captures, or model weights. Put local data
under `data/` as described in [data/README.md](data/README.md).

## Coordinate and create a branch

1. Agree on one primary contributor and a focused task. Benoy leads detection and Justin leads
   tracking and the history log; the assistant and evaluation are shared. Check open PRs before
   starting overlapping work, and agree on changes to `packages/core` before making them.
2. Preserve local work, fetch `origin`, and start from the latest `origin/main`. Do not push
   directly to `main` or work in the other person's branch.
3. Use lowercase, hyphen-separated names: `feature/yolo26-training`, `fix/track-id-swap`,
   `docs/evaluation-protocol`, `refactor/history-store`, `test/association-edge-cases`, or
   `chore/update-dependencies`.
4. Make focused commits with titles such as `fix: keep occluded objects out of removal events`.
   Avoid unrelated formatting, broad cleanup, empty commits, and pushes after every edit.

## Add a dependency

Add it to the package that uses it, for example
`uv add --package objhist-tracking scipy`. This updates that package's `pyproject.toml` and
`uv.lock` together. Commit both. Never hand-edit `uv.lock`; after a merge conflict in it,
take either side and run `uv lock` again. For the web app, run `pnpm --filter @objhist/web add <package>` and commit
`pnpm-lock.yaml`; never hand-edit it either.

## Verify before pushing

| Area | Local verification |
| --- | --- |
| All Python code | `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest` |
| Web app | `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`; for UI changes, a browser check per [apps/web/AGENTS.md](apps/web/AGENTS.md) |
| GPU training or inference | Also `make env-check` and `uv run pytest -m gpu` on a CUDA machine |
| Training or evaluation results | Exact command, dataset version, seed, hardware, and metrics in the PR |
| Documentation only | Check accuracy, referenced paths, and Markdown formatting |

CI runs the first two rows on Ubuntu, with CPU PyTorch (see [ci.yaml](.github/workflows/ci.yaml)).
Sync with `origin/main` before final verification and resolve conflicts by preserving both
contributors' intended behavior. If verification is blocked, report it and keep the work local
unless explicitly asked to share an unverified draft.

## Open and review a pull request

Use the PR template to describe the problem, behavior, testing evidence, and any model, data, or
evaluation impact. Every PR needs an approval from the other contributor and passing `python` and
`javascript` CI checks. New pushes after approval need a fresh review. Authors cannot approve their own PRs.

Review correctness, edge cases, test evidence, shared contracts, and the evidence rules in
[AGENTS.md](AGENTS.md): no fabricated detections or events, placement times reported as
intervals, missed detections kept distinct from removals, and no tuning on the test set.

## Branch protection on `main`

Enabled on 2026-10-08: a pull request with one approval, stale approvals dismissed on new
commits, the `python` and `javascript` status checks passing on a branch that is up to date with `main`, review
conversations resolved, and no force pushes or deletion. Repository administrators (currently
`benoyj20`) can bypass these rules; use the bypass only for an explicitly agreed exception, such
as merging before the second contributor has repository access.
