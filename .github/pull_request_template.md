## Change and purpose

<!-- Explain the problem and resulting behavior. Link the task/issue if available. -->

## Scope and dependencies

<!-- List affected packages, shared contracts (packages/core, history format,
scenario format, class list), overlapping work, and dependent PRs/merge order. -->

## Verification

<!-- List commands actually run and their results. State skipped, unavailable,
or failing checks explicitly. For behavior changes, include reproduction steps. -->

| Check or manual scenario | Result / evidence |
| --- | --- |
| `uv run ruff check .` | |
| `uv run ruff format --check .` | |
| `uv run pytest` | |
| GPU checks (`make env-check`, `uv run pytest -m gpu`), if relevant | |
| Web: `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`, if relevant | |

## UI evidence (if applicable)

<!-- Screenshots and widths actually checked: mobile 360–390 px, tablet 768 px,
desktop 1280–1440 px, and 200% zoom. Describe keyboard checks and loading, empty, error,
and unavailable states. Write N/A for non-UI changes. -->

## Model, data, and evaluation impact (if applicable)

<!-- Dataset version and class list, base weights, training command, seed,
hardware, and metrics (mAP, event precision/recall). Note whether the evaluation
protocol or thresholds changed and why. Write N/A when there is no impact. -->

## Review checklist

- [ ] Changes are focused; datasets, weights, captures, and secrets are excluded.
- [ ] Relevant local checks passed; limitations are stated above.
- [ ] The `python` and `javascript` CI checks passed on the final version.
- [ ] The other contributor approved the final code changes.
- [ ] Review conversations are resolved and affected documentation is updated.
