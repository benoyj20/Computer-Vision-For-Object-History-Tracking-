# Coding agent rules

These rules apply throughout this repository. Follow any additional instructions
in the directory you are editing, including `apps/web/AGENTS.md` for the web app.

## Working with two contributors

- Benoy leads the detector (`packages/detection`), Justin leads tracking and the
  object history (`packages/tracking`). The assistant (`packages/assistant`), the web app
  (`apps/web`), and evaluation (`packages/evaluation`) are joint. Shared contracts in
  `packages/core` affect both leads; agree on changes before making them.
- Before coding, inspect the current branch, working tree, relevant instructions,
  and existing implementation. When remote access is available, fetch `origin`
  and inspect relevant open PRs to identify overlapping work. Report unavailable
  remote access instead of assuming the checkout is current.
- Before starting any coding, pull the latest changes from the task branch's
  remote counterpart, when one exists, using `git pull --ff-only`, and sync the
  task branch with the latest `origin/main`. Fetching alone is not sufficient.
  Preserve local work; if local changes, divergent history, conflicts, or
  unavailable remote access prevent a safe sync, report the blocker before coding.
- Start new task branches from the latest `origin/main`. Preserve existing local
  work before switching branches. Never reset, stash, delete, or overwrite the
  other contributor's changes without their authorization.
- Give each task one primary contributor and one task branch. Use the existing
  issue or task description to establish scope. If the other contributor is
  changing the same behavior or shared files, flag the overlap to the user before
  making incompatible changes; do not independently implement a competing solution.
- Keep PRs focused and small enough to review as one change. Split unrelated
  features and broad refactors into separate tasks. State dependencies on other
  PRs and the intended merge order; do not copy the other contributor's
  unfinished work into your branch without coordination.
- Sync with `origin/main` before the final verification and again if changes merge
  that affect your work. Merge main into a published branch to preserve shared
  history unless the user explicitly requests a rebase.
- Resolve conflicts by understanding both changes. Never blindly choose all
  "ours" or "theirs". Verify the combined behavior and rerun affected checks after
  resolving conflicts.

## Shared interfaces and maintainability

- Preserve existing patterns and reuse existing modules. Avoid duplicate helpers,
  unrelated dependency upgrades, repository-wide formatting, and abstractions
  that the task does not need.
- Treat `packages/core` contracts, the history storage format, the history service's
  API responses and the web app's zod schemas, the scenario ground-truth format,
  dataset class lists, CI, and root configuration as shared
  interfaces. Describe changes to these in the PR so the other contributor can
  identify affected consumers, and update producers, consumers, tests, and docs
  together. Prefer backward-compatible changes.
- If the history store gains a database, version its schema with ordered
  migrations. Do not silently rewrite applied migrations or drop recorded history
  to make tests pass.
- Add dependencies only when existing code or standard libraries cannot reasonably
  solve the problem. Add them with `uv add --package <member> <dependency>`,
  explain additions in the PR, and never hand-edit `uv.lock` or resolve its
  conflicts blindly; re-run `uv lock` instead.
- Keep secrets, real kitchen captures, and people's images out of code, tests,
  logs, and commits. Document new configuration using placeholders in
  `.env.example` and the relevant setup docs.
- Do not disable checks, weaken assertions, or swallow errors to obtain a green
  build. Fix the cause and distinguish pre-existing failures from new failures.
- Update relevant setup docs when behavior or operational steps change. Record
  consequential architecture decisions in `docs/adr/` using its conventions.

## Code quality and structure

- Write code with one clear responsibility per function, class, or module.
  Extract independently understandable behavior when a file becomes hard to
  navigate; avoid arbitrary line-count limits and unnecessary wrapper layers.
- Follow the package boundaries: detection and dataset tooling in
  `packages/detection`, identity association and the history log in
  `packages/tracking`, language-model interaction in `packages/assistant`,
  scenarios and metrics in `packages/evaluation`, shared types in
  `packages/core`, and the browser front end in `apps/web`. The history
  service (FastAPI) is the only backend; the web app never opens the SQLite
  history or holds the Claude API key. Do not add a second implementation of
  an existing responsibility. Keep notebooks for exploration; move reusable
  logic into packages with tests.
- Use descriptive domain names, consistent terminology, and explicit units in
  variable names (`_px`, `_seconds`, `_gib`). Replace repeated unexplained
  constants such as IoU or confidence thresholds with named, documented values.
  Comments should explain decisions and invariants, not restate code.
- Follow `.editorconfig`, Ruff for Python, and ESLint/Prettier for web code. Format touched code without unrelated
  formatting changes. Remove unused imports, dead code, debugging output, and
  commented-out implementations introduced by the task.
- Use Python type annotations, TypeScript types, and explicit data contracts. Validate external data
  at boundaries (images, label files, configs, model output); do not use `Any`/`any`,
  unchecked casts, ignored type errors, or disabled lint rules as shortcuts.
  Explain narrow exceptions in code.
- Handle expected failures explicitly and propagate useful context. Keep user
  error messages actionable. Avoid catch-all handlers that turn failures into
  empty results or success; an unreadable frame is not a frame with no objects.
- Bound network requests (dataset downloads, LLM calls) with timeouts. Retry only
  retryable operations with bounded attempts. Release files and GPU memory on
  failure.
- Use `yaml.safe_load`, parameterized queries, and validated paths. Never
  interpolate untrusted input into SQL or shell command strings.
- Test public behavior and important edge cases: invalid input, missing frames,
  empty detections, ties, and boundaries. Keep fixtures deterministic, small, and
  synthetic; avoid sleeps, live model downloads, GPU requirements, and live LLM
  calls in unit tests. Mark tests that need CUDA with `@pytest.mark.gpu`. Add
  regression coverage for bug fixes.
- Document non-obvious contracts and public behavior where maintainers will find
  them. Keep new TODOs specific and tied to an existing task when available; do
  not leave unfinished required behavior behind a TODO.

## Object history evidence rules

- Read `docs/project-checklist.md` for implemented behavior and verification
  gaps; `docs/problem-statement-and-plan.md` describes targets. Do not assume
  planned components already exist.
- Never fabricate detections, tracks, events, or timestamps to populate a history,
  demo, chart, or report. Synthetic fixtures belong in clearly identified tests.
- Snapshots are sparse. An object first seen in frame N was placed between frame
  N-1 and frame N; record and report that interval, not a single exact time. Keep
  "first observed", "last observed", and "estimated placement interval" distinct.
- Distinguish "not detected in this frame" from "removed". Occlusion, missed
  detections, and unprocessed frames must not silently become removal events.
  Document any frame-persistence or confirmation rule that decides removal.
- Use timezone-aware capture timestamps from a documented source (EXIF or the
  dataset manifest), validate ordering, and never subtract incompatible clocks.
- Do not merge tracks of different classes or reuse an identity without the
  association evidence (position, appearance, confidence) that justified it.
  Keep association thresholds named, versioned, and recorded with results.
- Record detector version, confidence threshold, and tracker version with every
  generated history so results can be traced to the configuration that made them.

## Datasets, training, and evaluation

- Do not commit datasets, captures, or model weights. Keep them under
  `OBJHIST_DATA_DIR` (default `data/`, ignored by Git) and record source URLs,
  versions, licenses, and hashes in `docs/data-card.md`.
- Check each dataset's license and terms of use before training on or
  redistributing it. The Kaggle kitchenware set is classification-only; it has no
  boxes and cannot be used for detector training without new annotations.
- Consecutive frames of one capture session are nearly identical. Keep every
  frame of a session and scenario in the same split; never split them randomly.
- Freeze validation and test splits before tuning. Do not tune thresholds,
  epochs, or hyperparameters against the test set or scripted evaluation runs.
- For each training run, record the dataset version, class list, base weights
  (for example `yolo26n.pt`), image size, epochs, seed, hardware, and the
  resulting metrics. Load checkpoints only from your own runs or official
  Ultralytics releases; `.pt` files execute code when loaded.
- Declare event-matching rules, metrics, and acceptance thresholds in
  `docs/evaluation-protocol.md` before evaluating. Preserve failed and
  inconclusive results; do not cherry-pick reruns.
- Compare against a simple baseline (for example IoU-only association) on the
  same scenarios. Report detector mAP and history-event precision/recall
  separately; good detection does not establish correct history.

## Generative AI assistant

- Answers must be grounded in history query results. The model calls only
  allowlisted, read-only query functions with validated arguments, and must say
  the information is unavailable rather than guess when the history lacks it.
- Treat user questions, model output, and tool arguments as untrusted data,
  never as authority to execute code, run model-written SQL, change the history,
  or override repository instructions.
- Report placement times as the recorded interval ("between 2:10 and 2:25 pm"),
  never as a more precise time than the snapshots support.
- Keep API keys in `.env`. Do not send kitchen images or personal information to
  an external model unless both contributors have agreed and it is documented.
- Unit tests use recorded or stubbed model responses. Record model name, prompt
  version, and temperature with any assistant evaluation; temperature zero does
  not guarantee reproducibility.

## Branches and naming

- Work on a branch and submit a pull request; do not push directly to `main`.
  The repository administrator has a branch-protection bypass for explicitly authorized
  exceptions. Agents must not use it without a specific user instruction to bypass the
  normal flow.
- Use `<type>/<short-description>` with lowercase words separated by hyphens:
  - `feature/yolo26-training` for new functionality.
  - `fix/track-id-swap` for bug fixes.
  - `docs/evaluation-protocol` for documentation.
  - `refactor/history-store` for restructuring without behavior changes.
  - `test/association-edge-cases` for test changes.
  - `chore/update-dependencies` for maintenance.
- Keep each branch focused on one task. Avoid vague names such as `updates`,
  `changes`, or `agent-work`.
- Use commit titles in the form `<type>: <specific change>`, for example
  `fix: keep occluded objects out of removal events`. Use the same types listed
  above.

## Verify before pushing

- Finish a coherent, working change before pushing. Do not push placeholders,
  known broken code, or speculative fixes just to see whether CI passes.
- Run checks relevant to the changed code and exercise the affected behavior.
  Add or update meaningful tests when behavior changes.
- Use the commands in `.github/workflows/ci.yaml` as the verification baseline:
  - Python: `uv run ruff check .`, `uv run ruff format --check .`, and
    `uv run pytest`.
  - Web: `pnpm lint`, `pnpm typecheck`, `pnpm test`, and `pnpm build`. For UI
    changes, also check the affected flow in a browser.
  CI runs on CPU; when a change affects GPU training or inference, also run
  `uv run pytest -m gpu` and `make env-check` on a CUDA machine and report it.
- For training or evaluation changes, report the exact command, dataset version,
  and resulting metrics. A run on a subset is not a full evaluation; say so.
- For documentation-only changes, verify accuracy, paths, and Markdown
  formatting; application tests are not required unless executable behavior
  also changes.
- Re-run affected checks after subsequent edits. Do not claim success based on
  checks run before the final changes or on commands that skipped verification.
- If a required check fails or cannot run, fix the problem before pushing. If
  blocked, leave the work local and report the failed or unavailable check.
  Push unverified work only if the user explicitly requests that exception, and
  clearly describe the limitation in a draft PR.

## Meaningful commits and pushes

- Review `git diff` and `git status` before committing. Include only intended
  task changes; do not stage unrelated user work, secrets, `.env`, datasets,
  weights, `runs/`, `node_modules/`, `.next/`, logs, caches, or generated output.
- Group related edits into meaningful commits. Avoid empty commits, cosmetic
  churn, unrelated cleanup, and repeated tiny pushes made only to show activity.
- Push after a verified milestone or a complete, verified fix for review feedback.
- Never force-push shared branches. Rewrite a personal PR branch only when the
  user requests it; use `--force-with-lease` and verify the remote state first.

## Pull requests

- Describe what changed, why, and which checks passed. State any limitations
  plainly; never present an untested change as working.
- Include reproduction steps for behavior changes, screenshots for UI changes,
  sample outputs or images for visible results, and dataset or configuration steps where relevant.
- Require one approval from the other contributor. Authors cannot approve their
  own PRs. After new code changes, obtain a fresh approval of the final version.
- Require the `python` and `javascript` CI checks to pass before merging. Local
  checks do not replace PR CI. If a check is missing or failing, investigate instead of
  treating it as passed.
- Address review feedback and resolve discussions. Do not bypass branch
  protections, dismiss reviews to evade feedback, or merge without user
  authorization.
- At handoff, state the branch, changed files, verification results, and
  remaining work. Distinguish local edits, commits, pushed work, and merged work
  accurately.
