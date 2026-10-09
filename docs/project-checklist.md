# Project and testing checklist

Status snapshot: October 8, 2026. The [problem statement and plan](problem-statement-and-plan.md)
describes the targets; this checklist records what actually exists and what has evidence.

A checked item means the work or verification was completed. An unchecked item still needs
evidence, even when related code exists. Record dates and artifact or test references when
checking items off; keep manual checks distinct from automated tests.

## Current working flow

Repository setup only. No detector has been trained, and tracking, history, the assistant, and
evaluation are not implemented. The shared `FrameDetections` contract between the detector and
the tracker exists and is tested.

## Completed: repository setup (2026-10-08, branch `chore/project-setup`)

- [x] uv workspace with `core`, `detection`, `tracking`, `assistant`, and `evaluation` packages.
- [x] Selectable PyTorch builds: `--extra cu130` (NVIDIA) or `--extra cpu` (CI, CPU-only);
  see [ADR 0001](adr/0001-workspace-layout-and-pytorch-builds.md).
- [x] Shared contracts `BoundingBox`, `Detection`, `FrameDetections` with validation of box
  geometry, finite values, confidence range, image bounds, and timezone-aware capture times.
- [x] GPU environment check `python -m objhist_detection.env_check --require-cuda`.
- [x] Ruff lint/format and pytest configuration, with a `gpu` marker for CUDA-only tests.
- [x] GitHub Actions `python` job, PR template, CODEOWNERS, and contributor rules
  (`AGENTS.md`, `CONTRIBUTING.md`).
- [x] Repo-local Claude Code skills and agents copied from AgentWatch and adapted.
- [x] Starter Next.js web app in `apps/web` with web rules, `OBJHIST_API_URL` validation and tests,
  and a `javascript` CI job (2026-10-09, [ADR 0004](adr/0004-nextjs-front-end.md)).

### Setup verification (2026-10-08, Windows 11, RTX 5070 Laptop GPU 8 GB, driver 617.42)

- [x] `uv sync --all-packages --extra cu130`: torch 2.14.1+cu130, ultralytics 8.4.174,
  Python 3.12.12; `env_check --require-cuda` reported CUDA available.
- [x] Pretrained `yolo26n.pt` predicted on the Ultralytics sample image on `cuda:0`
  (about 15 ms inference after warm-up). Manual check, not an automated test.
- [x] `uv run pytest`: 17 passed (16 unit tests and 1 GPU test) with the CUDA build.
- [x] CI install path reproduced locally in a separate environment:
  `uv sync --all-packages --extra cpu --frozen`, Ruff checks passed, 16 passed and 1 GPU test
  skipped.
- [ ] Observe a passing remote CI run on GitHub.
- [ ] Setup reproduced on Justin's machine.

## Milestone 1: dataset (provided by Benoy)

- [ ] Choose the class list with the scenarios in mind (including items such as milk and dish
  soap) and record it in the [data card](data-card.md).
- [ ] Provide the snapshot sequences in the format in [data/README.md](../data/README.md),
  with a timezone-aware capture time for every image.
- [ ] Download the Roboflow dataset; record version, license, and hash.
- [x] Decide whether the Kaggle data is usable under its rules and worth re-annotating
  (2026-10-09: Benoy confirmed training is permitted; recorded in the data card).
- [x] Kaggle auto-labeling pipeline (2026-10-09, branch `feature/kaggle-auto-labels`):
  frozen class-stratified split, YOLOE box proposals with review flags, HTML review page, and
  YOLO-format export that requires a verdict on every validation and test image. Unit tests in
  `packages/detection/tests/test_kaggle_*.py`; proposals run on all 5,559 labeled images.
- [x] Review of the Kaggle proposals (all validation/test, flagged training, 300-image training
  spot-check), done by Claude visual review rather than a person (2026-10-09); error rates and
  audit in the data card. A human re-check of the test split is still recommended.
- [ ] Choose a labeling tool and label boxes on the provided images.
- [ ] Freeze splits grouped by capture session.

**Exit test:** a versioned dataset with a documented class list, split lists, and hashes that
trains with `make train`.

## Milestone 2: detector (lead: Benoy)

- [ ] Zero-shot COCO-pretrained YOLO26 baseline on the held-out split.
- [ ] Fine-tune YOLO26; record the command, seed, hardware, and metrics for each run.
  First run on `kaggle-kitchenware-v1` (2026-10-09): test mAP50 0.984, mAP50-95 0.961 on
  single-object close-ups; details in the data card. Still needs in-domain counter data.
- [ ] Compare model sizes (n/s) for accuracy versus speed on the 8 GB GPU.
- [ ] Convert predictions to `FrameDetections` with tests for coordinate conversion.
- [ ] Choose and record the confidence threshold on the validation split.

**Exit test:** mAP50 and mAP50-95 on the frozen test split, compared with the zero-shot baseline.

## Milestone 3: tracking and history (lead: Justin)

- [ ] Association by position and DINOv2/CLIP appearance embeddings across snapshots.
- [ ] Compare DINOv2 and CLIP embeddings on the validation scenarios; record the choice.
- [ ] Added, removed, and moved events with placement intervals between snapshots.
- [ ] Rule that keeps missed detections and occlusion distinct from removal.
- [ ] SQLite history store with a documented schema and ordered migrations.
- [ ] Unit tests for duplicates, occlusion, swaps, and empty frames.

**Exit test:** synthetic `FrameDetections` sequences produce the expected event log, including
the edge cases above.

## Milestone 4: generative AI assistant (joint)

- [ ] Allowlisted, read-only history query tools with strict input schemas over SQLite.
- [ ] Claude tool-use loop with `claude-opus-5-5`; refusal and API errors shown clearly.
- [ ] Answers cite history records and say "unknown" when evidence is missing.
- [ ] FastAPI history service exposing the assistant and read-only history endpoints.
- [ ] Next.js question page with loading, error, unreachable-service, and missing-key states.
- [ ] Latest snapshot with labeled boxes and a per-object timeline, with text alternatives.
- [ ] Tests with stubbed model responses, including rejected tool calls.

**Exit test:** a fixed question set over a recorded history answered correctly, with no
invented times or locations.

## Milestone 5: evaluation (joint)

- [ ] Freeze the [evaluation protocol](evaluation-protocol.md) thresholds before evaluating.
- [ ] Provide scripted scenario sequences for every listed condition and annotate their
  ground-truth changes.
- [ ] Implement event matching and metrics with unit tests.
- [ ] Run the IoU-only baseline, DINOv2, and CLIP association on the same scenarios.

**Exit test:** a results table of event precision/recall per condition, identity switches,
placement-interval coverage, and assistant accuracy, reproducible from recorded commands.

## Milestone 6: final report and demo

- [ ] End-to-end demo from new snapshots to an assistant answer.
- [ ] Final report with methods, results, failures, and limitations.

## Decisions to settle

- [x] LLM provider: Claude API, default model `claude-opus-5-5` (2026-10-08,
  [ADR 0002](adr/0002-assistant-claude-sqlite-web.md)).
- [x] History store: SQLite, opened read-only by the assistant (2026-10-08, ADR 0002).
- [x] Interface: a web page served by a Python backend (2026-10-08, ADR 0002).
- [x] Appearance features: deep embeddings, DINOv2 by default with CLIP compared on validation
  scenarios (2026-10-09, [ADR 0003](adr/0003-appearance-embeddings-and-provided-dataset.md)).
- [x] Capture hardware: out of scope; Benoy provides the image dataset (2026-10-09, ADR 0003).
  The snapshot interval is whatever the provided sequences use, recorded in the data card.
- [x] Web page stack: Next.js front end and FastAPI history service (2026-10-09,
  [ADR 0004](adr/0004-nextjs-front-end.md)).
- [x] `main` branch protection enabled (2026-10-08): PR with 1 approval, stale approvals
  dismissed, `python` and `javascript` checks required on an up-to-date branch, conversations resolved, no force
  pushes or deletion. Admin bypass stays available until Justin is a collaborator.
- [ ] Add Justin as a repository collaborator and to `.github/CODEOWNERS`.
