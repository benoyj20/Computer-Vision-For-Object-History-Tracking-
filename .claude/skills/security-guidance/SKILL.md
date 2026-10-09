---
name: security-guidance
description: Review proposed code edits or a diff for exploitable security issues, especially model-weight loading, dataset downloads, file paths, LLM tool calls, and API keys. Use for a requested security review or security-sensitive changes.
license: Apache-2.0; see ../../licenses/security-guidance-Apache-2.0.txt
---

# Security Guidance

This is an Object History Tracking manual-review adaptation, authored 2026-10-08
from the AgentWatch version and informed by Anthropic's
[Security Guidance plugin](https://github.com/anthropics/claude-plugins-official/tree/315c4e48967d9541c29c3c656441dded353ca7aa/plugins/security-guidance).
It does not install the upstream edit, stop, or commit hooks, run background
reviews, or block commands automatically.

Read `AGENTS.md` and any instructions for the affected directory. Establish the
review scope from the user's request or the current diff, including intended new
files. Trace external input to its consumers before calling a pattern a finding.

## Review relevant boundaries

- Model weights: PyTorch `.pt` checkpoints are pickles and execute code on load.
  Load only weights the team trained or downloaded from the official Ultralytics
  release. Flag any path that loads a checkpoint from user input or a URL.
- Datasets and downloads: check archive extraction for path traversal, bound
  download sizes and timeouts, and keep Roboflow/Kaggle credentials out of code,
  logs, and committed dataset manifests.
- File paths: image, scenario, and history paths from configuration or the
  assistant must stay inside the configured data directory.
- LLM assistant: model output and user questions are untrusted. Tool calls must
  go through the allowlisted, read-only history-query functions with validated
  arguments. The model never writes SQL that is executed verbatim, runs shell
  commands, or modifies the history store.
- Storage: use parameterized queries; reject malformed records and conflicting
  identities.
- Deserialization and configuration: use `yaml.safe_load`, avoid `eval`/`exec`,
  and inspect CI workflows that consume untrusted input.
- Privacy: kitchen images can contain people. Keep real captures out of the
  repository and out of LLM prompts unless the team has agreed to share them.

## Report actionable findings

For each finding, give severity, file and line, the input-to-sink path, an example
trigger without real secrets, and the smallest effective fix. Distinguish verified
behavior from an unverified concern. Explain existing safeguards when they rule
out a suspected issue; do not report keyword matches alone as vulnerabilities.

If authorized to fix issues, add regression coverage for rejected inputs and
propagated failures, and run the applicable repository checks. Do not disable
checks or broaden permissions to make a finding disappear. Report review scope
and verification limits.
