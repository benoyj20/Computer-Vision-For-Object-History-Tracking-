# 0002: Assistant uses the Claude API, a SQLite history, and a web page

Date: 2026-10-08. Status: accepted.

## Context

The assistant answers questions such as "how long has the milk been out?" from the recorded
object history. It needs a language model that calls tools reliably, a history store it can
query safely, and an interface people in a shared kitchen can open without installing anything.

## Decision

- **Language model:** the Claude API through the official `anthropic` Python SDK (1.x), with
  model `claude-opus-5-5` as the default, configurable through `OBJHIST_LLM_MODEL`. The key is
  read from `ANTHROPIC_API_KEY` on the server only.
- **Tool use:** Claude receives only allowlisted, read-only query tools (for example "where is
  object X", "history of object X", "objects present at time T") with strict input schemas.
  The model never writes SQL; each tool runs a fixed parameterized query.
- **History store:** SQLite, a single file under `OBJHIST_DATA_DIR`. The tracker writes it; the
  assistant opens it read-only (`mode=ro`). Schema changes use ordered migrations.
- **Interface:** a web page served by a Python backend (FastAPI recommended), so the API key,
  database, and Claude calls stay server-side and the project stays in one language.

## Consequences

- Questions and the history records returned by tools are sent to Anthropic. Kitchen images are
  not sent unless both contributors agree and document it (see `AGENTS.md`).
- Answers depend on a network connection and an API key; the web page must show a clear error
  when either is missing.
- Unit tests stub Claude responses; real-model evaluation is recorded separately with model name
  and prompt version, per the [evaluation protocol](../evaluation-protocol.md).
- The `anthropic` dependency is added to `packages/assistant` when the assistant code is written,
  not before.
