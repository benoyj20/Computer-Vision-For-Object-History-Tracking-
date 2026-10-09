# 0004: Next.js front end for the web page

Date: 2026-10-09. Status: accepted. Refines the interface part of
[ADR 0002](0002-assistant-claude-sqlite-web.md).

## Context

ADR 0002 chose a web page backed by a Python service but left the front-end approach open.
The page needs a question box, and later the latest snapshot with labeled boxes and a per-object
timeline. Plain HTML, Gradio/Streamlit, and React/Next.js were considered.

## Decision

- Build the front end as a Next.js (App Router) and React app in TypeScript at `apps/web`,
  using the same toolchain and versions as AgentWatch: Next.js 16.3, React 19.3, Tailwind CSS 4,
  shadcn/ui for components, zod for response validation, ESLint, and Node's test runner.
- Manage it with pnpm 10 in a pnpm workspace at the repository root, next to the uv workspace.
- Keep the FastAPI history service as the only backend. It owns the SQLite history, the Claude
  API key, and Claude calls. The web app reaches it through `OBJHIST_API_URL` on the server and
  never opens the database or holds the key.

## Consequences

- Contributors need Node.js 22+ and pnpm 10 in addition to uv.
- CI gains a `javascript` job (lint, typecheck, test, build), required on `main` alongside
  `python`.
- The service's response models and the web app's zod schemas are a shared contract; changes
  update both sides and their tests together.
- Web changes follow `apps/web/AGENTS.md`, including browser checks at mobile, tablet, and
  desktop widths.
