<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# Web code and UI rules

These rules supplement the repository-root `AGENTS.md`. Adapted from AgentWatch on 2026-10-09.

## Structure and data flow

- The web app is a front end only. The FastAPI history service owns the SQLite history,
  the Claude API key, and all Claude calls. The browser never receives the API key, raw SQL,
  or database paths, and the web app never opens the SQLite file directly.
- Keep App Router pages and route handlers focused on routing and orchestration. Put reusable
  views in `components/`, UI primitives in `components/ui/` (shadcn/ui, configured in
  `components.json`), and data/schema utilities in `lib/`. Extend an existing abstraction
  before creating a parallel one.
- Keep server-only data access and environment variables on the server (`lib/server-config.ts`
  imports `server-only`). Add `"use client"` only where interactivity requires it; keep client
  boundaries small and pass serializable, validated data across them.
- Validate every history-service response with zod schemas in `lib/` before rendering.
  Keep these schemas in step with the service's response models; a mismatch is an error,
  not something to coerce silently.
- Keep state minimal and scoped to its owner. Derive values rather than storing duplicate
  state. Clean up effects, timers, and subscriptions; prevent a stale response from replacing
  the answer to a newer question. Avoid unnecessary polling and duplicate requests.

## Honest history display

- Show only what the history service returned. Never invent object positions, identities,
  or times in the UI, and never fill gaps with placeholder data outside clearly marked tests.
- Show placement times as intervals ("between 2:10 and 2:25 pm") with the time zone, as the
  service reports them. Do not round an interval into a single exact time.
- Distinguish "not seen in the latest snapshot" from "removed", and "unknown" answers from
  errors. Show the snapshot time the answer is based on.
- Assistant answers are model output: render them as text, never as raw HTML.

## Responsive layouts and accessibility

- Build layouts from narrow screens upward using Tailwind and the tokens in `app/globals.css`.
  Prefer flexible grids, wrapping, and content-driven sizing over fixed widths or absolute
  positioning for primary layout. Box overlays on snapshots may use absolute positioning
  inside a container that scales with the image.
- Verify affected flows at approximately 360–390 px mobile, 768 px tablet, 1280–1440 px
  desktop, and 200% browser zoom. Record which viewports were actually checked in the PR.
- Avoid page-level horizontal overflow. Wide timelines and tables may scroll inside labeled
  containers. Wrap long identifiers and answers.
- Provide text alternatives for essential visual evidence: a snapshot with boxes needs a
  list of the objects shown; a timeline needs an accessible table or list.
- Use semantic headings, landmarks, buttons, links, form labels, and table headers. Keep focus
  visible, touch targets at least 44 by 44 CSS pixels, and contrast readable in light and dark
  themes. Do not communicate status through color alone. Respect reduced-motion preferences.

## Complete interaction states

- Implement loading, empty, error, success, and unavailable states. The page must say clearly
  when the history service is unreachable or the assistant is not configured (for example, a
  missing API key on the service).
- Prevent duplicate submissions while a question is pending, show progress for slow answers,
  and let the user retry after a failure.
- Use stable IDs as list keys. Do not download the whole history to display one object.
- Verify changed interactions in a browser, including keyboard navigation. Unit tests alone
  do not verify layout or usability. If browser verification is unavailable, report that
  limitation before pushing.
