# XDL Dashboard Redesign Implementation Plan

> Execute in the current session using the loaded frontend-design and ui-ux-pro-max guidance.

**Goal:** Replace the crowded HTML mockup with a usable, high contrast light training workbench.

**Architecture:** Keep the design preview independent from real training state. Use native HTML, inline CSS and a same-origin external JavaScript file. Preserve the existing backend API and production frontend.

**Tech Stack:** HTML, CSS, browser JavaScript, existing Go preview handler.

---

## Design direction

Use neutral white surfaces, restrained blue accents and clear Chinese labels. Keep navigation narrow, place the selected experiment at the top, show four meaningful metrics, and give the training curve and pipeline dedicated space. Separate loss and accuracy axes through chart tabs. Label all records and values as simulated; FakeData is not official CIFAR-10.

## Task 1: Rebuild the preview

- Replace `dashboard/design/index.html` with a responsive workbench shell, experiment browser, overview, pipeline, logs, configuration and artifact views.
- Create `dashboard/design/design.js` for coherent selection state, chart interactions, accessible dialogs and training controls.
- Make every visible action either navigate, update preview state or produce a tangible preview result. Include empty search, paused, queued, completed and failed states.
- Support keyboard navigation, visible focus, reduced motion, 1440/1280/768/390 px layouts and Escape dismissal.

## Task 2: Preserve preview delivery

- Extend the exact allowlist in `dashboard/backend/internal/api/server.go` to serve `design.js` from the existing preview root.
- Keep the current script-src policy; do not enable unsafe inline script.
- Add focused route coverage for HTML, JavaScript, missing and traversal paths.

## Task 3: Verify and refine

- Run `node --check dashboard/design/design.js`.
- Run `XDL_PUNCT_PATHS=dashboard/design/index.html /root/miniconda/envs/xdl/bin/python scripts/normalize_punctuation.py`, and separately check other edited source files.
- Run `go test ./...` in `dashboard/backend`.
- Build the existing Go service and restart the same preview port after checking running training state.
- Use local Chromium to inspect rendered screenshots and exercise navigation, search, run selection, metrics, pipeline, dialog validation, queueing and pause/resume/stop.
- Confirm no console errors, no page-level horizontal overflow and no external requests.

## Completion

- Rebuilt the preview and moved recent experiment selection into the sidebar to keep the training view compact.
- Go tests pass, including the exact preview-asset allowlist and preserved same-origin script policy.
- Chromium passed 14 interaction groups and 7 viewport widths (360-1920 px), including downloads, modal validation, failed/queued/completed states, checkpoint recovery lineage and keyboard tabs.
- Fixed same-page hash navigation discovered during browser verification.
- Confirmed Windows can fetch the preview with HTTP 200. The existing service remains on port 8765.
- Saved final desktop, node-detail and experiment-creation screenshots under the ignored local dashboard output directory.
