# XDL Workflow Editor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the blank-canvas node workflow editor to the real XDL Dashboard frontend without removing the existing run monitoring view.

**Architecture:** Keep workflow draft state in a dedicated Preact component and persist it to localStorage. Nodes use explicit `inputs` references and absolute canvas coordinates; the inspector edits node metadata and connections. Preflight runs entirely in the browser, while queueing maps the assembled workflow to an existing catalog template and submits through the current `/api/runs` API.

**Tech Stack:** Preact, TypeScript, lucide-preact, existing XDL Dashboard API, CSS.

---

### Task 1: Add workflow editor component

**Files:**
- Create: `dashboard/frontend/src/workflow_editor.tsx`
- Modify: `dashboard/frontend/src/main.tsx`

**Output:** A new `WorkflowEditor` view reachable from the sidebar, with an empty canvas, node palette, node selection, inspector, preflight, local draft persistence, and queue action.

**Test:** TypeScript build passes and the view renders with no API dependency failures when templates are unavailable.

### Task 2: Add canvas and inspector styles

**Files:**
- Modify: `dashboard/frontend/src/styles.css`

**Output:** Dark graphite workflow workspace with responsive palette/canvas/inspector layout, draggable node cards, connection paths, warning/error states, focus states, and mobile stacking.

**Test:** Vite build succeeds and screenshots at desktop and mobile widths show no clipped primary controls or horizontal page overflow.

### Task 3: Cover workflow logic

**Files:**
- Create: `dashboard/frontend/src/workflow_editor.test.ts`
- Modify: `dashboard/frontend/package.json` only if a focused test script is needed.

**Output:** Focused tests for required-node preflight, orphan inputs, node deletion cleanup, and local draft serialization.

**Test:** Run the focused test command if the existing frontend test runner is available; otherwise validate through TypeScript build and browser interaction checks.

### Task 4: Verify integration

**Files:**
- No new source files.

**Output:** Updated frontend build artifacts and verified dashboard behavior alongside the existing backend.

**Test:** Run `pnpm run build` in `dashboard/frontend`, `go test ./...` in `dashboard/backend`, and inspect the workflow view at desktop/mobile sizes. Verify adding nodes, dragging nodes, connecting nodes, deleting nodes, preflight errors, and queue submission feedback.
