# Fixed Training Node Workflow Implementation Plan

**Goal:** Add a light, fixed, top-to-bottom training workflow editor to the HTML design preview, with editable parameters and per-node Python drafts.

**Architecture:** Keep the existing monitor intact. Add isolated workflow CSS and JavaScript, an allowlisted preview route for these assets, and a small event bridge to create simulated queue records. Generated configuration follows existing schema v1 and target + params conventions. Browser code is never executed as Python.

## Tasks

1. Add a primary workflow view with eight ordered nodes: dataset, transforms, DataLoader, model, loss, optimizer, trainer and checkpoint. Keep node order fixed and connectors visible. Provide an outline and live summary.
2. Implement dataset/model/loss/optimizer choices, hyperparameters, explicit class synchronization, validation and generated schema-v1 configuration. Export a workflow JSON containing parameters, applied code and generated configuration.
3. Add expandable per-node Python editors with line numbers, draft/apply/reset states and parameter-change conflict warnings. Preserve drafts in localStorage with a visible fallback if storage fails.
4. Bridge validated workflows into the existing simulated experiment queue while preserving the complete snapshot and custom-code markers. Pending or stale code edits block queueing until resolved.
5. Verify actual Chromium rendering, seven viewport widths, node selection, parameter dependencies, code draft persistence, export content, validation and queue snapshots. Re-run existing monitoring checks and Go route tests.

## Acceptance

- User can configure dataset, model, loss and hyperparameters without dragging nodes.
- Node implementation snippets can be expanded and partially edited.
- Generated parameters never silently overwrite custom code.
## Completion

- Added a first-class `工作流编辑器` page with eight fixed, top-to-bottom nodes. The order is dataset, transforms, DataLoader, model, loss, optimizer, trainer, checkpoint.
- Added linked controls for dataset, class count, image size, model, loss, optimizer, scheduler, batch, epochs, precision, device, checkpoint policy and other schema-v1 fields.
- Added generated YAML preview and export using existing `target + params` conventions, plus local Python snippets per node. Snippets are preserved but explicitly never executed by the design preview.
- Added draft/apply/reset states, stale-code warnings after parameter changes, localStorage persistence, validation, export of the complete workflow snapshot and a simulated queue bridge into the existing monitor.
- Added same-origin `workflow.js` and `workflow.css` routes without weakening the existing CSP.
- Chromium verification passed 20 areas across 10 viewport widths, including class synchronization, model/loss/optimizer switching, code persistence, stale review, queue snapshot inspection, validation and responsive layout. No browser console errors.
- `go test ./...` passes in `dashboard/backend`.
