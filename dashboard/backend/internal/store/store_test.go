package store

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"
)

func TestStorePersistsRunsAndEvents(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dashboard.sqlite")
	ctx := context.Background()
	database, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	run := Run{
		ID: "run-1", TemplateID: "synthetic-classification", Status: StatusQueued,
		SubmittedAt: now, UpdatedAt: now, ConfigJSON: map[string]any{"epochs": 2},
		ConfigPath: "config.yaml", ManifestPath: "manifest.json", ControlPath: "control.json",
		OutputDir: "runs/run-1", Launcher: "none", Devices: []int{},
	}
	if err := database.InsertRun(ctx, run); err != nil {
		t.Fatal(err)
	}
	eventID, err := database.RecordEvent(ctx, Event{
		RunID: "run-1", Type: "run_state", CreatedAt: now,
		Payload: map[string]any{"state": "queued"}, Reliable: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if eventID == 0 {
		t.Fatal("event id must be assigned")
	}
	if err := database.Close(); err != nil {
		t.Fatal(err)
	}

	database, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	loaded, err := database.GetRun(ctx, "run-1")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.TemplateID != run.TemplateID || loaded.Status != StatusQueued {
		t.Fatalf("unexpected run: %+v", loaded)
	}
	events, err := database.ListEvents(ctx, "run-1", 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Payload["state"] != "queued" {
		t.Fatalf("unexpected events: %+v", events)
	}
}

func TestStorePersistsWorkflows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dashboard.sqlite")
	ctx := context.Background()
	database, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	item := Workflow{
		ID: "wf-1", Version: 1, Name: "demo graph", ExecutionProfile: "schema-v1",
		SnapshotJSON: `{"version":1}`, SnapshotHash: "abc123",
		CreatedAt: now, UpdatedAt: now,
	}
	if err := database.InsertWorkflow(ctx, item); err != nil {
		t.Fatal(err)
	}
	if err := database.Close(); err != nil {
		t.Fatal(err)
	}

	database, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	loaded, err := database.GetWorkflow(ctx, "wf-1")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Name != item.Name || loaded.SnapshotHash != item.SnapshotHash {
		t.Fatalf("unexpected workflow: %+v", loaded)
	}
	if loaded.SnapshotJSON != item.SnapshotJSON {
		t.Fatalf("snapshot json not preserved: %q", loaded.SnapshotJSON)
	}
	listed, err := database.ListWorkflows(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(listed) != 1 || listed[0].ID != "wf-1" {
		t.Fatalf("unexpected workflow list: %+v", listed)
	}
	if listed[0].SnapshotJSON != "" {
		t.Fatal("workflow list must not return snapshot bodies")
	}
}

func TestStoreRejectsWorkflowWithoutSnapshot(t *testing.T) {
	database, err := Open(filepath.Join(t.TempDir(), "dashboard.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	if err := database.InsertWorkflow(context.Background(), Workflow{ID: "wf-empty"}); err == nil {
		t.Fatal("workflow without snapshot must be rejected")
	}
}

func TestStoreMigratesLegacyRunsTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.sqlite")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.Exec(`CREATE TABLE runs (
		id TEXT PRIMARY KEY, parent_run_id TEXT, template_id TEXT NOT NULL,
		status TEXT NOT NULL, submitted_at TEXT NOT NULL, updated_at TEXT NOT NULL,
		started_at TEXT, finished_at TEXT, config_json TEXT NOT NULL,
		config_path TEXT NOT NULL, manifest_path TEXT NOT NULL, control_path TEXT NOT NULL,
		output_dir TEXT NOT NULL, checkpoint_path TEXT, launcher TEXT NOT NULL,
		devices_json TEXT NOT NULL, process_id INTEGER,
		last_step INTEGER NOT NULL DEFAULT 0, last_epoch INTEGER NOT NULL DEFAULT 0,
		error_message TEXT)`); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if _, err := legacy.Exec(`INSERT INTO runs (
		id, template_id, status, submitted_at, updated_at, config_json, config_path,
		manifest_path, control_path, output_dir, launcher, devices_json
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		"legacy-run", "synthetic-classification", StatusQueued, now, now, "{}",
		"config.yaml", "manifest.json", "control.json", "runs/legacy-run", "none", "[]"); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	database, err := Open(path)
	if err != nil {
		t.Fatalf("legacy database must migrate: %v", err)
	}
	defer database.Close()
	run, err := database.GetRun(context.Background(), "legacy-run")
	if err != nil {
		t.Fatal(err)
	}
	if run.WorkflowID != "" || run.WorkflowSnapshotHash != "" || run.CompilerVersion != "" {
		t.Fatalf("unexpected workflow columns on legacy run: %+v", run)
	}
}

func TestStoreRecordsWorkflowRunAssociation(t *testing.T) {
	database, err := Open(filepath.Join(t.TempDir(), "dashboard.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	ctx := context.Background()
	now := time.Now().UTC()
	run := Run{
		ID: "run-wf", TemplateID: "", Status: StatusQueued,
		SubmittedAt: now, UpdatedAt: now, ConfigJSON: map[string]any{"workflow_id": "wf-1"},
		ConfigPath: "config.yaml", ManifestPath: "manifest.json", ControlPath: "control.json",
		OutputDir: "runs/run-wf", Launcher: "none", Devices: []int{},
		WorkflowID: "wf-1", WorkflowSnapshotHash: "hash-1",
	}
	if err := database.InsertRun(ctx, run); err != nil {
		t.Fatal(err)
	}
	loaded, err := database.GetRun(ctx, "run-wf")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.WorkflowID != "wf-1" || loaded.WorkflowSnapshotHash != "hash-1" {
		t.Fatalf("workflow association lost: %+v", loaded)
	}
	if err := database.SetCompilerVersion(ctx, "run-wf", "workflow-compiler/1"); err != nil {
		t.Fatal(err)
	}
	loaded, err = database.GetRun(ctx, "run-wf")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.CompilerVersion != "workflow-compiler/1" {
		t.Fatalf("compiler version not stored: %+v", loaded)
	}
}
