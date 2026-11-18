package queue

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/runner"
	"github.com/xlong-c/xdl-dashboard/internal/store"
	"github.com/xlong-c/xdl-dashboard/internal/templates"
	"github.com/xlong-c/xdl-dashboard/internal/workflow"
)

const testWorkflowCatalog = `{
  "version": 1,
  "limits": {"max_nodes": 64, "max_edges": 256, "max_params_depth": 6,
             "max_params_bytes": 65536, "max_snapshot_bytes": 262144,
             "max_name_length": 120, "max_id_length": 64},
  "profiles": {"schema-v1": "schema", "dag-runtime": "runtime"}
}`

const testTemplateCatalog = `{
  "version": 1,
  "templates": [
    {
      "id": "demo",
      "definition": "templates/demo.yaml",
      "source_config": "config/demo.yaml",
      "allowed_overrides": ["trainer.max_epochs"],
      "requires_gpu": false
    }
  ]
}`

const testTemplateDefinition = `version: 1
id: demo
label: Demo template
summary: demo
source_config: config/demo.yaml
fields:
  - path: trainer.max_epochs
    type: integer
    label: epochs
    default: 1
`

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newTestManager(t *testing.T) (*Manager, *store.Store, config.Config) {
	t.Helper()
	root := t.TempDir()
	catalogPath := filepath.Join(root, "dashboard", "templates", "catalog.json")
	writeFile(t, catalogPath, testTemplateCatalog)
	writeFile(t, filepath.Join(root, "dashboard", "templates", "demo.yaml"), testTemplateDefinition)
	writeFile(t, filepath.Join(root, "config", "demo.yaml"), "config_version: 1\n")
	workflowCatalogPath := filepath.Join(root, "xdl", "workflow", "node_catalog.json")
	writeFile(t, workflowCatalogPath, testWorkflowCatalog)

	catalog, err := templates.Load(catalogPath, root, filepath.Join(root, "data"))
	if err != nil {
		t.Fatalf("load templates: %v", err)
	}
	workflowCatalog, err := workflow.LoadCatalog(workflowCatalogPath)
	if err != nil {
		t.Fatalf("load workflow catalog: %v", err)
	}
	database, err := store.Open(filepath.Join(root, "dashboard.sqlite"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	cfg := config.Default()
	cfg.ProjectRoot = root
	cfg.DataRoot = filepath.Join(root, "data")
	cfg.OutputRoot = filepath.Join(root, "runs")
	cfg.PythonExecutable = filepath.Join(root, "missing-python")
	cfg.MetricsInterval = 2 * time.Second
	hub := events.NewHub(64)
	jobRunner := runner.New(cfg, database, hub)
	manager := New(cfg, database, catalog, workflowCatalog, jobRunner, hub, nil)
	return manager, database, cfg
}

func insertWorkflow(t *testing.T, database *store.Store, fixture string) store.Workflow {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "templates", "workflows", fixture))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	catalog, err := workflow.LoadCatalog(filepath.Join("..", "..", "..", "..", "xdl", "workflow", "node_catalog.json"))
	if err != nil {
		t.Fatalf("load workflow catalog: %v", err)
	}
	snapshot, issues := workflow.Prepare(raw, catalog)
	if workflow.HasErrors(issues) {
		t.Fatalf("fixture rejected: %s", workflow.ErrorMessages(issues))
	}
	snapshot.ID = "wf-test"
	canonical, err := snapshot.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	item := store.Workflow{
		ID:               snapshot.ID,
		Version:          snapshot.Version,
		Name:             snapshot.Name,
		ExecutionProfile: snapshot.ExecutionProfile,
		SnapshotJSON:     string(canonical),
		SnapshotHash:     workflow.HashBytes(canonical),
		CreatedAt:        now,
		UpdatedAt:        now,
	}
	if err := database.InsertWorkflow(context.Background(), item); err != nil {
		t.Fatal(err)
	}
	return item
}

func TestSubmitWorkflowFreezesSnapshotAndManifest(t *testing.T) {
	manager, database, cfg := newTestManager(t)
	item := insertWorkflow(t, database, "synthetic-cpu-demo.json")

	run, err := manager.SubmitWorkflow(context.Background(), item.ID, WorkflowSubmitRequest{Name: "wf run"})
	if err != nil {
		t.Fatalf("submit workflow: %v", err)
	}
	if run.WorkflowID != item.ID {
		t.Fatalf("run workflow id = %q", run.WorkflowID)
	}
	if run.WorkflowSnapshotHash != item.SnapshotHash {
		t.Fatalf("run snapshot hash = %q, want %q", run.WorkflowSnapshotHash, item.SnapshotHash)
	}
	if run.TemplateID != "" {
		t.Fatalf("workflow run must not claim a template id: %q", run.TemplateID)
	}

	snapshotPath := filepath.Join(cfg.OutputRoot, run.ID, "workflow.json")
	raw, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("read frozen snapshot: %v", err)
	}
	if workflow.HashBytes(raw) != item.SnapshotHash {
		t.Fatal("frozen snapshot hash does not match the workflow record")
	}

	manifestRaw, err := os.ReadFile(run.ManifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(manifestRaw, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest["kind"] != WorkflowManifestKind {
		t.Fatalf("manifest kind = %v", manifest["kind"])
	}
	if manifest["workflow_snapshot_hash"] != item.SnapshotHash {
		t.Fatalf("manifest hash = %v", manifest["workflow_snapshot_hash"])
	}
	if manifest["execution_profile"] != "schema-v1" {
		t.Fatalf("manifest profile = %v", manifest["execution_profile"])
	}
	if !strings.HasSuffix(manifest["workflow_path"].(string), "workflow.json") {
		t.Fatalf("manifest workflow_path = %v", manifest["workflow_path"])
	}
	if manifest["data_root"] != cfg.DataRoot {
		t.Fatalf("manifest data_root = %v", manifest["data_root"])
	}

	stored, err := database.GetRun(context.Background(), run.ID)
	if err != nil {
		t.Fatalf("load run: %v", err)
	}
	if stored.WorkflowID != item.ID {
		t.Fatalf("stored workflow id = %q", stored.WorkflowID)
	}
	if stored.ConfigJSON["workflow_name"] != item.Name {
		t.Fatalf("run config lost workflow name: %+v", stored.ConfigJSON)
	}
}

func TestSubmitWorkflowRejectsUnknownWorkflow(t *testing.T) {
	manager, _, _ := newTestManager(t)
	if _, err := manager.SubmitWorkflow(context.Background(), "wf-missing", WorkflowSubmitRequest{}); err == nil {
		t.Fatal("unknown workflow must be rejected")
	}
}

func TestSubmitWorkflowRejectsDistributedLauncherWithoutDevices(t *testing.T) {
	manager, database, _ := newTestManager(t)
	item := insertWorkflow(t, database, "synthetic-cpu-demo.json")
	if _, err := manager.SubmitWorkflow(context.Background(), item.ID, WorkflowSubmitRequest{Launcher: "torchrun"}); err == nil {
		t.Fatal("distributed launcher without devices must be rejected")
	}
}

func TestSubmitWorkflowUsesSnapshotDeviceDefaults(t *testing.T) {
	manager, database, _ := newTestManager(t)
	item := insertWorkflow(t, database, "synthetic-cpu-demo.json")
	if _, err := manager.SubmitWorkflow(context.Background(), item.ID, WorkflowSubmitRequest{}); err != nil {
		t.Fatalf("cpu workflow must submit without devices: %v", err)
	}
}

func TestTemplateSubmitStillWorks(t *testing.T) {
	manager, _, _ := newTestManager(t)
	run, err := manager.Submit(context.Background(), SubmitRequest{
		TemplateID: "demo",
		Parameters: map[string]any{"trainer.max_epochs": 2},
		Launcher:   "none",
	})
	if err != nil {
		t.Fatalf("template submit: %v", err)
	}
	if run.TemplateID != "demo" || run.WorkflowID != "" {
		t.Fatalf("unexpected template run: %+v", run)
	}
}
