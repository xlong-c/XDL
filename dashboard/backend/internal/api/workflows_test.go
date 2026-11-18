package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/queue"
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

type fakeChecker struct {
	issues []workflow.Issue
	err    error
	calls  int
}

func (f *fakeChecker) Validate(_ context.Context, _ []byte) ([]workflow.Issue, error) {
	f.calls++
	return f.issues, f.err
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newTestServer(t *testing.T, checker workflow.SemanticChecker) (*Server, *store.Store, config.Config) {
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
	manager := queue.New(cfg, database, catalog, workflowCatalog, jobRunner, hub, nil)
	return NewServer(cfg, database, manager, catalog, workflowCatalog, checker, hub), database, cfg
}

func fixtureSnapshot(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "templates", "workflows", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return raw
}

func do(t *testing.T, server *Server, method, path string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	var reader *bytes.Reader
	if body == nil {
		reader = bytes.NewReader(nil)
	} else {
		reader = bytes.NewReader(body)
	}
	request := httptest.NewRequest(method, path, reader)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	return response
}

func decodeBody(t *testing.T, response *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response %q: %v", response.Body.String(), err)
	}
	return payload
}

func TestWorkflowCatalogEndpointForwardsNodeCatalog(t *testing.T) {
	server, _, _ := newTestServer(t, &fakeChecker{})
	response := do(t, server, http.MethodGet, "/api/workflows/catalog", nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	catalog, ok := payload["catalog"].(map[string]any)
	if !ok {
		t.Fatalf("catalog missing: %v", payload)
	}
	if catalog["version"] != float64(1) {
		t.Fatalf("catalog version = %v", catalog["version"])
	}
}

func TestCreateWorkflowAssignsIDAndHash(t *testing.T) {
	checker := &fakeChecker{}
	server, _, _ := newTestServer(t, checker)
	response := do(t, server, http.MethodPost, "/api/workflows", fixtureSnapshot(t, "synthetic-cpu-demo.json"))
	if response.Code != http.StatusCreated {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	id, _ := payload["id"].(string)
	if len(id) == 0 {
		t.Fatalf("workflow id missing: %v", payload)
	}
	if payload["execution_profile"] != "schema-v1" {
		t.Fatalf("profile = %v", payload["execution_profile"])
	}
	hash, _ := payload["snapshot_hash"].(string)
	if len(hash) != 64 {
		t.Fatalf("snapshot hash = %q", hash)
	}
	if checker.calls != 1 {
		t.Fatalf("semantic checker calls = %d", checker.calls)
	}
	snapshot, ok := payload["snapshot"].(map[string]any)
	if !ok {
		t.Fatalf("snapshot missing: %v", payload)
	}
	if snapshot["id"] != id {
		t.Fatalf("snapshot id = %v, want %v", snapshot["id"], id)
	}
}

func TestCreateWorkflowRejectsStructuralErrors(t *testing.T) {
	server, _, _ := newTestServer(t, &fakeChecker{})
	response := do(t, server, http.MethodPost, "/api/workflows", []byte(`{"version": 2}`))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	issues, ok := payload["issues"].([]any)
	if !ok || len(issues) == 0 {
		t.Fatalf("issues missing: %v", payload)
	}
}

func TestCreateWorkflowRejectsSemanticErrors(t *testing.T) {
	checker := &fakeChecker{issues: []workflow.Issue{
		{Code: "type_mismatch", Level: "error", Message: "bad edge"},
	}}
	server, _, _ := newTestServer(t, checker)
	response := do(t, server, http.MethodPost, "/api/workflows", fixtureSnapshot(t, "synthetic-cpu-demo.json"))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	if payload["error"].(map[string]any)["code"] != "invalid_workflow" {
		t.Fatalf("unexpected error payload: %v", payload)
	}
}

func TestCreateWorkflowRequiresSemanticChecker(t *testing.T) {
	server, _, _ := newTestServer(t, nil)
	response := do(t, server, http.MethodPost, "/api/workflows", fixtureSnapshot(t, "synthetic-cpu-demo.json"))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	issues := payload["issues"].([]any)
	if issues[0].(map[string]any)["code"] != "compiler_unavailable" {
		t.Fatalf("unexpected issues: %v", issues)
	}
}

func TestValidateWorkflowEndpoint(t *testing.T) {
	server, _, _ := newTestServer(t, &fakeChecker{})
	body, err := json.Marshal(map[string]any{
		"snapshot": json.RawMessage(fixtureSnapshot(t, "cifar10-demo.json")),
	})
	if err != nil {
		t.Fatal(err)
	}
	response := do(t, server, http.MethodPost, "/api/workflows/validate", body)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
	payload := decodeBody(t, response)
	if payload["valid"] != true {
		t.Fatalf("valid = %v; issues = %v", payload["valid"], payload["issues"])
	}
}

func TestGetAndListWorkflows(t *testing.T) {
	server, _, _ := newTestServer(t, &fakeChecker{})
	created := do(t, server, http.MethodPost, "/api/workflows", fixtureSnapshot(t, "synthetic-cpu-demo.json"))
	if created.Code != http.StatusCreated {
		t.Fatalf("create status = %d", created.Code)
	}
	id := decodeBody(t, created)["id"].(string)

	single := do(t, server, http.MethodGet, "/api/workflows/"+id, nil)
	if single.Code != http.StatusOK {
		t.Fatalf("get status = %d; body = %s", single.Code, single.Body.String())
	}
	if _, ok := decodeBody(t, single)["snapshot"]; !ok {
		t.Fatal("single workflow must include snapshot")
	}

	list := do(t, server, http.MethodGet, "/api/workflows?limit=10", nil)
	if list.Code != http.StatusOK {
		t.Fatalf("list status = %d", list.Code)
	}
	items := decodeBody(t, list)["workflows"].([]any)
	if len(items) != 1 {
		t.Fatalf("workflows = %v", items)
	}
	if _, ok := items[0].(map[string]any)["snapshot"]; ok {
		t.Fatal("workflow list must not include snapshots")
	}

	missing := do(t, server, http.MethodGet, "/api/workflows/wf-missing", nil)
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing status = %d", missing.Code)
	}
}

func TestCreateRunFromWorkflow(t *testing.T) {
	server, database, _ := newTestServer(t, &fakeChecker{})
	created := do(t, server, http.MethodPost, "/api/workflows", fixtureSnapshot(t, "synthetic-cpu-demo.json"))
	if created.Code != http.StatusCreated {
		t.Fatalf("create status = %d; body = %s", created.Code, created.Body.String())
	}
	id := decodeBody(t, created)["id"].(string)

	runResponseRecorder := do(t, server, http.MethodPost, "/api/workflows/"+id+"/runs", []byte(`{"name":"wf run"}`))
	if runResponseRecorder.Code != http.StatusAccepted {
		t.Fatalf("run status = %d; body = %s", runResponseRecorder.Code, runResponseRecorder.Body.String())
	}
	payload := decodeBody(t, runResponseRecorder)
	if payload["workflow_id"] != id {
		t.Fatalf("run workflow id = %v", payload["workflow_id"])
	}
	if payload["workflow_snapshot_hash"] == "" {
		t.Fatalf("run snapshot hash missing: %v", payload)
	}
	stored, err := database.GetRun(context.Background(), payload["id"].(string))
	if err != nil {
		t.Fatalf("load run: %v", err)
	}
	if stored.WorkflowID != id {
		t.Fatalf("stored workflow id = %q", stored.WorkflowID)
	}
}

func TestCreateRunFromMissingWorkflow(t *testing.T) {
	server, _, _ := newTestServer(t, &fakeChecker{})
	response := do(t, server, http.MethodPost, "/api/workflows/wf-missing/runs", []byte(`{}`))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d; body = %s", response.Code, response.Body.String())
	}
}
