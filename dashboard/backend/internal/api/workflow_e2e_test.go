package api

import (
	"encoding/json"
	"net/http"
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

func TestWorkflowTrainingE2E(t *testing.T) {
	python := os.Getenv("XDL_DASHBOARD_E2E_PYTHON")
	if python == "" {
		t.Skip("set XDL_DASHBOARD_E2E_PYTHON to an interpreter with XDL and PyTorch")
	}
	python, err := workflow.ResolvePythonExecutable(python)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("OMP_NUM_THREADS", "1")
	t.Setenv("MKL_NUM_THREADS", "1")
	root, err := filepath.Abs("../../../..")
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Default()
	cfg.ProjectRoot = root
	cfg.DataRoot = t.TempDir()
	cfg.OutputRoot = t.TempDir()
	cfg.PythonExecutable = python
	cfg.MetricsInterval = time.Second
	catalog, err := templates.Load(filepath.Join(root, "dashboard/templates/catalog.json"), root, cfg.DataRoot)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := workflow.LoadCatalog(filepath.Join(root, "xdl/workflow/node_catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	database, err := store.Open(filepath.Join(t.TempDir(), "dashboard.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	hub := events.NewHub(64)
	jobRunner := runner.New(cfg, database, hub)
	manager := queue.New(cfg, database, catalog, nodes, jobRunner, hub, nil)
	checker := workflow.NewCompilerChecker(workflow.CompilerConfig{
		PythonExecutable: python, ProjectRoot: root, DataRoot: cfg.DataRoot,
	})
	server := NewServer(cfg, database, manager, catalog, nodes, checker, hub)

	for _, tc := range []struct {
		profile, fixture string
		steps            int
	}{
		{workflow.ProfileSchemaV1, "synthetic-cpu-demo.json", 8},
		{workflow.ProfileDagRuntime, "dag-runtime-demo.json", 6},
		{workflow.ProfileDagRuntime, "step-graph-cpu-demo.json", 6},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			snapshot, err := workflow.DecodeSnapshot(fixtureSnapshot(t, tc.fixture))
			if err != nil {
				t.Fatal(err)
			}
			for _, node := range snapshot.Nodes {
				switch node.ID {
				case "trainer":
					node.Params["batch_size"] = 8
				case "ds-b":
					node.Params["num_samples"] = 16
				}
			}
			raw, err := json.Marshal(snapshot)
			if err != nil {
				t.Fatal(err)
			}
			response := do(t, server, http.MethodPost, "/api/workflows", raw)
			if response.Code != http.StatusCreated {
				t.Fatalf("create workflow: %d %s", response.Code, response.Body.String())
			}
			saved := decodeBody(t, response)
			if saved["execution_profile"] != tc.profile {
				t.Fatalf("unexpected saved profile: %v", saved["execution_profile"])
			}
			if snapshot.StepGraph != nil {
				response = do(t, server, http.MethodGet, "/api/workflows/"+saved["id"].(string), nil)
				var restored struct {
					Snapshot json.RawMessage `json:"snapshot"`
				}
				if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &restored) != nil {
					t.Fatalf("reload workflow: %d %s", response.Code, response.Body.String())
				}
				decoded, err := workflow.DecodeSnapshot(restored.Snapshot)
				if err != nil || decoded.StepGraph == nil || len(decoded.StepGraph.Nodes) != len(snapshot.StepGraph.Nodes) || len(decoded.StepGraph.Edges) != len(snapshot.StepGraph.Edges) {
					t.Fatalf("step graph lost on reload: %v", err)
				}
				if digest := workflow.HashBytes(restored.Snapshot); digest != saved["snapshot_hash"] {
					t.Fatalf("restored snapshot hash: %s, want %v", digest, saved["snapshot_hash"])
				}
			}
			response = do(t, server, http.MethodPost, "/api/workflows/"+saved["id"].(string)+"/runs", []byte(`{}`))
			if response.Code != http.StatusAccepted {
				t.Fatalf("queue workflow: %d %s", response.Code, response.Body.String())
			}
			runPath := "/api/runs/" + decodeBody(t, response)["id"].(string)
			var history []eventResponse
			deadline := time.Now().Add(60 * time.Second)
			for time.Now().Before(deadline) {
				response = do(t, server, http.MethodGet, runPath+"/events", nil)
				var payload struct{ Events []eventResponse }
				if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &payload) != nil {
					t.Fatalf("read run events: %d %s", response.Code, response.Body.String())
				}
				history = payload.Events
				for _, event := range history {
					// The runner emits a final status after reaping the process.
					if status := event.Payload["status"]; event.Type == "worker_exit" && status != nil {
						if status != store.StatusSucceeded {
							t.Fatalf("training failed: %+v", history)
						}
						response = do(t, server, http.MethodGet, runPath, nil)
						run := decodeBody(t, response)
						if run["status"] != store.StatusSucceeded || run["last_step"] != float64(tc.steps) ||
							run["compiler_version"] == nil || run["workflow_id"] != saved["id"] ||
							run["workflow_snapshot_hash"] != saved["snapshot_hash"] {
							t.Fatalf("unexpected run metadata (want %d steps): %+v", tc.steps, run)
						}
						return
					}
				}
				time.Sleep(100 * time.Millisecond)
			}
			t.Fatalf("timed out waiting for training: %+v", history)
		})
	}
}
