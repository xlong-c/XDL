package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/queue"
	"github.com/xlong-c/xdl-dashboard/internal/store"
	"github.com/xlong-c/xdl-dashboard/internal/system"
	"github.com/xlong-c/xdl-dashboard/internal/templates"
	"github.com/xlong-c/xdl-dashboard/internal/workflow"
)

type Server struct {
	cfg       config.Config
	database  *store.Store
	manager   *queue.Manager
	catalog   *templates.Catalog
	workflows *workflow.Catalog
	checker   workflow.SemanticChecker
	hub       *events.Hub
	started   time.Time
	frontend  http.Handler
	design    http.Handler
	ready     atomic.Bool
}

// maxSnapshotBody 限制 workflow 请求体大小, 与目录 limits 无关, 用于兜住畸形请求.
const maxSnapshotBody = 1 << 20

type runResponse struct {
	ID              string         `json:"id"`
	ParentRunID     string         `json:"parent_run_id,omitempty"`
	TemplateID      string         `json:"template_id"`
	Status          string         `json:"status"`
	SubmittedAt     string         `json:"submitted_at"`
	UpdatedAt       string         `json:"updated_at"`
	StartedAt       string         `json:"started_at,omitempty"`
	FinishedAt      string         `json:"finished_at,omitempty"`
	Devices         []int          `json:"devices"`
	Launcher        string         `json:"launcher"`
	ProcessID       int            `json:"process_id,omitempty"`
	LastStep        int            `json:"last_step"`
	LastEpoch       int            `json:"last_epoch"`
	CheckpointPath  string         `json:"checkpoint_path,omitempty"`
	OutputDir       string         `json:"output_dir"`
	ErrorMessage    string         `json:"error_message,omitempty"`
	Config          map[string]any `json:"config,omitempty"`
	WorkflowID      string         `json:"workflow_id,omitempty"`
	WorkflowHash    string         `json:"workflow_snapshot_hash,omitempty"`
	CompilerVersion string         `json:"compiler_version,omitempty"`
}

type workflowResponse struct {
	ID               string          `json:"id"`
	Version          int             `json:"version"`
	Name             string          `json:"name"`
	ExecutionProfile string          `json:"execution_profile"`
	SnapshotHash     string          `json:"snapshot_hash"`
	CreatedAt        string          `json:"created_at"`
	UpdatedAt        string          `json:"updated_at"`
	Snapshot         json.RawMessage `json:"snapshot,omitempty"`
}

type validateWorkflowRequest struct {
	Snapshot json.RawMessage `json:"snapshot"`
}

type eventResponse struct {
	ID        int64          `json:"id"`
	RunID     string         `json:"run_id"`
	Type      string         `json:"type"`
	CreatedAt string         `json:"created_at"`
	Payload   map[string]any `json:"payload"`
	Reliable  bool           `json:"reliable"`
}

func NewServer(cfg config.Config, database *store.Store, manager *queue.Manager, catalog *templates.Catalog, workflows *workflow.Catalog, checker workflow.SemanticChecker, hub *events.Hub) *Server {
	server := &Server{
		cfg: cfg, database: database, manager: manager, catalog: catalog,
		workflows: workflows, checker: checker, hub: hub, started: time.Now().UTC(),
	}
	server.frontend = http.FileServer(http.Dir(cfg.FrontendDir))
	designRoot := filepath.Join(cfg.ProjectRoot, "dashboard", "design")
	server.design = http.FileServer(http.Dir(designRoot))
	return server
}

func (s *Server) SetReady(value bool) {
	s.ready.Store(value)
}

func (s *Server) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	writer.Header().Set("X-Content-Type-Options", "nosniff")
	writer.Header().Set("X-Frame-Options", "SAMEORIGIN")
	writer.Header().Set("Referrer-Policy", "no-referrer")
	writer.Header().Set("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; base-uri 'self'; frame-ancestors 'self'")
	if strings.HasPrefix(request.URL.Path, "/api/") {
		s.handleAPI(writer, request)
		return
	}
	if strings.HasPrefix(request.URL.Path, "/design") {
		s.handleDesign(writer, request)
		return
	}
	s.handleFrontend(writer, request)
}

func (s *Server) handleAPI(writer http.ResponseWriter, request *http.Request) {
	path := strings.TrimPrefix(request.URL.Path, "/api/")
	switch {
	case path == "health" && request.Method == http.MethodGet:
		s.handleHealth(writer)
	case path == "templates" && request.Method == http.MethodGet:
		s.handleTemplates(writer)
	case path == "runs" && request.Method == http.MethodGet:
		s.handleListRuns(writer, request)
	case path == "runs" && request.Method == http.MethodPost:
		s.handleSubmitRun(writer, request)
	case path == "system" && request.Method == http.MethodGet:
		s.handleLatestSystem(writer, request)
	case path == "system/stream" && request.Method == http.MethodGet:
		s.handleSSE(writer, request, "")
	case path == "workflows" && request.Method == http.MethodGet:
		s.handleListWorkflows(writer, request)
	case path == "workflows" && request.Method == http.MethodPost:
		s.handleCreateWorkflow(writer, request)
	case path == "workflows/catalog" && request.Method == http.MethodGet:
		s.handleWorkflowCatalog(writer)
	case path == "workflows/validate" && request.Method == http.MethodPost:
		s.handleValidateWorkflow(writer, request)
	case strings.HasPrefix(path, "workflows/"):
		s.handleWorkflowPath(writer, request, strings.TrimPrefix(path, "workflows/"))
	case strings.HasPrefix(path, "runs/"):
		s.handleRunPath(writer, request, strings.TrimPrefix(path, "runs/"))
	default:
		writeError(writer, http.StatusNotFound, "not_found", "API endpoint not found")
	}
}

func (s *Server) handleHealth(writer http.ResponseWriter) {
	writeJSON(writer, http.StatusOK, map[string]any{
		"ready":       s.ready.Load(),
		"uptime_sec":  time.Since(s.started).Seconds(),
		"listen_addr": s.cfg.ListenAddr,
	})
}

func (s *Server) handleTemplates(writer http.ResponseWriter) {
	writeJSON(writer, http.StatusOK, map[string]any{"templates": s.catalog.List()})
}

func (s *Server) handleListRuns(writer http.ResponseWriter, request *http.Request) {
	limit := 100
	if raw := request.URL.Query().Get("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			writeError(writer, http.StatusBadRequest, "invalid_limit", "limit must be an integer")
			return
		}
		limit = parsed
	}
	runs, err := s.manager.List(request.Context(), limit)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "list_failed", err.Error())
		return
	}
	response := make([]runResponse, 0, len(runs))
	for _, run := range runs {
		response = append(response, toRunResponse(run))
	}
	writeJSON(writer, http.StatusOK, map[string]any{"runs": response})
}

func (s *Server) handleSubmitRun(writer http.ResponseWriter, request *http.Request) {
	var input queue.SubmitRequest
	if err := decodeJSON(request, &input); err != nil {
		writeError(writer, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	run, err := s.manager.Submit(request.Context(), input)
	if err != nil {
		writeError(writer, http.StatusBadRequest, "submission_rejected", err.Error())
		return
	}
	writeJSON(writer, http.StatusAccepted, toRunResponse(run))
}

func (s *Server) handleRunPath(writer http.ResponseWriter, request *http.Request, path string) {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeError(writer, http.StatusNotFound, "not_found", "run id is missing")
		return
	}
	runID := parts[0]
	if len(parts) == 1 && request.Method == http.MethodGet {
		run, err := s.manager.Get(request.Context(), runID)
		if err != nil {
			writeError(writer, http.StatusNotFound, "run_not_found", err.Error())
			return
		}
		writeJSON(writer, http.StatusOK, toRunResponse(run))
		return
	}
	if len(parts) == 2 && parts[1] == "stream" && request.Method == http.MethodGet {
		s.handleSSE(writer, request, runID)
		return
	}
	if len(parts) == 2 && parts[1] == "logs" && request.Method == http.MethodGet {
		s.handleRunEvents(writer, request, runID, "log")
		return
	}
	if len(parts) == 2 && parts[1] == "metrics" && request.Method == http.MethodGet {
		s.handleRunMetrics(writer, request, runID)
		return
	}
	if len(parts) == 2 && parts[1] == "events" && request.Method == http.MethodGet {
		s.handleRunEvents(writer, request, runID, "")
		return
	}
	if len(parts) == 2 && parts[1] == "artifacts" && request.Method == http.MethodGet {
		s.handleArtifacts(writer, request, runID)
		return
	}
	if len(parts) >= 3 && parts[1] == "artifact" && request.Method == http.MethodGet {
		s.handleArtifactFile(writer, request, runID, strings.Join(parts[2:], "/"))
		return
	}
	if len(parts) == 3 && parts[1] == "actions" && request.Method == http.MethodPost {
		s.handleAction(writer, request, runID, parts[2])
		return
	}
	writeError(writer, http.StatusNotFound, "not_found", "run endpoint not found")
}

func (s *Server) handleAction(writer http.ResponseWriter, request *http.Request, runID, action string) {
	if action == "resume-from-checkpoint" {
		var input queue.SubmitRequest
		if request.Body != nil {
			if err := decodeJSON(request, &input); err != nil && !errors.Is(err, io.EOF) {
				writeError(writer, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
		}
		input.ParentRunID = runID
		parent, err := s.manager.Get(request.Context(), runID)
		if err != nil {
			writeError(writer, http.StatusNotFound, "run_not_found", err.Error())
			return
		}
		if parent.WorkflowID != "" {
			resumed, err := s.manager.SubmitWorkflow(
				request.Context(),
				parent.WorkflowID,
				queue.WorkflowSubmitRequest{
					Name:             input.Name,
					Launcher:         input.Launcher,
					Devices:          input.Devices,
					ParentRunID:      runID,
					ResumeCheckpoint: input.ResumeCheckpoint,
				},
			)
			if err != nil {
				writeError(writer, http.StatusBadRequest, "resume_rejected", err.Error())
				return
			}
			writeJSON(writer, http.StatusAccepted, toRunResponse(resumed))
			return
		}
		input.TemplateID = parent.TemplateID
		if input.Parameters == nil {
			if submitted, ok := parent.ConfigJSON["submitted_parameters"].(map[string]any); ok {
				input.Parameters = submitted
			}
		}
		run, err := s.manager.Submit(request.Context(), input)
		if err != nil {
			writeError(writer, http.StatusBadRequest, "resume_rejected", err.Error())
			return
		}
		writeJSON(writer, http.StatusAccepted, toRunResponse(run))
		return
	}
	if err := s.manager.Control(request.Context(), runID, action); err != nil {
		writeError(writer, http.StatusConflict, "action_rejected", err.Error())
		return
	}
	writeJSON(writer, http.StatusAccepted, map[string]any{"run_id": runID, "action": action})
}

func (s *Server) handleRunEvents(writer http.ResponseWriter, request *http.Request, runID, eventType string) {
	afterID := int64(0)
	if raw := request.Header.Get("Last-Event-ID"); raw != "" {
		afterID, _ = strconv.ParseInt(raw, 10, 64)
	}
	events, err := s.database.ListEvents(request.Context(), runID, afterID, 1000)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "events_failed", err.Error())
		return
	}
	if eventType != "" {
		filtered := make([]store.Event, 0, len(events))
		for _, event := range events {
			if event.Type == eventType {
				filtered = append(filtered, event)
			}
		}
		events = filtered
	}
	response := make([]eventResponse, 0, len(events))
	for _, event := range events {
		response = append(response, toEventResponse(event))
	}
	writeJSON(writer, http.StatusOK, map[string]any{"events": response})
}

func (s *Server) handleRunMetrics(writer http.ResponseWriter, request *http.Request, runID string) {
	limit := 10000
	if raw := request.URL.Query().Get("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed <= 0 {
			writeError(writer, http.StatusBadRequest, "invalid_limit", "limit must be a positive integer")
			return
		}
		limit = parsed
	}
	metrics, err := s.database.ListMetrics(request.Context(), runID, limit)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "metrics_failed", err.Error())
		return
	}
	response := make([]map[string]any, 0, len(metrics))
	for _, sample := range metrics {
		response = append(response, map[string]any{
			"id":          sample.ID,
			"created_at":  sample.CreatedAt.Format(time.RFC3339Nano),
			"global_step": sample.GlobalStep,
			"epoch":       sample.Epoch,
			"metrics":     sample.Metrics,
		})
	}
	writeJSON(writer, http.StatusOK, map[string]any{"metrics": response})
}

func (s *Server) handleSSE(writer http.ResponseWriter, request *http.Request, runID string) {
	flusher, ok := writer.(http.Flusher)
	if !ok {
		writeError(writer, http.StatusInternalServerError, "sse_unavailable", "streaming is unavailable")
		return
	}
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("Connection", "keep-alive")
	writer.WriteHeader(http.StatusOK)
	flusher.Flush()
	afterID := int64(0)
	if raw := request.Header.Get("Last-Event-ID"); raw != "" {
		afterID, _ = strconv.ParseInt(raw, 10, 64)
	}
	subscription := s.hub.Subscribe(afterID, s.cfg.EventBuffer)
	defer subscription.Close()
	history, err := s.database.ListEvents(request.Context(), runID, afterID, 1000)
	if runID == "" {
		history, err = s.database.ListSystemEvents(request.Context(), afterID, 1000)
	}
	if err != nil {
		return
	}
	for _, event := range history {
		if err := writeSSE(writer, flusher, event); err != nil {
			return
		}
		if event.ID > afterID {
			afterID = event.ID
		}
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case event, ok := <-subscription.C:
			if !ok {
				return
			}
			if event.ID <= afterID {
				continue
			}
			if runID != "" && event.RunID != runID {
				continue
			}
			if runID == "" && event.RunID != "" {
				continue
			}
			if err := writeSSE(writer, flusher, store.Event{
				ID: event.ID, RunID: event.RunID, Type: event.Type,
				CreatedAt: event.CreatedAt, Payload: event.Payload, Reliable: event.Reliable,
			}); err != nil {
				return
			}
			afterID = event.ID
		}
	}
}

func writeSSE(writer http.ResponseWriter, flusher http.Flusher, event store.Event) error {
	data, err := json.Marshal(eventResponse{
		ID: event.ID, RunID: event.RunID, Type: event.Type,
		CreatedAt: event.CreatedAt.Format(time.RFC3339Nano),
		Payload:   event.Payload, Reliable: event.Reliable,
	})
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(writer, "id: %d\nevent: %s\ndata: %s\n\n", event.ID, event.Type, data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func (s *Server) handleDesign(writer http.ResponseWriter, request *http.Request) {
	root := filepath.Join(s.cfg.ProjectRoot, "dashboard", "design")
	relative := strings.TrimPrefix(request.URL.Path, "/design")
	asset := "index.html"
	switch relative {
	case "", "/", "/index.html":
	case "/design.js":
		asset = "design.js"
	case "/workflow.js":
		asset = "workflow.js"
	case "/workflow.css":
		asset = "workflow.css"
	case "/concepts.html":
		asset = "concepts.html"
	case "/concepts.js":
		asset = "concepts.js"
	case "/concepts.css":
		asset = "concepts.css"
	default:
		writeError(writer, http.StatusNotFound, "design_missing", "design preview asset not found")
		return
	}
	indexPath := filepath.Join(root, asset)
	if _, err := os.Stat(indexPath); err != nil {
		writeError(writer, http.StatusNotFound, "design_missing", "design preview index.html not found")
		return
	}
	file, err := os.Open(indexPath)
	if err != nil {
		writeError(writer, http.StatusNotFound, "design_missing", "design preview index.html not found")
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		writeError(writer, http.StatusNotFound, "design_missing", "design preview index.html not found")
		return
	}
	http.ServeContent(writer, request, asset, info.ModTime(), file)
}

func (s *Server) handleFrontend(writer http.ResponseWriter, request *http.Request) {
	cleanPath := filepath.Clean("/" + request.URL.Path)
	if strings.Contains(cleanPath, "/../") || cleanPath == "/.." {
		writeError(writer, http.StatusBadRequest, "invalid_path", "invalid frontend path")
		return
	}
	if _, err := os.Stat(filepath.Join(s.cfg.FrontendDir, "index.html")); err != nil {
		writeError(writer, http.StatusNotFound, "frontend_missing", "frontend/dist/index.html is not built")
		return
	}
	if request.URL.Path == "/" {
		s.frontend.ServeHTTP(writer, request)
		return
	}
	candidate := filepath.Join(s.cfg.FrontendDir, filepath.Clean(request.URL.Path))
	if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
		s.frontend.ServeHTTP(writer, request)
		return
	}
	request.URL.Path = "/"
	s.frontend.ServeHTTP(writer, request)
}

func (s *Server) StartSystemStream(ctx context.Context, collector *system.Collector, processIDs func() []int) {
	go func() {
		for sample := range collector.Start(ctx, processIDs) {
			id, err := s.database.RecordEvent(ctx, store.Event{RunID: "", Type: "system", CreatedAt: sample.CreatedAt, Payload: sample.Payload, Reliable: false})
			if err != nil {
				continue
			}
			s.hub.Publish(events.Event{ID: id, RunID: "", Type: "system", CreatedAt: sample.CreatedAt, Payload: sample.Payload, Reliable: false})
			_ = s.database.RecordSystemSample(ctx, sample.CreatedAt, sample.Payload)
		}
	}()
}

func toRunResponse(run store.Run) runResponse {
	// nil 切片会被编码成 null, 前端按数组处理会崩; 这里统一成空数组.
	devices := run.Devices
	if devices == nil {
		devices = []int{}
	}
	return runResponse{
		ID: run.ID, ParentRunID: run.ParentRunID, TemplateID: run.TemplateID, Status: run.Status,
		SubmittedAt: run.SubmittedAt.Format(time.RFC3339Nano), UpdatedAt: run.UpdatedAt.Format(time.RFC3339Nano),
		StartedAt: formatTime(run.StartedAt), FinishedAt: formatTime(run.FinishedAt), Devices: devices,
		Launcher: run.Launcher, ProcessID: run.ProcessID, LastStep: run.LastStep, LastEpoch: run.LastEpoch,
		CheckpointPath: run.CheckpointPath, OutputDir: run.OutputDir, ErrorMessage: run.ErrorMessage,
		Config: run.ConfigJSON, WorkflowID: run.WorkflowID,
		WorkflowHash: run.WorkflowSnapshotHash, CompilerVersion: run.CompilerVersion,
	}
}

func toEventResponse(event store.Event) eventResponse {
	return eventResponse{ID: event.ID, RunID: event.RunID, Type: event.Type, CreatedAt: event.CreatedAt.Format(time.RFC3339Nano), Payload: event.Payload, Reliable: event.Reliable}
}

func formatTime(value *time.Time) string {
	if value == nil {
		return ""
	}
	return value.Format(time.RFC3339Nano)
}

func decodeJSON(request *http.Request, target any) error {
	decoder := json.NewDecoder(io.LimitReader(request.Body, 2<<20))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func writeError(writer http.ResponseWriter, status int, code, message string) {
	writeJSON(writer, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}
