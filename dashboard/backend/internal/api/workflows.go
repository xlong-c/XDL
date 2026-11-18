package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/queue"
	"github.com/xlong-c/xdl-dashboard/internal/store"
	"github.com/xlong-c/xdl-dashboard/internal/workflow"
)

// handleWorkflowCatalog 原样转发节点目录, Go 不解释组件语义.
func (s *Server) handleWorkflowCatalog(writer http.ResponseWriter) {
	if s.workflows == nil {
		writeError(writer, http.StatusServiceUnavailable, "catalog_unavailable", "workflow node catalog is not loaded")
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusOK)
	_, _ = writer.Write([]byte(`{"catalog":`))
	_, _ = writer.Write(s.workflows.Raw)
	_, _ = writer.Write([]byte("}\n"))
}

func (s *Server) handleListWorkflows(writer http.ResponseWriter, request *http.Request) {
	limit := 100
	if raw := request.URL.Query().Get("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed <= 0 {
			writeError(writer, http.StatusBadRequest, "invalid_limit", "limit must be a positive integer")
			return
		}
		limit = parsed
	}
	items, err := s.database.ListWorkflows(request.Context(), limit)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "list_failed", err.Error())
		return
	}
	response := make([]workflowResponse, 0, len(items))
	for _, item := range items {
		response = append(response, toWorkflowResponse(item, false))
	}
	writeJSON(writer, http.StatusOK, map[string]any{"workflows": response})
}

func (s *Server) handleCreateWorkflow(writer http.ResponseWriter, request *http.Request) {
	raw, err := io.ReadAll(io.LimitReader(request.Body, maxSnapshotBody))
	if err != nil {
		writeError(writer, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	snapshot, issues := workflow.Prepare(raw, s.workflows)
	if workflow.HasErrors(issues) {
		writeIssues(writer, http.StatusBadRequest, "invalid_workflow", "workflow snapshot rejected", issues)
		return
	}
	// id 一律由服务端分配; 客户端提交的 id 不参与语义, 也不影响校验结果.
	snapshot.ID, err = workflow.NewID("wf-")
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "id_failed", err.Error())
		return
	}
	canonical, err := snapshot.CanonicalJSON()
	if err != nil {
		writeError(writer, http.StatusBadRequest, "invalid_workflow", err.Error())
		return
	}
	if semanticIssues := s.semanticIssues(request, canonical); workflow.HasErrors(semanticIssues) {
		status := http.StatusBadRequest
		if hasCompilerUnavailable(semanticIssues) {
			status = http.StatusServiceUnavailable
		}
		writeIssues(writer, status, "invalid_workflow", "workflow semantic validation failed", semanticIssues)
		return
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
	if err := s.database.InsertWorkflow(request.Context(), item); err != nil {
		writeError(writer, http.StatusInternalServerError, "store_failed", err.Error())
		return
	}
	writeJSON(writer, http.StatusCreated, toWorkflowResponse(item, true))
}

func (s *Server) handleValidateWorkflow(writer http.ResponseWriter, request *http.Request) {
	var input validateWorkflowRequest
	if err := decodeJSON(request, &input); err != nil {
		writeError(writer, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if len(input.Snapshot) == 0 {
		writeError(writer, http.StatusBadRequest, "invalid_request", "snapshot is required")
		return
	}
	snapshot, issues := workflow.Prepare(input.Snapshot, s.workflows)
	if !workflow.HasErrors(issues) {
		canonical, err := snapshot.CanonicalJSON()
		if err != nil {
			writeError(writer, http.StatusBadRequest, "invalid_workflow", err.Error())
			return
		}
		issues = append(issues, s.semanticIssues(request, canonical)...)
	}
	response := map[string]any{
		"valid":  !workflow.HasErrors(issues),
		"issues": issues,
	}
	if snapshot.ID != "" {
		response["snapshot_hash"] = workflow.HashBytes(input.Snapshot)
	}
	writeJSON(writer, http.StatusOK, response)
}

func (s *Server) handleWorkflowPath(writer http.ResponseWriter, request *http.Request, path string) {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeError(writer, http.StatusNotFound, "not_found", "workflow id is missing")
		return
	}
	workflowID := parts[0]
	if len(parts) == 1 && request.Method == http.MethodGet {
		item, err := s.database.GetWorkflow(request.Context(), workflowID)
		if err != nil {
			writeError(writer, http.StatusNotFound, "workflow_not_found", err.Error())
			return
		}
		writeJSON(writer, http.StatusOK, toWorkflowResponse(item, true))
		return
	}
	if len(parts) == 2 && parts[1] == "runs" && request.Method == http.MethodPost {
		var input queue.WorkflowSubmitRequest
		if request.Body != nil {
			if err := decodeJSON(request, &input); err != nil && !errors.Is(err, io.EOF) {
				writeError(writer, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
		}
		run, err := s.manager.SubmitWorkflow(request.Context(), workflowID, input)
		if err != nil {
			writeError(writer, http.StatusBadRequest, "submission_rejected", err.Error())
			return
		}
		writeJSON(writer, http.StatusAccepted, toRunResponse(run))
		return
	}
	writeError(writer, http.StatusNotFound, "not_found", "workflow endpoint not found")
}

// semanticIssues 调用 Python compiler 做 XDL 语义预检.
//
// 没有配置 checker 时返回一条阻断问题, 不放行未做语义校验的图.
func (s *Server) semanticIssues(request *http.Request, canonical []byte) []workflow.Issue {
	if s.checker == nil {
		return []workflow.Issue{workflow.ErrorIssue(
			workflow.CodeCompilerUnavailable,
			"workflow 语义预检未配置, 拒绝创建或排队 workflow",
			"",
		)}
	}
	issues, err := s.checker.Validate(request.Context(), canonical)
	if err != nil {
		return []workflow.Issue{workflow.ErrorIssue(
			workflow.CodeCompilerUnavailable,
			"workflow 语义预检失败: "+err.Error(),
			"",
		)}
	}
	return issues
}

func hasCompilerUnavailable(issues []workflow.Issue) bool {
	for _, issue := range issues {
		if issue.Code == workflow.CodeCompilerUnavailable {
			return true
		}
	}
	return false
}

func toWorkflowResponse(item store.Workflow, includeSnapshot bool) workflowResponse {
	response := workflowResponse{
		ID:               item.ID,
		Version:          item.Version,
		Name:             item.Name,
		ExecutionProfile: item.ExecutionProfile,
		SnapshotHash:     item.SnapshotHash,
		CreatedAt:        item.CreatedAt.Format(time.RFC3339Nano),
		UpdatedAt:        item.UpdatedAt.Format(time.RFC3339Nano),
	}
	if includeSnapshot && item.SnapshotJSON != "" {
		response.Snapshot = json.RawMessage(item.SnapshotJSON)
	}
	return response
}

func writeIssues(writer http.ResponseWriter, status int, code, message string, issues []workflow.Issue) {
	writeJSON(writer, status, map[string]any{
		"error":  map[string]string{"code": code, "message": message},
		"issues": issues,
	})
}
