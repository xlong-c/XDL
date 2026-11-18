package api

import (
	"database/sql"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type artifactResponse struct {
	Kind      string         `json:"kind"`
	Path      string         `json:"path"`
	CreatedAt string         `json:"created_at"`
	SizeBytes int64          `json:"size_bytes,omitempty"`
	URL       string         `json:"url,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

func (s *Server) handleLatestSystem(writer http.ResponseWriter, request *http.Request) {
	createdAt, payload, err := s.database.LatestSystemSample(request.Context())
	if err == sql.ErrNoRows {
		writeJSON(writer, http.StatusOK, map[string]any{"sample": map[string]any{}})
		return
	}
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "system_sample_failed", err.Error())
		return
	}
	writeJSON(writer, http.StatusOK, map[string]any{"created_at": createdAt.Format(time.RFC3339Nano), "sample": payload})
}

func (s *Server) handleArtifacts(writer http.ResponseWriter, request *http.Request, runID string) {
	run, err := s.manager.Get(request.Context(), runID)
	if err != nil {
		writeError(writer, http.StatusNotFound, "run_not_found", err.Error())
		return
	}
	if err := indexRunArtifacts(request.Context(), s.database, run); err != nil {
		writeError(writer, http.StatusInternalServerError, "artifact_index_failed", err.Error())
		return
	}
	artifacts, err := s.database.ListArtifacts(request.Context(), runID)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "artifacts_failed", err.Error())
		return
	}
	result := make([]artifactResponse, 0, len(artifacts))
	for _, artifact := range artifacts {
		sizeBytes := int64(0)
		if value, ok := artifact.Metadata["size_bytes"].(float64); ok {
			sizeBytes = int64(value)
		}
		item := artifactResponse{
			Kind: artifact.Kind, Path: artifact.RelativePath,
			CreatedAt: artifact.CreatedAt.Format(time.RFC3339Nano),
			SizeBytes: sizeBytes, Metadata: artifact.Metadata,
		}
		if artifact.Kind == "preview" || artifact.Kind == "image" {
			item.URL = artifactURL(runID, artifact.RelativePath)
		}
		result = append(result, item)
	}
	fresh, _ := s.manager.Get(request.Context(), runID)
	writeJSON(writer, http.StatusOK, map[string]any{"artifacts": result, "checkpoint_path": fresh.CheckpointPath})
}

func (s *Server) handleArtifactFile(writer http.ResponseWriter, request *http.Request, runID, relative string) {
	run, err := s.manager.Get(request.Context(), runID)
	if err != nil {
		writeError(writer, http.StatusNotFound, "run_not_found", err.Error())
		return
	}
	clean := filepath.Clean(filepath.FromSlash(relative))
	if filepath.IsAbs(clean) || clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		writeError(writer, http.StatusBadRequest, "invalid_artifact_path", "artifact path must stay inside the run output")
		return
	}
	if extension := strings.ToLower(filepath.Ext(clean)); extension != ".png" && extension != ".jpg" && extension != ".jpeg" && extension != ".webp" {
		writeError(writer, http.StatusForbidden, "artifact_type_denied", "only image artifacts can be previewed")
		return
	}
	root, err := filepath.EvalSymlinks(run.OutputDir)
	if err != nil {
		writeError(writer, http.StatusNotFound, "artifact_not_found", "run output is unavailable")
		return
	}
	candidate, err := filepath.EvalSymlinks(filepath.Join(root, clean))
	if err != nil {
		writeError(writer, http.StatusNotFound, "artifact_not_found", "artifact was not found")
		return
	}
	rel, err := filepath.Rel(root, candidate)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		writeError(writer, http.StatusForbidden, "artifact_path_denied", "artifact path escapes the run output")
		return
	}
	file, err := os.Open(candidate)
	if err != nil {
		writeError(writer, http.StatusNotFound, "artifact_not_found", "artifact was not found")
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		writeError(writer, http.StatusNotFound, "artifact_not_found", "artifact is not a regular file")
		return
	}
	contentType := mime.TypeByExtension(filepath.Ext(candidate))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	writer.Header().Set("Content-Type", contentType)
	writer.Header().Set("Content-Disposition", "inline")
	http.ServeContent(writer, request, filepath.Base(candidate), info.ModTime(), file)
}

func artifactURL(runID, relative string) string {
	parts := strings.Split(filepath.ToSlash(relative), "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return fmt.Sprintf("/api/runs/%s/artifact/%s", url.PathEscape(runID), strings.Join(parts, "/"))
}
