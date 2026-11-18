package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/xlong-c/xdl-dashboard/internal/config"
)

func TestDesignPreviewAssets(t *testing.T) {
	root := t.TempDir()
	design := filepath.Join(root, "dashboard", "design")
	if err := os.MkdirAll(design, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"index.html":    "<!doctype html><title>XDL design</title>",
		"design.js":     "'use strict'; console.log('design preview');",
		"workflow.js":   "'use strict'; console.log('workflow preview');",
		"workflow.css":  ".workflow-preview { color: green; }",
		"concepts.html": "<!doctype html><title>XDL concepts</title>",
		"concepts.js":   "'use strict'; console.log('concepts preview');",
		"concepts.css":  ".concepts-preview { color: blue; }",
	} {
		if err := os.WriteFile(filepath.Join(design, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	server := &Server{cfg: config.Config{ProjectRoot: root}}
	for _, test := range []struct {
		path        string
		status      int
		contentType string
	}{
		{"/design", http.StatusOK, "text/html"},
		{"/design/", http.StatusOK, "text/html"},
		{"/design/index.html", http.StatusOK, "text/html"},
		{"/design/design.js", http.StatusOK, "javascript"},
		{"/design/workflow.js", http.StatusOK, "javascript"},
		{"/design/workflow.css", http.StatusOK, "text/css"},
		{"/design/concepts.html", http.StatusOK, "text/html"},
		{"/design/concepts.js", http.StatusOK, "javascript"},
		{"/design/concepts.css", http.StatusOK, "text/css"},
		{"/design/missing.js", http.StatusNotFound, "application/json"},
		{"/design/../config.example.json", http.StatusNotFound, "application/json"},
		{"/design/%2e%2e/config.example.json", http.StatusNotFound, "application/json"},
		{"/design/design.js/extra", http.StatusNotFound, "application/json"},
	} {
		t.Run(test.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			server.ServeHTTP(response, httptest.NewRequest(http.MethodGet, test.path, nil))
			if response.Code != test.status {
				t.Fatalf("status = %d; want %d", response.Code, test.status)
			}
			if got := response.Header().Get("Content-Type"); !strings.Contains(got, test.contentType) {
				t.Errorf("Content-Type = %q; want %q", got, test.contentType)
			}
			if location := response.Header().Get("Location"); location != "" {
				t.Errorf("unexpected canonical redirect to %q", location)
			}
			if policy := response.Header().Get("Content-Security-Policy"); !strings.Contains(policy, "script-src 'self';") {
				t.Errorf("same-origin script policy not preserved: %q", policy)
			}
		})
	}
}
