package workflow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testCatalog(t *testing.T) *Catalog {
	t.Helper()
	path := filepath.Join("..", "..", "..", "..", "xdl", "workflow", "node_catalog.json")
	catalog, err := LoadCatalog(path)
	if err != nil {
		t.Fatalf("load node catalog: %v", err)
	}
	return catalog
}

func fixturePath(name string) string {
	return filepath.Join("..", "..", "..", "templates", "workflows", name)
}

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(fixturePath(name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return raw
}

func TestPrepareAcceptsSharedFixtures(t *testing.T) {
	catalog := testCatalog(t)
	for _, name := range []string{"cifar10-demo.json", "synthetic-cpu-demo.json", "dag-runtime-demo.json"} {
		snapshot, issues := Prepare(loadFixture(t, name), catalog)
		if HasErrors(issues) {
			t.Fatalf("%s rejected: %s", name, ErrorMessages(issues))
		}
		if snapshot.Version != SnapshotVersion {
			t.Fatalf("%s version = %d", name, snapshot.Version)
		}
		if _, err := snapshot.TopologicalOrder(); err != nil {
			t.Fatalf("%s topological order: %v", name, err)
		}
	}
}

func TestCanonicalJSONIsStableAcrossOrdering(t *testing.T) {
	catalog := testCatalog(t)
	snapshot, issues := Prepare(loadFixture(t, "synthetic-cpu-demo.json"), catalog)
	if HasErrors(issues) {
		t.Fatalf("fixture rejected: %s", ErrorMessages(issues))
	}
	reversed := snapshot
	reversed.Nodes = append([]Node(nil), snapshot.Nodes...)
	reversed.Edges = append([]Edge(nil), snapshot.Edges...)
	for i, j := 0, len(reversed.Nodes)-1; i < j; i, j = i+1, j-1 {
		reversed.Nodes[i], reversed.Nodes[j] = reversed.Nodes[j], reversed.Nodes[i]
	}
	for i, j := 0, len(reversed.Edges)-1; i < j; i, j = i+1, j-1 {
		reversed.Edges[i], reversed.Edges[j] = reversed.Edges[j], reversed.Edges[i]
	}
	first, err := snapshot.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	second, err := reversed.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("canonical JSON differs:\n%s\n%s", first, second)
	}
	if HashBytes(first) != HashBytes(second) {
		t.Fatal("canonical hash differs")
	}
}

func TestCanonicalJSONKeepsNumericLiterals(t *testing.T) {
	catalog := testCatalog(t)
	raw := []byte(`{
		"version": 1,
		"name": "numbers",
		"execution_profile": "schema-v1",
		"nodes": [
			{"id": "a", "kind": "trainer", "target": "builtin:Trainer", "params": {"max_epochs": 1, "grad_clip_max_norm": 5.0}}
		],
		"edges": []
	}`)
	snapshot, issues := Prepare(raw, catalog)
	if HasErrors(issues) {
		t.Fatalf("rejected: %s", ErrorMessages(issues))
	}
	canonical, err := snapshot.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	text := string(canonical)
	if !strings.Contains(text, `"max_epochs":1`) {
		t.Fatalf("integer literal was not preserved: %s", text)
	}
	if !strings.Contains(text, `"grad_clip_max_norm":5.0`) {
		t.Fatalf("float literal was not preserved: %s", text)
	}
}

func TestDecodeSnapshotRejectsUnknownFields(t *testing.T) {
	raw := []byte(`{"version": 1, "name": "x", "execution_profile": "schema-v1", "nodes": [], "edges": [], "surprise": 1}`)
	if _, err := DecodeSnapshot(raw); err == nil {
		t.Fatal("unknown field must be rejected")
	}
}

func TestDecodeSnapshotRejectsTrailingData(t *testing.T) {
	raw := []byte(`{"version": 1, "name": "x", "execution_profile": "schema-v1", "nodes": [], "edges": []} {}`)
	if _, err := DecodeSnapshot(raw); err == nil {
		t.Fatal("trailing data must be rejected")
	}
}

func TestNormalizeGeneratesMissingEdgeIDs(t *testing.T) {
	raw := []byte(`{
		"version": 1,
		"name": "ids",
		"execution_profile": "schema-v1",
		"nodes": [
			{"id": "a", "kind": "transform", "target": "t:op"},
			{"id": "b", "kind": "transform", "target": "t:op"}
		],
		"edges": [{"from": {"node": "a", "port": "pipeline"}, "to": {"node": "b", "port": "pipeline"}}]
	}`)
	catalog := testCatalog(t)
	snapshot, issues := Prepare(raw, catalog)
	if HasErrors(issues) {
		t.Fatalf("rejected: %s", ErrorMessages(issues))
	}
	if len(snapshot.Edges) != 1 || snapshot.Edges[0].ID != "edge-1" {
		t.Fatalf("edge id was not generated: %+v", snapshot.Edges)
	}
	if snapshot.Nodes[0].Params == nil {
		t.Fatal("params must be normalized to an object")
	}
}

func TestLoadCatalogReadsLimitsFromSharedFile(t *testing.T) {
	catalog := testCatalog(t)
	if catalog.Limits.MaxNodes <= 0 || catalog.Limits.MaxSnapshotBytes <= 0 {
		t.Fatalf("limits not loaded: %+v", catalog.Limits)
	}
	if !catalog.SupportsProfile(ProfileSchemaV1) || !catalog.SupportsProfile(ProfileDagRuntime) {
		t.Fatalf("profiles not loaded: %+v", catalog.Profiles())
	}
	var decoded map[string]any
	if err := json.Unmarshal(catalog.Raw, &decoded); err != nil {
		t.Fatalf("raw catalog must stay valid JSON: %v", err)
	}
}

func TestNewIDUsesPrefixAndIsRandom(t *testing.T) {
	first, err := NewID("wf-")
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewID("wf-")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(first, "wf-") || first == second {
		t.Fatalf("unexpected ids: %q %q", first, second)
	}
}
