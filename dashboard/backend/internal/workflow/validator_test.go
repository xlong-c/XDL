package workflow

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func decodeFixture(t *testing.T, name string) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(loadFixture(t, name), &payload); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
	return payload
}

func encodePayload(t *testing.T, payload map[string]any) []byte {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("encode payload: %v", err)
	}
	return raw
}

func issueCodes(issues []Issue) map[string]bool {
	codes := make(map[string]bool, len(issues))
	for _, issue := range issues {
		codes[issue.Code] = true
	}
	return codes
}

func preparePayload(t *testing.T, payload map[string]any) (Snapshot, []Issue) {
	t.Helper()
	return Prepare(encodePayload(t, payload), testCatalog(t))
}

func nodeOf(t *testing.T, payload map[string]any, id string) map[string]any {
	t.Helper()
	nodes, ok := payload["nodes"].([]any)
	if !ok {
		t.Fatal("nodes is not a list")
	}
	for _, raw := range nodes {
		node, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if node["id"] == id {
			return node
		}
	}
	t.Fatalf("node %q not found", id)
	return nil
}

func TestValidateRejectsMissingVersion(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	delete(payload, "version")
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeUnsupportedVersion] {
		t.Fatalf("expected unsupported_version, got %v", issues)
	}
}

func TestValidateRejectsUnknownVersion(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	payload["version"] = 7
	_, issues := preparePayload(t, payload)
	if len(issues) != 1 || issues[0].Code != CodeUnsupportedVersion {
		t.Fatalf("expected only unsupported_version, got %v", issues)
	}
}

func TestValidateRejectsMissingProfile(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	delete(payload, "execution_profile")
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeInvalidSnapshot] {
		t.Fatalf("expected invalid_snapshot, got %v", issues)
	}
}

func TestValidateRejectsUnknownProfile(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	payload["execution_profile"] = "gpu-magic"
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeUnsupportedProfile] {
		t.Fatalf("expected unsupported_profile, got %v", issues)
	}
}

func TestValidateRejectsDuplicateNodeID(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	duplicate := nodeOf(t, payload, "model")
	nodes := payload["nodes"].([]any)
	payload["nodes"] = append(nodes, duplicate)
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeDuplicateID] {
		t.Fatalf("expected duplicate_id, got %v", issues)
	}
}

func TestValidateRejectsUnknownNodeReference(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	edges := payload["edges"].([]any)
	first := edges[0].(map[string]any)
	first["to"].(map[string]any)["node"] = "ghost"
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeUnknownNode] {
		t.Fatalf("expected unknown_node, got %v", issues)
	}
}

func TestValidateRejectsSelfLoop(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	edges := payload["edges"].([]any)
	edges = append(edges, map[string]any{
		"id":   "e-self",
		"from": map[string]any{"node": "dl", "port": "batches"},
		"to":   map[string]any{"node": "dl", "port": "dataset"},
	})
	payload["edges"] = edges
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeSelfLoop] {
		t.Fatalf("expected self_loop, got %v", issues)
	}
}

func TestValidateRejectsDuplicateEdge(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	edges := payload["edges"].([]any)
	duplicate := map[string]any{}
	for key, value := range edges[0].(map[string]any) {
		duplicate[key] = value
	}
	duplicate["id"] = "e-dup"
	payload["edges"] = append(edges, duplicate)
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeDuplicateEdge] {
		t.Fatalf("expected duplicate_edge, got %v", issues)
	}
}

func TestValidateRejectsCycle(t *testing.T) {
	payload := map[string]any{
		"version":           1,
		"name":              "cycle",
		"execution_profile": "schema-v1",
		"nodes": []any{
			map[string]any{"id": "a", "kind": "transform", "target": "t:op"},
			map[string]any{"id": "b", "kind": "transform", "target": "t:op"},
		},
		"edges": []any{
			map[string]any{"id": "e1", "from": map[string]any{"node": "a", "port": "pipeline"}, "to": map[string]any{"node": "b", "port": "pipeline"}},
			map[string]any{"id": "e2", "from": map[string]any{"node": "b", "port": "pipeline"}, "to": map[string]any{"node": "a", "port": "pipeline"}},
		},
	}
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeCycleDetected] {
		t.Fatalf("expected cycle_detected, got %v", issues)
	}
}

func TestValidateRejectsOversizedSnapshot(t *testing.T) {
	catalog := testCatalog(t)
	catalog.Limits.MaxSnapshotBytes = 64
	_, issues := Prepare(loadFixture(t, "synthetic-cpu-demo.json"), catalog)
	if !issueCodes(issues)[CodeLimitExceeded] {
		t.Fatalf("expected limit_exceeded, got %v", issues)
	}
}

func TestValidateRejectsTooManyNodes(t *testing.T) {
	catalog := testCatalog(t)
	catalog.Limits.MaxNodes = 2
	_, issues := Prepare(loadFixture(t, "synthetic-cpu-demo.json"), catalog)
	if !issueCodes(issues)[CodeLimitExceeded] {
		t.Fatalf("expected limit_exceeded, got %v", issues)
	}
}

func TestValidateRejectsDeepParams(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	trainer := nodeOf(t, payload, "trainer")
	nested := any(1)
	for index := 0; index < 12; index++ {
		nested = map[string]any{"a": nested}
	}
	trainer["params"] = map[string]any{"max_epochs": 1, "deep": nested}
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeLimitExceeded] {
		t.Fatalf("expected limit_exceeded, got %v", issues)
	}
}

func TestValidateRejectsOversizedParams(t *testing.T) {
	payload := decodeFixture(t, "synthetic-cpu-demo.json")
	trainer := nodeOf(t, payload, "trainer")
	trainer["params"] = map[string]any{"max_epochs": 1, "blob": strings.Repeat("x", 70000)}
	_, issues := preparePayload(t, payload)
	if !issueCodes(issues)[CodeLimitExceeded] {
		t.Fatalf("expected limit_exceeded, got %v", issues)
	}
}

func TestValidateAcceptsEveryFixtureWithGeneratedEdgeIDs(t *testing.T) {
	for _, name := range []string{"cifar10-demo.json", "synthetic-cpu-demo.json", "dag-runtime-demo.json"} {
		payload := decodeFixture(t, name)
		edges := payload["edges"].([]any)
		for _, raw := range edges {
			delete(raw.(map[string]any), "id")
		}
		want := len(edges)
		snapshot, issues := preparePayload(t, payload)
		if HasErrors(issues) {
			t.Fatalf("%s rejected: %s", name, ErrorMessages(issues))
		}
		if len(snapshot.Edges) != want {
			t.Fatalf("%s edge count = %d, want %d", name, len(snapshot.Edges), want)
		}
		// 生成 ID 按原始索引编号, canonical 化后按 ID 字典序排序.
		seen := make(map[string]bool, want)
		for _, edge := range snapshot.Edges {
			if !strings.HasPrefix(edge.ID, "edge-") {
				t.Fatalf("%s edge id %q was not generated", name, edge.ID)
			}
			if seen[edge.ID] {
				t.Fatalf("%s edge id %q duplicated", name, edge.ID)
			}
			seen[edge.ID] = true
		}
		for index := 1; index <= want; index++ {
			if !seen[fmt.Sprintf("edge-%d", index)] {
				t.Fatalf("%s missing generated id edge-%d", name, index)
			}
		}
	}
}
