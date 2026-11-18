// Package workflow 实现 WorkflowSnapshot v1 的 Go 侧契约.
//
// 职责边界: 本包只做 JSON envelope, 结构校验, 规模限制, 引用完整性与环检测, 以及
// canonical JSON 和哈希. 节点类型, 端口语义, target 与参数语义属于 Python compiler,
// 本包不复制一份组件目录.
package workflow

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
)

const (
	// SnapshotVersion 是唯一受支持的快照版本.
	SnapshotVersion = 1
	// ProfileSchemaV1 表示编译到现有 XDL schema v1 YAML.
	ProfileSchemaV1 = "schema-v1"
	// ProfileDagRuntime 表示走 XDL workflow runtime.
	ProfileDagRuntime = "dag-runtime"
)

// Profiles 是受支持的执行 profile.
var Profiles = []string{ProfileSchemaV1, ProfileDagRuntime}

// Position 是画布坐标, 属于 UI 元数据.
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// PortRef 是一条边的端点.
type PortRef struct {
	Node string `json:"node"`
	Port string `json:"port"`
}

// Node 是一个图节点.
type Node struct {
	ID       string         `json:"id"`
	Kind     string         `json:"kind"`
	Target   string         `json:"target"`
	Params   map[string]any `json:"params"`
	Position *Position      `json:"position,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Edge 是一条有类型边.
type Edge struct {
	ID   string  `json:"id"`
	From PortRef `json:"from"`
	To   PortRef `json:"to"`
}

// Settings 是排队与运行默认值.
type Settings struct {
	Launcher string         `json:"launcher,omitempty"`
	Devices  []int          `json:"devices,omitempty"`
	Runtime  map[string]any `json:"runtime,omitempty"`
}

type StepGraph struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// Snapshot 是 WorkflowSnapshot v1.
type Snapshot struct {
	Version          int        `json:"version"`
	ID               string     `json:"id,omitempty"`
	Name             string     `json:"name"`
	ExecutionProfile string     `json:"execution_profile"`
	Nodes            []Node     `json:"nodes"`
	Edges            []Edge     `json:"edges"`
	Settings         Settings   `json:"settings"`
	StepGraph        *StepGraph `json:"step_graph,omitempty"`
}

// DecodeSnapshot 严格解码一份快照 JSON, 数字按字面量保留.
func DecodeSnapshot(raw []byte) (Snapshot, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	var snapshot Snapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return Snapshot{}, fmt.Errorf("decode workflow snapshot: %w", err)
	}
	if err := ensureEOF(decoder); err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}

func ensureEOF(decoder *json.Decoder) error {
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("decode workflow snapshot: trailing data after JSON object")
		}
		return fmt.Errorf("decode workflow snapshot: %w", err)
	}
	return nil
}

// Canonicalize 返回规范化快照: 节点与边按 ID 排序, 保证稳定字节.
func (s Snapshot) Canonicalize() Snapshot {
	result := s
	result.Nodes = append([]Node(nil), s.Nodes...)
	result.Edges = append([]Edge(nil), s.Edges...)
	sort.Slice(result.Nodes, func(i, j int) bool { return result.Nodes[i].ID < result.Nodes[j].ID })
	sort.Slice(result.Edges, func(i, j int) bool { return result.Edges[i].ID < result.Edges[j].ID })
	for index := range result.Nodes {
		if result.Nodes[index].Params == nil {
			result.Nodes[index].Params = map[string]any{}
		}
	}
	if s.StepGraph != nil {
		step := (Snapshot{Nodes: s.StepGraph.Nodes, Edges: s.StepGraph.Edges}).Canonicalize()
		result.StepGraph = &StepGraph{Nodes: step.Nodes, Edges: step.Edges}
	}
	return result
}

// CanonicalJSON 返回规范化 JSON 字节. map 键由 encoding/json 自动排序.
func (s Snapshot) CanonicalJSON() ([]byte, error) {
	return json.Marshal(s.Canonicalize())
}

// Hash 返回规范化 JSON 的 SHA-256.
func (s Snapshot) Hash() (string, error) {
	raw, err := s.CanonicalJSON()
	if err != nil {
		return "", err
	}
	return HashBytes(raw), nil
}

// NewID 生成带前缀的随机标识, 用于 workflow id.
func NewID(prefix string) (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate workflow id: %w", err)
	}
	return prefix + hex.EncodeToString(value[:8]), nil
}

// HashBytes 返回字节的 SHA-256 小写十六进制.
func HashBytes(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// NodeByID 按 ID 查找节点.
func (s Snapshot) NodeByID(id string) (Node, bool) {
	for _, node := range s.Nodes {
		if node.ID == id {
			return node, true
		}
	}
	return Node{}, false
}

// TopologicalOrder 返回稳定拓扑序; 存在环时返回错误.
func (s Snapshot) TopologicalOrder() ([]string, error) {
	indegree := make(map[string]int, len(s.Nodes))
	outgoing := make(map[string][]string, len(s.Nodes))
	for _, node := range s.Nodes {
		indegree[node.ID] = 0
		outgoing[node.ID] = nil
	}
	for _, edge := range s.Edges {
		if _, ok := indegree[edge.To.Node]; !ok {
			continue
		}
		if _, ok := outgoing[edge.From.Node]; !ok {
			continue
		}
		indegree[edge.To.Node]++
		outgoing[edge.From.Node] = append(outgoing[edge.From.Node], edge.To.Node)
	}
	ready := make([]string, 0, len(indegree))
	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}
	sort.Strings(ready)
	order := make([]string, 0, len(s.Nodes))
	for len(ready) > 0 {
		next := make([]string, 0, len(ready))
		for _, id := range ready {
			order = append(order, id)
			for _, target := range outgoing[id] {
				indegree[target]--
				if indegree[target] == 0 {
					next = append(next, target)
				}
			}
		}
		sort.Strings(next)
		ready = next
	}
	if len(order) != len(s.Nodes) {
		return nil, errors.New("snapshot contains a cycle")
	}
	return order, nil
}
