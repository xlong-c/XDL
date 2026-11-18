package workflow

import (
	"encoding/json"
	"fmt"
	"strings"
)

// 结构层错误码, 与 xdl/workflow/errors.py 保持一致.
const (
	CodeInvalidSnapshot     = "invalid_snapshot"
	CodeUnsupportedVersion  = "unsupported_version"
	CodeDuplicateID         = "duplicate_id"
	CodeUnknownNode         = "unknown_node"
	CodeDuplicateEdge       = "duplicate_edge"
	CodeSelfLoop            = "self_loop"
	CodeCycleDetected       = "cycle_detected"
	CodeLimitExceeded       = "limit_exceeded"
	CodeUnsupportedProfile  = "unsupported_profile"
	CodeCompilerUnavailable = "compiler_unavailable"
)

// LevelError 与 LevelWarning 是问题的两个级别.
const (
	LevelError   = "error"
	LevelWarning = "warning"
)

// Issue 是一条可定位的校验问题.
type Issue struct {
	Code    string `json:"code"`
	Level   string `json:"level"`
	Message string `json:"message"`
	Path    string `json:"path,omitempty"`
	NodeID  string `json:"node_id,omitempty"`
	EdgeID  string `json:"edge_id,omitempty"`
}

// ErrorIssue 构造 error 级问题.
func ErrorIssue(code, message, path string) Issue {
	return Issue{Code: code, Level: LevelError, Message: message, Path: path}
}

// WarningIssue 构造 warning 级问题.
func WarningIssue(code, message, path string) Issue {
	return Issue{Code: code, Level: LevelWarning, Message: message, Path: path}
}

// HasErrors 判断是否存在阻断错误.
func HasErrors(issues []Issue) bool {
	for _, issue := range issues {
		if issue.Level == LevelError {
			return true
		}
	}
	return false
}

// ErrorMessages 把问题拼成一行错误信息.
func ErrorMessages(issues []Issue) string {
	parts := make([]string, 0, len(issues))
	for _, issue := range issues {
		parts = append(parts, issue.Code+": "+issue.Message)
	}
	return strings.Join(parts, "; ")
}

// Prepare 解码并规范化一份快照, 返回规范化结果与结构问题.
//
// 规范化包含: 补齐缺失的边 ID, 保证 params 不是 nil, 节点与边按 ID 排序.
func Prepare(raw []byte, catalog *Catalog) (Snapshot, []Issue) {
	limits := catalogLimits(catalog)
	if limits.MaxSnapshotBytes > 0 && len(raw) > limits.MaxSnapshotBytes {
		return Snapshot{}, []Issue{ErrorIssue(
			CodeLimitExceeded,
			fmt.Sprintf("快照大小 %d 字节超过上限 %d", len(raw), limits.MaxSnapshotBytes),
			"",
		)}
	}
	snapshot, err := DecodeSnapshot(raw)
	if err != nil {
		return Snapshot{}, []Issue{ErrorIssue(CodeInvalidSnapshot, err.Error(), "")}
	}
	normalized := Normalize(snapshot)
	return normalized, Validate(normalized, catalog)
}

func catalogLimits(catalog *Catalog) Limits {
	if catalog == nil {
		return DefaultLimits()
	}
	return catalog.Limits
}

// Normalize 补齐边 ID 并保证 params 为对象.
func Normalize(snapshot Snapshot) Snapshot {
	result := snapshot
	result.Nodes = append([]Node(nil), snapshot.Nodes...)
	for index := range result.Nodes {
		if result.Nodes[index].Params == nil {
			result.Nodes[index].Params = map[string]any{}
		}
	}
	result.Edges = append([]Edge(nil), snapshot.Edges...)
	for index := range result.Edges {
		if strings.TrimSpace(result.Edges[index].ID) == "" {
			result.Edges[index].ID = fmt.Sprintf("edge-%d", index+1)
		}
	}
	if snapshot.StepGraph != nil {
		step := Normalize(Snapshot{Nodes: snapshot.StepGraph.Nodes, Edges: snapshot.StepGraph.Edges})
		result.StepGraph = &StepGraph{Nodes: step.Nodes, Edges: step.Edges}
	}
	return result.Canonicalize()
}

// Validate 校验快照结构与规模.
func Validate(snapshot Snapshot, catalog *Catalog) []Issue {
	limits := catalogLimits(catalog)
	issues := make([]Issue, 0)
	if snapshot.Version != SnapshotVersion {
		return append(issues, ErrorIssue(
			CodeUnsupportedVersion,
			fmt.Sprintf("不支持的 workflow snapshot version %d, 只接受 %d", snapshot.Version, SnapshotVersion),
			"version",
		))
	}
	issues = append(issues, validateEnvelope(snapshot, limits, catalog)...)
	issues = append(issues, validateNodes(snapshot.Nodes, limits)...)
	issues = append(issues, validateEdges(snapshot, limits)...)
	if _, err := snapshot.TopologicalOrder(); err != nil {
		issues = append(issues, ErrorIssue(CodeCycleDetected, "节点图存在环: "+err.Error(), "edges"))
	}
	if snapshot.StepGraph != nil {
		step := Snapshot{Nodes: snapshot.StepGraph.Nodes, Edges: snapshot.StepGraph.Edges}
		stepIssues := append(validateNodes(step.Nodes, limits), validateEdges(step, limits)...)
		if _, err := step.TopologicalOrder(); err != nil {
			stepIssues = append(stepIssues, ErrorIssue(CodeCycleDetected, "训练步图存在环", "edges"))
		}
		for _, issue := range stepIssues {
			issue.Path = "step_graph/" + issue.Path
			issues = append(issues, issue)
		}
	}
	return issues
}

func validateEnvelope(snapshot Snapshot, limits Limits, catalog *Catalog) []Issue {
	issues := make([]Issue, 0)
	name := strings.TrimSpace(snapshot.Name)
	if name == "" {
		issues = append(issues, ErrorIssue(CodeInvalidSnapshot, "name 必须是非空字符串", "name"))
	} else if limits.MaxNameLength > 0 && len(snapshot.Name) > limits.MaxNameLength {
		issues = append(issues, ErrorIssue(
			CodeLimitExceeded,
			fmt.Sprintf("name 长度 %d 超过上限 %d", len(snapshot.Name), limits.MaxNameLength),
			"name",
		))
	}
	if limits.MaxIDLength > 0 && len(snapshot.ID) > limits.MaxIDLength {
		issues = append(issues, ErrorIssue(
			CodeLimitExceeded,
			fmt.Sprintf("id 长度超过上限 %d", limits.MaxIDLength),
			"id",
		))
	}
	profile := strings.TrimSpace(snapshot.ExecutionProfile)
	if profile == "" {
		issues = append(issues, ErrorIssue(
			CodeInvalidSnapshot,
			"execution_profile 必须显式声明, 不接受默认猜测",
			"execution_profile",
		))
	} else if !catalogSupportsProfile(catalog, profile) {
		issues = append(issues, ErrorIssue(
			CodeUnsupportedProfile,
			fmt.Sprintf("不支持的 execution_profile %q", profile),
			"execution_profile",
		))
	}
	return issues
}

func catalogSupportsProfile(catalog *Catalog, profile string) bool {
	if catalog == nil {
		for _, candidate := range Profiles {
			if candidate == profile {
				return true
			}
		}
		return false
	}
	return catalog.SupportsProfile(profile)
}

func validateNodes(nodes []Node, limits Limits) []Issue {
	issues := make([]Issue, 0)
	if len(nodes) == 0 {
		issues = append(issues, ErrorIssue(CodeInvalidSnapshot, "nodes 不能为空", "nodes"))
	}
	if limits.MaxNodes > 0 && len(nodes) > limits.MaxNodes {
		issues = append(issues, ErrorIssue(
			CodeLimitExceeded,
			fmt.Sprintf("节点数 %d 超过上限 %d", len(nodes), limits.MaxNodes),
			"nodes",
		))
	}
	seen := make(map[string]int, len(nodes))
	for index, node := range nodes {
		path := fmt.Sprintf("nodes/%d", index)
		id := strings.TrimSpace(node.ID)
		if id == "" {
			issues = append(issues, ErrorIssue(CodeInvalidSnapshot, "节点 id 必须是非空字符串", path+"/id"))
			continue
		}
		if limits.MaxIDLength > 0 && len(node.ID) > limits.MaxIDLength {
			issues = append(issues, ErrorIssue(
				CodeLimitExceeded,
				fmt.Sprintf("节点 id 长度超过上限 %d", limits.MaxIDLength),
				path+"/id",
			))
		}
		if previous, ok := seen[node.ID]; ok {
			issues = append(issues, Issue{
				Code:    CodeDuplicateID,
				Level:   LevelError,
				Message: fmt.Sprintf("节点 id %q 重复 (首次出现在 nodes/%d)", node.ID, previous),
				Path:    path + "/id",
				NodeID:  node.ID,
			})
			continue
		}
		seen[node.ID] = index
		if strings.TrimSpace(node.Kind) == "" {
			issues = append(issues, Issue{
				Code:    CodeInvalidSnapshot,
				Level:   LevelError,
				Message: "节点 kind 必须是非空字符串",
				Path:    path + "/kind",
				NodeID:  node.ID,
			})
			continue
		}
		if strings.TrimSpace(node.Target) == "" {
			issues = append(issues, Issue{
				Code:    CodeInvalidSnapshot,
				Level:   LevelError,
				Message: "节点 target 必须是非空字符串",
				Path:    path + "/target",
				NodeID:  node.ID,
			})
			continue
		}
		issues = append(issues, validateParams(node.Params, limits, path+"/params", node.ID)...)
	}
	return issues
}

func validateParams(params map[string]any, limits Limits, path, nodeID string) []Issue {
	issues := make([]Issue, 0)
	encoded, err := json.Marshal(params)
	if err != nil {
		return append(issues, Issue{
			Code:    CodeInvalidSnapshot,
			Level:   LevelError,
			Message: "params 无法序列化为 JSON: " + err.Error(),
			Path:    path,
			NodeID:  nodeID,
		})
	}
	if limits.MaxParamsBytes > 0 && len(encoded) > limits.MaxParamsBytes {
		issues = append(issues, Issue{
			Code:    CodeLimitExceeded,
			Level:   LevelError,
			Message: fmt.Sprintf("params 大小 %d 字节超过上限 %d", len(encoded), limits.MaxParamsBytes),
			Path:    path,
			NodeID:  nodeID,
		})
	}
	if limits.MaxParamsDepth > 0 {
		if depth := maxDepth(params); depth > limits.MaxParamsDepth {
			issues = append(issues, Issue{
				Code:    CodeLimitExceeded,
				Level:   LevelError,
				Message: fmt.Sprintf("params 嵌套深度 %d 超过上限 %d", depth, limits.MaxParamsDepth),
				Path:    path,
				NodeID:  nodeID,
			})
		}
	}
	return issues
}

func maxDepth(value any) int {
	switch typed := value.(type) {
	case map[string]any:
		deepest := 0
		for _, item := range typed {
			if depth := maxDepth(item); depth > deepest {
				deepest = depth
			}
		}
		return deepest + 1
	case []any:
		deepest := 0
		for _, item := range typed {
			if depth := maxDepth(item); depth > deepest {
				deepest = depth
			}
		}
		return deepest + 1
	default:
		return 0
	}
}

func validateEdges(snapshot Snapshot, limits Limits) []Issue {
	issues := make([]Issue, 0)
	if limits.MaxEdges > 0 && len(snapshot.Edges) > limits.MaxEdges {
		issues = append(issues, ErrorIssue(
			CodeLimitExceeded,
			fmt.Sprintf("边数 %d 超过上限 %d", len(snapshot.Edges), limits.MaxEdges),
			"edges",
		))
	}
	nodeIDs := make(map[string]struct{}, len(snapshot.Nodes))
	for _, node := range snapshot.Nodes {
		nodeIDs[node.ID] = struct{}{}
	}
	seenIDs := make(map[string]int, len(snapshot.Edges))
	seenPairs := make(map[string]string, len(snapshot.Edges))
	for index, edge := range snapshot.Edges {
		path := fmt.Sprintf("edges/%d", index)
		if previous, ok := seenIDs[edge.ID]; ok {
			issues = append(issues, Issue{
				Code:    CodeDuplicateID,
				Level:   LevelError,
				Message: fmt.Sprintf("边 id %q 重复 (首次出现在 edges/%d)", edge.ID, previous),
				Path:    path + "/id",
				EdgeID:  edge.ID,
			})
			continue
		}
		seenIDs[edge.ID] = index
		missing := false
		for side, ref := range map[string]PortRef{"from": edge.From, "to": edge.To} {
			if _, ok := nodeIDs[ref.Node]; !ok {
				issues = append(issues, Issue{
					Code:    CodeUnknownNode,
					Level:   LevelError,
					Message: fmt.Sprintf("边 %s 的 %s 引用不存在的节点 %q", edge.ID, side, ref.Node),
					Path:    path + "/" + side + "/node",
					EdgeID:  edge.ID,
				})
				missing = true
			}
		}
		if missing {
			continue
		}
		if edge.From.Node == edge.To.Node {
			issues = append(issues, Issue{
				Code:    CodeSelfLoop,
				Level:   LevelError,
				Message: fmt.Sprintf("边 %s 连接了节点自身 %q", edge.ID, edge.From.Node),
				Path:    path,
				NodeID:  edge.From.Node,
				EdgeID:  edge.ID,
			})
			continue
		}
		key := edge.From.Node + "\x00" + edge.From.Port + "\x00" + edge.To.Node + "\x00" + edge.To.Port
		if previous, ok := seenPairs[key]; ok {
			issues = append(issues, Issue{
				Code:    CodeDuplicateEdge,
				Level:   LevelError,
				Message: fmt.Sprintf("边 %s 与 %s 重复连接", edge.ID, previous),
				Path:    path,
				EdgeID:  edge.ID,
			})
			continue
		}
		seenPairs[key] = edge.ID
	}
	return issues
}
