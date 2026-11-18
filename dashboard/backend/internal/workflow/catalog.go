package workflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
)

// Limits 是快照规模上限.
//
// 这些数字的唯一来源是节点目录 JSON 的 limits 键. Go 只读取这一处内容, 目的是让
// 存储层限制和 Python compiler 的限制不会各自漂移; 目录的其余部分原样转发给前端.
type Limits struct {
	MaxNodes         int `json:"max_nodes"`
	MaxEdges         int `json:"max_edges"`
	MaxParamsDepth   int `json:"max_params_depth"`
	MaxParamsBytes   int `json:"max_params_bytes"`
	MaxSnapshotBytes int `json:"max_snapshot_bytes"`
	MaxNameLength    int `json:"max_name_length"`
	MaxIDLength      int `json:"max_id_length"`
}

// DefaultLimits 是目录缺失 limits 键时的兜底值, 与 Python 侧默认值一致.
func DefaultLimits() Limits {
	return Limits{
		MaxNodes:         64,
		MaxEdges:         256,
		MaxParamsDepth:   6,
		MaxParamsBytes:   65536,
		MaxSnapshotBytes: 262144,
		MaxNameLength:    120,
		MaxIDLength:      64,
	}
}

// Catalog 是节点目录的 dashboard 侧视图.
type Catalog struct {
	Raw      []byte
	Version  int
	Limits   Limits
	profiles []string
}

type catalogFile struct {
	Version  int               `json:"version"`
	Limits   Limits            `json:"limits"`
	Profiles map[string]string `json:"profiles"`
}

// LoadCatalog 读取节点目录 JSON.
func LoadCatalog(path string) (*Catalog, error) {
	if path == "" {
		return nil, errors.New("workflow catalog path cannot be empty")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read workflow node catalog: %w", err)
	}
	var file catalogFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("decode workflow node catalog: %w", err)
	}
	if file.Version != 1 {
		return nil, fmt.Errorf("unsupported workflow node catalog version %d", file.Version)
	}
	limits := file.Limits
	if limits.MaxNodes <= 0 {
		limits = DefaultLimits()
	}
	profiles := make([]string, 0, len(file.Profiles))
	for name := range file.Profiles {
		profiles = append(profiles, name)
	}
	sort.Strings(profiles)
	if len(profiles) == 0 {
		profiles = append([]string(nil), Profiles...)
	}
	return &Catalog{Raw: raw, Version: file.Version, Limits: limits, profiles: profiles}, nil
}

// Profiles 返回目录声明的执行 profile.
func (c *Catalog) Profiles() []string {
	if c == nil {
		return append([]string(nil), Profiles...)
	}
	return append([]string(nil), c.profiles...)
}

// SupportsProfile 判断 profile 是否受支持.
func (c *Catalog) SupportsProfile(name string) bool {
	for _, profile := range c.Profiles() {
		if profile == name {
			return true
		}
	}
	return false
}
