package workflow

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// SemanticChecker 做 XDL 语义预检. 结构校验由本包完成, 语义校验必须来自
// Python compiler, Go 不复制节点目录和组件构建逻辑.
type SemanticChecker interface {
	Validate(ctx context.Context, snapshot []byte) ([]Issue, error)
}

// CompilerConfig 是调用 Python compiler 的配置.
type CompilerConfig struct {
	PythonExecutable string
	ProjectRoot      string
	DataRoot         string
	Module           string
	Timeout          time.Duration
}

// CompilerChecker 通过子进程调用 dashboard.worker.workflow_cli 做语义预检.
type CompilerChecker struct {
	cfg CompilerConfig
}

// NewCompilerChecker 构造一个 Python compiler 预检器.
func NewCompilerChecker(cfg CompilerConfig) *CompilerChecker {
	if cfg.Module == "" {
		cfg.Module = "dashboard.worker.workflow_cli"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &CompilerChecker{cfg: cfg}
}

type compileRequest struct {
	Action   string          `json:"action"`
	Snapshot json.RawMessage `json:"snapshot"`
	Options  compileOptions  `json:"options"`
}

type compileOptions struct {
	DataRoot      string `json:"data_root,omitempty"`
	ProjectRoot   string `json:"project_root,omitempty"`
	CheckRegistry bool   `json:"check_registry"`
}

type compileResponse struct {
	OK               bool    `json:"ok"`
	Action           string  `json:"action"`
	CompilerVersion  string  `json:"compiler_version"`
	ExecutionProfile string  `json:"execution_profile"`
	Issues           []Issue `json:"issues"`
	Warnings         []Issue `json:"warnings"`
}

// Validate 调用 Python compiler 校验快照.
//
// 子进程不可用或超时都会返回 compiler_unavailable 错误问题, 而不是静默放行.
func (c *CompilerChecker) Validate(ctx context.Context, snapshot []byte) ([]Issue, error) {
	if strings.TrimSpace(c.cfg.PythonExecutable) == "" {
		return nil, errors.New("python_executable is not configured")
	}
	request := compileRequest{
		Action:   "validate",
		Snapshot: json.RawMessage(snapshot),
		Options: compileOptions{
			DataRoot:      c.cfg.DataRoot,
			ProjectRoot:   c.cfg.ProjectRoot,
			CheckRegistry: true,
		},
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode compiler request: %w", err)
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()
	command := exec.CommandContext(timeoutCtx, c.cfg.PythonExecutable, "-m", c.cfg.Module)
	// 超时后即使子进程留下了持有管道的孙进程, 也要在有限时间内返回.
	command.WaitDelay = 2 * time.Second
	command.Dir = c.cfg.ProjectRoot
	command.Stdin = bytes.NewReader(payload)
	command.Env = append(os.Environ(),
		"PYTHONPATH="+prependPathList(c.cfg.ProjectRoot, os.Getenv("PYTHONPATH")),
	)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		message := strings.TrimSpace(stderr.String())
		if message == "" {
			message = err.Error()
		}
		return []Issue{ErrorIssue(
			CodeCompilerUnavailable,
			"无法运行 workflow compiler: "+message,
			"",
		)}, nil
	}
	var response compileResponse
	if err := json.Unmarshal(stdout.Bytes(), &response); err != nil {
		return []Issue{ErrorIssue(
			CodeCompilerUnavailable,
			"workflow compiler 返回了不可解析的结果: "+err.Error(),
			"",
		)}, nil
	}
	issues := make([]Issue, 0, len(response.Issues)+len(response.Warnings))
	issues = append(issues, response.Issues...)
	issues = append(issues, response.Warnings...)
	if !response.OK && !HasErrors(issues) {
		issues = append(issues, ErrorIssue(
			CodeInvalidSnapshot,
			"workflow compiler 拒绝该快照但没有给出具体问题",
			"",
		))
	}
	return issues, nil
}

func prependPathList(root, existing string) string {
	if existing == "" {
		return root
	}
	return root + string(os.PathListSeparator) + existing
}

// ResolvePythonExecutable 解析配置里的解释器路径.
func ResolvePythonExecutable(configured string) (string, error) {
	if configured == "" {
		return "", errors.New("python_executable is not configured")
	}
	if filepath.IsAbs(configured) {
		if _, err := os.Stat(configured); err != nil {
			return "", fmt.Errorf("python executable %q: %w", configured, err)
		}
		return configured, nil
	}
	if path, err := exec.LookPath(configured); err == nil {
		return path, nil
	}
	if configured == "python" {
		for _, candidate := range []string{"python3", "py"} {
			if path, err := exec.LookPath(candidate); err == nil {
				return path, nil
			}
		}
	}
	return "", fmt.Errorf("python executable %q was not found", configured)
}
