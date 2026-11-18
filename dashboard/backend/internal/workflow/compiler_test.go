package workflow

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// writeFakeCompiler 写一个假的 python 解释器, 用来测试子进程协议而不依赖 Python.
func writeFakeCompiler(t *testing.T, script string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("fake compiler script relies on POSIX shell")
	}
	path := filepath.Join(t.TempDir(), "fake-python")
	content := "#!/bin/sh\ncat > /dev/null\n" + script + "\n"
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("write fake compiler: %v", err)
	}
	return path
}

func TestCompilerCheckerReportsIssues(t *testing.T) {
	executable := writeFakeCompiler(t, `printf '%s' '{"ok":false,"action":"validate","issues":[{"code":"type_mismatch","level":"error","message":"bad edge"}],"warnings":[{"code":"missing_port","level":"warning","message":"no val"}]}'`)
	checker := NewCompilerChecker(CompilerConfig{
		PythonExecutable: executable,
		ProjectRoot:      t.TempDir(),
		Timeout:          5 * time.Second,
	})
	issues, err := checker.Validate(context.Background(), []byte(`{"version":1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(issues) != 2 {
		t.Fatalf("expected 2 issues, got %+v", issues)
	}
	if issues[0].Code != "type_mismatch" || issues[1].Level != LevelWarning {
		t.Fatalf("unexpected issues: %+v", issues)
	}
}

func TestCompilerCheckerRejectsSilentRejection(t *testing.T) {
	executable := writeFakeCompiler(t, `printf '%s' '{"ok":false,"issues":[]}'`)
	checker := NewCompilerChecker(CompilerConfig{
		PythonExecutable: executable,
		ProjectRoot:      t.TempDir(),
		Timeout:          5 * time.Second,
	})
	issues, err := checker.Validate(context.Background(), []byte(`{"version":1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !HasErrors(issues) {
		t.Fatalf("silent rejection must produce an error issue: %+v", issues)
	}
}

func TestCompilerCheckerReportsUnavailableExecutable(t *testing.T) {
	checker := NewCompilerChecker(CompilerConfig{
		PythonExecutable: filepath.Join(t.TempDir(), "missing-python"),
		ProjectRoot:      t.TempDir(),
		Timeout:          5 * time.Second,
	})
	issues, err := checker.Validate(context.Background(), []byte(`{"version":1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(issues) != 1 || issues[0].Code != CodeCompilerUnavailable {
		t.Fatalf("expected compiler_unavailable, got %+v", issues)
	}
}

func TestCompilerCheckerReportsUnparseableOutput(t *testing.T) {
	executable := writeFakeCompiler(t, `printf '%s' 'not json'`)
	checker := NewCompilerChecker(CompilerConfig{
		PythonExecutable: executable,
		ProjectRoot:      t.TempDir(),
		Timeout:          5 * time.Second,
	})
	issues, err := checker.Validate(context.Background(), []byte(`{"version":1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(issues) != 1 || issues[0].Code != CodeCompilerUnavailable {
		t.Fatalf("expected compiler_unavailable, got %+v", issues)
	}
	if !strings.Contains(issues[0].Message, "不可解析") {
		t.Fatalf("unexpected message: %s", issues[0].Message)
	}
}

func TestCompilerCheckerTimesOut(t *testing.T) {
	executable := writeFakeCompiler(t, `sleep 5`)
	checker := NewCompilerChecker(CompilerConfig{
		PythonExecutable: executable,
		ProjectRoot:      t.TempDir(),
		Timeout:          100 * time.Millisecond,
	})
	started := time.Now()
	issues, err := checker.Validate(context.Background(), []byte(`{"version":1}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if time.Since(started) > 3*time.Second {
		t.Fatal("timeout was not enforced")
	}
	if len(issues) != 1 || issues[0].Code != CodeCompilerUnavailable {
		t.Fatalf("expected compiler_unavailable, got %+v", issues)
	}
}

func TestResolvePythonExecutableFindsInterpreter(t *testing.T) {
	path, err := ResolvePythonExecutable("python3")
	if err != nil {
		t.Skipf("python3 not available: %v", err)
	}
	if path == "" {
		t.Fatal("resolved path is empty")
	}
}

func TestResolvePythonExecutableRejectsMissingPath(t *testing.T) {
	if _, err := ResolvePythonExecutable(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Fatal("missing absolute path must be rejected")
	}
}
