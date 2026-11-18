package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"
)

type Config struct {
	ListenAddr          string        `json:"listen_addr"`
	DatabasePath        string        `json:"database_path"`
	ProjectRoot         string        `json:"project_root"`
	PythonExecutable    string        `json:"python_executable"`
	DataRoot            string        `json:"data_root"`
	OutputRoot          string        `json:"output_root"`
	FrontendDir         string        `json:"frontend_dir"`
	TemplateCatalog     string        `json:"template_catalog"`
	WorkflowCatalog     string        `json:"workflow_catalog"`
	WorkerModule        string        `json:"worker_module"`
	SystemInterval      time.Duration `json:"-"`
	SystemIntervalText  string        `json:"system_interval"`
	MetricsInterval     time.Duration `json:"-"`
	MetricsIntervalText string        `json:"metrics_interval"`
	CompilerTimeout     time.Duration `json:"-"`
	CompilerTimeoutText string        `json:"compiler_timeout"`
	EventBuffer         int           `json:"event_buffer"`
	MaxParallelRuns     int           `json:"max_parallel_runs"`
}

func Default() Config {
	return Config{
		ListenAddr:          "127.0.0.1:8765",
		DatabasePath:        "./others/dashboard/dashboard.sqlite",
		ProjectRoot:         ".",
		PythonExecutable:    "python",
		DataRoot:            "./data",
		OutputRoot:          "./others/dashboard/runs",
		FrontendDir:         "./dashboard/frontend/dist",
		TemplateCatalog:     "./dashboard/templates/catalog.json",
		WorkflowCatalog:     "./xdl/workflow/node_catalog.json",
		WorkerModule:        "dashboard.worker.train_worker",
		SystemIntervalText:  "2s",
		MetricsIntervalText: "2s",
		CompilerTimeoutText: "60s",
		EventBuffer:         256,
		MaxParallelRuns:     0,
	}
}

func Load(path string) (Config, error) {
	cfg := Default()
	if path == "" {
		return cfg, errors.New("config path cannot be empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read dashboard config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("decode dashboard config: %w", err)
	}
	cfg.applyDefaults()
	if err := cfg.resolve(filepath.Dir(path)); err != nil {
		return cfg, err
	}
	if err := cfg.Validate(); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func (c *Config) resolve(baseDir string) error {
	var err error
	c.ProjectRoot, err = resolvePath(baseDir, c.ProjectRoot)
	if err != nil {
		return fmt.Errorf("project_root: %w", err)
	}
	c.DatabasePath, err = resolvePath(c.ProjectRoot, c.DatabasePath)
	if err != nil {
		return fmt.Errorf("database_path: %w", err)
	}
	c.DataRoot, err = resolvePath(c.ProjectRoot, c.DataRoot)
	if err != nil {
		return fmt.Errorf("data_root: %w", err)
	}
	c.OutputRoot, err = resolvePath(c.ProjectRoot, c.OutputRoot)
	if err != nil {
		return fmt.Errorf("output_root: %w", err)
	}
	c.FrontendDir, err = resolvePath(c.ProjectRoot, c.FrontendDir)
	if err != nil {
		return fmt.Errorf("frontend_dir: %w", err)
	}
	c.TemplateCatalog, err = resolvePath(c.ProjectRoot, c.TemplateCatalog)
	if err != nil {
		return fmt.Errorf("template_catalog: %w", err)
	}
	c.WorkflowCatalog, err = resolvePath(c.ProjectRoot, c.WorkflowCatalog)
	if err != nil {
		return fmt.Errorf("workflow_catalog: %w", err)
	}
	if c.WorkerModule == "" {
		c.WorkerModule = "dashboard.worker.train_worker"
	}
	if c.EventBuffer < 32 {
		c.EventBuffer = 32
	}
	if c.MaxParallelRuns < 0 {
		return errors.New("max_parallel_runs cannot be negative")
	}
	c.SystemInterval, err = parseInterval(c.SystemIntervalText, 2*time.Second)
	if err != nil {
		return fmt.Errorf("system_interval: %w", err)
	}
	c.MetricsInterval, err = parseInterval(c.MetricsIntervalText, 2*time.Second)
	if err != nil {
		return fmt.Errorf("metrics_interval: %w", err)
	}
	c.CompilerTimeout, err = parseInterval(c.CompilerTimeoutText, 60*time.Second)
	if err != nil {
		return fmt.Errorf("compiler_timeout: %w", err)
	}
	return nil
}

func (c *Config) applyDefaults() {
	defaults := Default()
	if c.ListenAddr == "" {
		c.ListenAddr = defaults.ListenAddr
	}
	if c.DatabasePath == "" {
		c.DatabasePath = defaults.DatabasePath
	}
	if c.ProjectRoot == "" {
		c.ProjectRoot = defaults.ProjectRoot
	}
	if c.PythonExecutable == "" {
		c.PythonExecutable = defaults.PythonExecutable
	}
	if c.DataRoot == "" {
		c.DataRoot = defaults.DataRoot
	}
	if c.OutputRoot == "" {
		c.OutputRoot = defaults.OutputRoot
	}
	if c.FrontendDir == "" {
		c.FrontendDir = defaults.FrontendDir
	}
	if c.TemplateCatalog == "" {
		c.TemplateCatalog = defaults.TemplateCatalog
	}
	if c.WorkflowCatalog == "" {
		c.WorkflowCatalog = defaults.WorkflowCatalog
	}
	if c.SystemIntervalText == "" {
		c.SystemIntervalText = defaults.SystemIntervalText
	}
	if c.MetricsIntervalText == "" {
		c.MetricsIntervalText = defaults.MetricsIntervalText
	}
	if c.CompilerTimeoutText == "" {
		c.CompilerTimeoutText = defaults.CompilerTimeoutText
	}
	if c.EventBuffer == 0 {
		c.EventBuffer = defaults.EventBuffer
	}
}

func (c Config) Validate() error {
	if c.ListenAddr == "" {
		return errors.New("listen_addr cannot be empty")
	}
	if !isLoopbackAddress(c.ListenAddr) {
		return errors.New("non-loopback listen_addr requires authentication, which is not enabled")
	}
	if c.ProjectRoot == "" || c.PythonExecutable == "" {
		return errors.New("project_root and python_executable are required")
	}
	return nil
}

func isLoopbackAddress(address string) bool {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	return host == "localhost" || (ip != nil && ip.IsLoopback())
}

func resolvePath(baseDir, value string) (string, error) {
	if value == "" {
		return "", errors.New("path cannot be empty")
	}
	if !filepath.IsAbs(value) {
		value = filepath.Join(baseDir, value)
	}
	absolute, err := filepath.Abs(filepath.Clean(value))
	if err != nil {
		return "", err
	}
	return absolute, nil
}

func parseInterval(value string, fallback time.Duration) (time.Duration, error) {
	if value == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return 0, err
	}
	if parsed < 100*time.Millisecond {
		return 0, errors.New("interval must be at least 100ms")
	}
	return parsed, nil
}
