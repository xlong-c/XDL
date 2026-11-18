package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/store"
)

const eventPrefix = "XDL_EVENT\t"

type Runner struct {
	cfg         config.Config
	store       *store.Store
	hub         *events.Hub
	mu          sync.Mutex
	processes   map[string]*exec.Cmd
	exitHandler func(string)
}

type ControlCommand struct {
	Sequence       int64  `json:"seq"`
	Command        string `json:"command"`
	SaveCheckpoint bool   `json:"save_checkpoint"`
}

func New(cfg config.Config, database *store.Store, hub *events.Hub) *Runner {
	return &Runner{cfg: cfg, store: database, hub: hub, processes: make(map[string]*exec.Cmd)}
}

func (r *Runner) SetExitHandler(handler func(string)) {
	r.mu.Lock()
	r.exitHandler = handler
	r.mu.Unlock()
}

func (r *Runner) Start(ctx context.Context, run store.Run) error {
	if run.ID == "" {
		return errors.New("run id cannot be empty")
	}
	if err := os.MkdirAll(filepath.Dir(run.ManifestPath), 0o755); err != nil {
		return fmt.Errorf("create manifest directory: %w", err)
	}
	cmd, err := r.buildCommand(run)
	if err != nil {
		return err
	}
	configureProcess(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create worker stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("create worker stderr pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start worker: %w", err)
	}
	r.mu.Lock()
	r.processes[run.ID] = cmd
	r.mu.Unlock()
	if err := r.store.SetProcess(ctx, run.ID, cmd.Process.Pid); err != nil {
		_ = terminateProcess(cmd, true)
		return err
	}
	_ = r.store.UpdateRunStatus(ctx, run.ID, store.StatusRunning, "")
	outputReaders := sync.WaitGroup{}
	outputReaders.Add(2)
	go func() {
		defer outputReaders.Done()
		r.readOutput(run.ID, stdout, "stdout")
	}()
	go func() {
		defer outputReaders.Done()
		r.readOutput(run.ID, stderr, "stderr")
	}()
	go func() {
		outputReaders.Wait()
		r.wait(run.ID, cmd)
	}()
	return nil
}

func (r *Runner) Stop(ctx context.Context, runID string, force bool) error {
	r.mu.Lock()
	cmd := r.processes[runID]
	r.mu.Unlock()
	if cmd == nil {
		return fmt.Errorf("run %s is not active", runID)
	}
	if err := terminateProcess(cmd, force); err != nil {
		return fmt.Errorf("terminate run %s: %w", runID, err)
	}
	return r.store.UpdateRunStatus(ctx, runID, store.StatusKilled, "force terminated")
}

func (r *Runner) WriteControl(run store.Run, command ControlCommand) error {
	if command.Sequence <= 0 {
		return errors.New("control sequence must be positive")
	}
	if command.Command != "pause" && command.Command != "resume" && command.Command != "stop" && command.Command != "checkpoint_pause" {
		return fmt.Errorf("unsupported control command %q", command.Command)
	}
	payload, err := json.Marshal(command)
	if err != nil {
		return fmt.Errorf("encode control command: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(run.ControlPath), 0o755); err != nil {
		return fmt.Errorf("create control directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(run.ControlPath), ".control-*.tmp")
	if err != nil {
		return fmt.Errorf("create control temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(payload); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write control command: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close control command: %w", err)
	}
	if err := os.Rename(tmpName, run.ControlPath); err != nil {
		return fmt.Errorf("publish control command: %w", err)
	}
	return nil
}

func (r *Runner) ActiveProcessIDs() []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]int, 0, len(r.processes))
	for _, cmd := range r.processes {
		if cmd.Process != nil {
			result = append(result, cmd.Process.Pid)
		}
	}
	return result
}

func (r *Runner) buildCommand(run store.Run) (*exec.Cmd, error) {
	if r.cfg.PythonExecutable == "" {
		return nil, errors.New("python_executable is not configured")
	}
	worker := r.cfg.WorkerModule
	if worker == "" {
		worker = "dashboard.worker.train_worker"
	}
	args := []string{"-m"}
	switch run.Launcher {
	case "", "none":
		args = append(args, worker)
	case "torchrun":
		if len(run.Devices) < 1 {
			return nil, errors.New("torchrun requires at least one device")
		}
		args = append(args, "torch.distributed.run", "--standalone", "--nproc-per-node", strconv.Itoa(len(run.Devices)), "-m", worker)
	case "accelerate":
		if len(run.Devices) < 1 {
			return nil, errors.New("accelerate requires at least one device")
		}
		args = append(args, "accelerate.commands.launch", "--num_processes", strconv.Itoa(len(run.Devices)), "-m", worker)
	default:
		return nil, fmt.Errorf("unsupported launcher %q", run.Launcher)
	}
	pythonExecutable, err := resolvePythonExecutable(r.cfg.PythonExecutable)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(pythonExecutable, args...)
	cmd.Dir = r.cfg.ProjectRoot
	cmd.Env = append(os.Environ(),
		"XDL_DASHBOARD_MANIFEST="+run.ManifestPath,
		"PYTHONPATH="+prependPythonPath(r.cfg.ProjectRoot, os.Getenv("PYTHONPATH")),
	)
	if len(run.Devices) > 0 {
		values := make([]string, len(run.Devices))
		for i, device := range run.Devices {
			values[i] = strconv.Itoa(device)
		}
		cmd.Env = append(cmd.Env, "CUDA_VISIBLE_DEVICES="+strings.Join(values, ","))
	}
	return cmd, nil
}

func (r *Runner) readOutput(runID string, reader io.Reader, stream string) {
	scanner := bufio.NewScanner(reader)
	buffer := make([]byte, 64*1024)
	scanner.Buffer(buffer, 2*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, eventPrefix) {
			if err := r.handleStructuredEvent(runID, strings.TrimPrefix(line, eventPrefix)); err != nil {
				_ = r.publish(runID, "log", map[string]any{"stream": "runner", "message": err.Error()}, false)
			}
			continue
		}
		_ = r.publish(runID, "log", map[string]any{"stream": stream, "message": line}, false)
	}
	if err := scanner.Err(); err != nil {
		_ = r.publish(runID, "log", map[string]any{"stream": stream, "message": err.Error()}, true)
	}
}

func (r *Runner) handleStructuredEvent(runID, raw string) error {
	var value struct {
		Type      string         `json:"type"`
		Payload   map[string]any `json:"payload"`
		Reliable  bool           `json:"reliable"`
		Timestamp float64        `json:"ts"`
	}
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return fmt.Errorf("decode worker event: %w", err)
	}
	if value.Type == "" {
		return errors.New("worker event type is empty")
	}
	if err := r.publish(runID, value.Type, value.Payload, value.Reliable); err != nil {
		return err
	}
	if value.Type == "metrics" {
		step := intValue(value.Payload["global_step"])
		epoch := intValue(value.Payload["epoch"])
		metrics := floatMap(value.Payload["metrics"])
		_ = r.store.RecordMetric(context.Background(), runID, time.Now().UTC(), step, epoch, metrics)
		_ = r.store.UpdateProgress(context.Background(), runID, step, epoch)
	}
	if value.Type == "run_state" {
		if state, ok := value.Payload["state"].(string); ok {
			mapped := mapWorkerState(state)
			if mapped != "" {
				_ = r.store.UpdateRunStatus(context.Background(), runID, mapped, "")
			}
		}
	}
	if value.Type == "worker_exit" {
		// metrics 事件会被 EventSink 合并, 快速结束的 run 可能只上报了第一帧;
		// worker_exit 携带最终 step/epoch, 用它把运行记录对齐到真实进度.
		_ = r.store.UpdateProgress(
			context.Background(),
			runID,
			intValue(value.Payload["global_step"]),
			intValue(value.Payload["epoch"]),
		)
		controlState := stringValue(value.Payload["control_state"])
		stopMode := stringValue(value.Payload["stop_mode"])
		checkpointRequested := boolValue(value.Payload["checkpoint_requested"])
		if stopMode == "checkpoint_pause" || (stopMode == "" && controlState == "stopping" && checkpointRequested) {
			_ = r.store.UpdateRunStatus(context.Background(), runID, store.StatusCheckpointPaused, "")
		} else if stopMode == "stop" || controlState == "stopping" {
			_ = r.store.UpdateRunStatus(context.Background(), runID, store.StatusStopped, "")
		}
		if checkpoint := stringValue(value.Payload["checkpoint_path"]); checkpoint != "" {
			if run, err := r.store.GetRun(context.Background(), runID); err == nil {
				if absolute, err := filepath.Abs(checkpoint); err == nil {
					relative, relErr := filepath.Rel(run.OutputDir, absolute)
					if relErr == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
						_ = r.store.SetCheckpointPath(context.Background(), runID, absolute)
					}
				}
			}
		}
	}
	if value.Type == "workflow_compiled" {
		if version := stringValue(value.Payload["compiler_version"]); version != "" {
			_ = r.store.SetCompilerVersion(context.Background(), runID, version)
		}
	}
	if value.Type == "worker_error" || value.Type == "run_error" {
		_ = r.store.UpdateRunStatus(context.Background(), runID, store.StatusFailed, stringValue(value.Payload["message"]))
	}
	return nil
}

func (r *Runner) publish(runID, eventType string, payload map[string]any, reliable bool) error {
	id, err := r.store.RecordEvent(context.Background(), store.Event{RunID: runID, Type: eventType, CreatedAt: time.Now().UTC(), Payload: payload, Reliable: reliable})
	if err != nil {
		return err
	}
	r.hub.Publish(events.Event{ID: id, RunID: runID, Type: eventType, CreatedAt: time.Now().UTC(), Payload: payload, Reliable: reliable})
	return nil
}

func (r *Runner) wait(runID string, cmd *exec.Cmd) {
	err := cmd.Wait()
	r.mu.Lock()
	delete(r.processes, runID)
	r.mu.Unlock()
	status := store.StatusSucceeded
	message := ""
	if err != nil {
		status = store.StatusFailed
		message = err.Error()
	}
	if run, getErr := r.store.GetRun(context.Background(), runID); getErr == nil {
		switch run.Status {
		case store.StatusKilled:
			status = store.StatusKilled
			message = run.ErrorMessage
		case store.StatusStopped:
			status = store.StatusStopped
		case store.StatusCheckpointPaused:
			status = store.StatusCheckpointPaused
		}
	}
	_ = r.store.UpdateRunStatus(context.Background(), runID, status, message)
	_ = r.publish(runID, "worker_exit", map[string]any{"status": status, "error": message}, true)
	r.mu.Lock()
	exitHandler := r.exitHandler
	r.mu.Unlock()
	if exitHandler != nil {
		exitHandler(runID)
	}
}

func resolvePythonExecutable(configured string) (string, error) {
	if configured == "" {
		return "", errors.New("python_executable is not configured")
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
	return "", fmt.Errorf("python executable %q was not found; set python_executable to the XDL environment interpreter", configured)
}

func prependPythonPath(projectRoot, existing string) string {
	if existing == "" {
		return projectRoot
	}
	return projectRoot + string(os.PathListSeparator) + existing
}

func mapWorkerState(value string) string {
	switch value {
	case "starting":
		return store.StatusStarting
	case "validating":
		return store.StatusValidating
	case "running":
		return store.StatusRunning
	case "pause_requested":
		return store.StatusPauseRequested
	case "paused":
		return store.StatusPaused
	case "stopping":
		return store.StatusStopping
	case "succeeded":
		return store.StatusSucceeded
	default:
		return ""
	}
}

func intValue(value any) int {
	switch typed := value.(type) {
	case float64:
		return int(typed)
	case int:
		return typed
	default:
		return 0
	}
}

func floatMap(value any) map[string]float64 {
	result := make(map[string]float64)
	values, ok := value.(map[string]any)
	if !ok {
		return result
	}
	for key, raw := range values {
		if number, ok := raw.(float64); ok {
			result[key] = number
		}
	}
	return result
}

func boolValue(value any) bool {
	if typed, ok := value.(bool); ok {
		return typed
	}
	return false
}

func stringValue(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	return ""
}
