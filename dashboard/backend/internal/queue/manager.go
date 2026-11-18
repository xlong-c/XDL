package queue

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/runner"
	"github.com/xlong-c/xdl-dashboard/internal/store"
	"github.com/xlong-c/xdl-dashboard/internal/system"
	"github.com/xlong-c/xdl-dashboard/internal/templates"
	"github.com/xlong-c/xdl-dashboard/internal/workflow"
)

type SubmitRequest struct {
	TemplateID       string         `json:"template_id"`
	Name             string         `json:"name"`
	Parameters       map[string]any `json:"parameters"`
	Launcher         string         `json:"launcher"`
	Devices          []int          `json:"devices"`
	ParentRunID      string         `json:"parent_run_id"`
	ResumeCheckpoint string         `json:"resume_checkpoint"`
}

// WorkflowSubmitRequest 描述以 workflow 快照创建 run 的请求.
type WorkflowSubmitRequest struct {
	Name             string `json:"name"`
	Launcher         string `json:"launcher"`
	Devices          []int  `json:"devices"`
	ParentRunID      string `json:"parent_run_id"`
	ResumeCheckpoint string `json:"resume_checkpoint"`
}

// WorkflowManifestKind 标记 manifest 描述的是 workflow run.
const WorkflowManifestKind = "workflow"

// WorkflowSnapshotFilename 是固化到 run 目录的快照文件名.
const WorkflowSnapshotFilename = "workflow.json"

type Manager struct {
	cfg       config.Config
	database  *store.Store
	catalog   *templates.Catalog
	workflows *workflow.Catalog
	runner    *runner.Runner
	hub       *events.Hub
	gpu       system.GPUProvider

	mu          sync.Mutex
	pending     []store.Run
	active      map[string]store.Run
	deviceOwner map[int]string
	cpuActive   bool
	commands    map[string]int64
}

func New(cfg config.Config, database *store.Store, catalog *templates.Catalog, workflows *workflow.Catalog, jobRunner *runner.Runner, hub *events.Hub, gpu system.GPUProvider) *Manager {
	manager := &Manager{
		cfg: cfg, database: database, catalog: catalog, workflows: workflows,
		runner: jobRunner, hub: hub, gpu: gpu,
		active: make(map[string]store.Run), deviceOwner: make(map[int]string),
		commands: make(map[string]int64),
	}
	jobRunner.SetExitHandler(manager.onRunnerExit)
	return manager
}

func (m *Manager) Recover(ctx context.Context) error {
	runs, err := m.database.ListRuns(ctx, 1000)
	if err != nil {
		return err
	}
	for _, run := range runs {
		switch run.Status {
		case store.StatusStarting, store.StatusRunning, store.StatusPaused, store.StatusPauseRequested, store.StatusStopping,
			store.StatusCheckpointPausing:
			if err := m.database.UpdateRunStatus(ctx, run.ID, store.StatusInterrupted, "dashboard service restarted"); err != nil {
				return err
			}
		case store.StatusQueued:
			m.pending = append(m.pending, run)
		}
	}
	sort.Slice(m.pending, func(i, j int) bool {
		return m.pending[i].SubmittedAt.Before(m.pending[j].SubmittedAt)
	})
	m.dispatch()
	return nil
}

func (m *Manager) Submit(ctx context.Context, request SubmitRequest) (store.Run, error) {
	_, catalogItem, ok := m.catalog.Get(request.TemplateID)
	if !ok {
		return store.Run{}, fmt.Errorf("unknown template %q", request.TemplateID)
	}
	parent, resumeCheckpoint, err := m.resolveParent(ctx, request.ParentRunID, request.ResumeCheckpoint)
	if err != nil {
		return store.Run{}, err
	}
	if request.ParentRunID != "" && parent.TemplateID != request.TemplateID {
		return store.Run{}, errors.New("checkpoint recovery must use the parent run template")
	}
	if request.ParentRunID != "" && request.Parameters == nil && parent.ConfigJSON != nil {
		if saved, ok := parent.ConfigJSON["submitted_parameters"].(map[string]any); ok {
			request.Parameters = saved
		}
	}
	parameters, err := m.catalog.ValidateParameters(request.TemplateID, request.Parameters)
	if err != nil {
		return store.Run{}, err
	}
	launcher := request.Launcher
	if launcher == "" {
		launcher = "none"
	}
	if launcher != "none" && launcher != "torchrun" && launcher != "accelerate" {
		return store.Run{}, fmt.Errorf("unsupported launcher %q", launcher)
	}
	devices, err := normalizeDevices(request.Devices)
	if err != nil {
		return store.Run{}, err
	}
	if catalogItem.RequiresGPU && len(devices) == 0 {
		return store.Run{}, errors.New("this template requires at least one NVIDIA GPU")
	}
	if len(devices) > 1 && launcher == "none" {
		return store.Run{}, errors.New("multi-GPU runs require torchrun or accelerate")
	}
	if len(devices) == 0 && launcher != "none" {
		return store.Run{}, errors.New("distributed launcher requires selected GPU devices")
	}
	if err := m.validateDeviceIDs(ctx, devices); err != nil {
		return store.Run{}, err
	}

	id, err := newID()
	if err != nil {
		return store.Run{}, err
	}
	runDir := filepath.Join(m.cfg.OutputRoot, id)
	for _, dir := range []string{
		runDir,
		filepath.Join(runDir, "logs"),
		filepath.Join(runDir, "checkpoints"),
		filepath.Join(runDir, "previews"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return store.Run{}, fmt.Errorf("create run directory: %w", err)
		}
	}
	sourcePath, err := m.catalog.SourcePath(request.TemplateID)
	if err != nil {
		return store.Run{}, err
	}
	configPath := filepath.Join(runDir, "config.yaml")
	manifestPath := filepath.Join(runDir, "manifest.json")
	controlPath := filepath.Join(runDir, "control.json")
	name := request.Name
	if name == "" {
		name = request.TemplateID + "-" + id[:8]
	}
	deviceName := "cpu"
	if len(devices) > 0 {
		deviceName = "cuda"
	}
	systemOverrides := map[string]any{
		"runtime.experiment_name":    name,
		"runtime.output_dir":         runDir,
		"runtime.device":             deviceName,
		"logging.log_dir":            filepath.Join(runDir, "logs"),
		"logging.enable_console":     false,
		"logging.enable_tensorboard": false,
		"logging.enable_wandb":       false,
		"checkpoint.dirpath":         filepath.Join(runDir, "checkpoints"),
		"checkpoint.save_last":       true,
	}
	if download, ok := parameters["train_dataset.params.download"]; ok {
		systemOverrides["val_dataset.params.download"] = download
	}
	manifest := map[string]any{
		"config_path":           configPath,
		"template_path":         sourcePath,
		"control_path":          controlPath,
		"overrides":             parameters,
		"allowed_paths":         catalogItem.AllowedOverride,
		"system_overrides":      systemOverrides,
		"use_accelerate":        launcher == "accelerate",
		"resume_checkpoint":     resumeCheckpoint,
		"metrics_interval_s":    m.cfg.MetricsInterval.Seconds(),
		"metrics_every_n_steps": 1,
	}
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return store.Run{}, fmt.Errorf("encode run manifest: %w", err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0o600); err != nil {
		return store.Run{}, fmt.Errorf("write run manifest: %w", err)
	}
	now := time.Now().UTC()
	configJSON := map[string]any{
		"submitted_parameters": request.Parameters,
		"resolved_parameters":  parameters,
		"launcher":             launcher,
		"devices":              devices,
		"name":                 name,
	}
	run := store.Run{
		ID: id, ParentRunID: request.ParentRunID, TemplateID: request.TemplateID,
		Status: store.StatusQueued, SubmittedAt: now, UpdatedAt: now,
		ConfigJSON: configJSON, ConfigPath: configPath, ManifestPath: manifestPath,
		ControlPath: controlPath, OutputDir: runDir, CheckpointPath: resumeCheckpoint,
		Launcher: launcher, Devices: devices,
	}
	if err := m.database.InsertRun(ctx, run); err != nil {
		return store.Run{}, err
	}
	m.mu.Lock()
	m.pending = append(m.pending, run)
	m.mu.Unlock()
	_, _ = m.database.RecordEvent(ctx, store.Event{
		RunID: id, Type: "run_state", CreatedAt: now,
		Payload:  map[string]any{"state": store.StatusQueued, "template_id": request.TemplateID},
		Reliable: true,
	})
	m.hub.Publish(events.Event{
		RunID: id, Type: "run_state", CreatedAt: now,
		Payload:  map[string]any{"state": store.StatusQueued, "template_id": request.TemplateID},
		Reliable: true,
	})
	m.dispatch()
	return run, nil
}

// SubmitWorkflow 以不可变 workflow 快照创建 run.
//
// 语义预检在创建快照时已完成; 这里重新做结构校验与规范化, 并把规范化快照固化到
// run 目录, 保证 run 使用的图不随后续 workflow 记录变化而变化.
func (m *Manager) SubmitWorkflow(ctx context.Context, workflowID string, request WorkflowSubmitRequest) (store.Run, error) {
	item, err := m.database.GetWorkflow(ctx, workflowID)
	if err != nil {
		return store.Run{}, fmt.Errorf("load workflow %q: %w", workflowID, err)
	}
	snapshot, issues := workflow.Prepare([]byte(item.SnapshotJSON), m.workflows)
	if workflow.HasErrors(issues) {
		return store.Run{}, fmt.Errorf("workflow snapshot is invalid: %s", workflow.ErrorMessages(issues))
	}
	launcher := request.Launcher
	if launcher == "" {
		launcher = snapshot.Settings.Launcher
	}
	if launcher == "" {
		launcher = "none"
	}
	if launcher != "none" && launcher != "torchrun" && launcher != "accelerate" {
		return store.Run{}, fmt.Errorf("unsupported launcher %q", launcher)
	}
	requestedDevices := request.Devices
	if len(requestedDevices) == 0 {
		requestedDevices = snapshot.Settings.Devices
	}
	devices, err := normalizeDevices(requestedDevices)
	if err != nil {
		return store.Run{}, err
	}
	if len(devices) > 1 && launcher == "none" {
		return store.Run{}, errors.New("multi-GPU runs require torchrun or accelerate")
	}
	if len(devices) == 0 && launcher != "none" {
		return store.Run{}, errors.New("distributed launcher requires selected GPU devices")
	}
	if err := m.validateDeviceIDs(ctx, devices); err != nil {
		return store.Run{}, err
	}
	parent, resumeCheckpoint, err := m.resolveParent(ctx, request.ParentRunID, request.ResumeCheckpoint)
	if err != nil {
		return store.Run{}, err
	}
	if request.ParentRunID != "" && parent.WorkflowID != workflowID {
		return store.Run{}, errors.New("checkpoint recovery must use the parent run workflow")
	}

	id, err := newID()
	if err != nil {
		return store.Run{}, err
	}
	runDir := filepath.Join(m.cfg.OutputRoot, id)
	for _, dir := range []string{
		runDir,
		filepath.Join(runDir, "logs"),
		filepath.Join(runDir, "checkpoints"),
		filepath.Join(runDir, "previews"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return store.Run{}, fmt.Errorf("create run directory: %w", err)
		}
	}
	snapshotPath := filepath.Join(runDir, WorkflowSnapshotFilename)
	snapshotBytes, err := snapshot.CanonicalJSON()
	if err != nil {
		return store.Run{}, fmt.Errorf("encode workflow snapshot: %w", err)
	}
	if err := os.WriteFile(snapshotPath, snapshotBytes, 0o600); err != nil {
		return store.Run{}, fmt.Errorf("write workflow snapshot: %w", err)
	}
	snapshotHash := workflow.HashBytes(snapshotBytes)
	configPath := filepath.Join(runDir, "config.yaml")
	manifestPath := filepath.Join(runDir, "manifest.json")
	controlPath := filepath.Join(runDir, "control.json")
	name := request.Name
	if name == "" {
		name = snapshot.Name + "-" + id[:8]
	}
	deviceName := "cpu"
	if len(devices) > 0 {
		deviceName = "cuda"
	}
	systemOverrides := map[string]any{
		"runtime.experiment_name":    name,
		"runtime.output_dir":         runDir,
		"runtime.device":             deviceName,
		"runtime.data_dir":           m.cfg.DataRoot,
		"logging.log_dir":            filepath.Join(runDir, "logs"),
		"logging.enable_console":     false,
		"logging.enable_tensorboard": false,
		"logging.enable_wandb":       false,
		"checkpoint.dirpath":         filepath.Join(runDir, "checkpoints"),
		"checkpoint.save_last":       true,
	}
	manifest := map[string]any{
		"kind":                   WorkflowManifestKind,
		"workflow_path":          snapshotPath,
		"workflow_snapshot_hash": snapshotHash,
		"execution_profile":      snapshot.ExecutionProfile,
		"config_path":            configPath,
		"control_path":           controlPath,
		"project_root":           m.cfg.ProjectRoot,
		"data_root":              m.cfg.DataRoot,
		"system_overrides":       systemOverrides,
		"resume_checkpoint":      resumeCheckpoint,
		"metrics_interval_s":     m.cfg.MetricsInterval.Seconds(),
		"metrics_every_n_steps":  1,
	}
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return store.Run{}, fmt.Errorf("encode run manifest: %w", err)
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0o600); err != nil {
		return store.Run{}, fmt.Errorf("write run manifest: %w", err)
	}
	now := time.Now().UTC()
	configJSON := map[string]any{
		"workflow_id":            workflowID,
		"workflow_name":          snapshot.Name,
		"workflow_snapshot_hash": snapshotHash,
		"execution_profile":      snapshot.ExecutionProfile,
		"launcher":               launcher,
		"devices":                devices,
		"name":                   name,
	}
	run := store.Run{
		ID: id, ParentRunID: request.ParentRunID, TemplateID: "",
		Status: store.StatusQueued, SubmittedAt: now, UpdatedAt: now,
		ConfigJSON: configJSON, ConfigPath: configPath, ManifestPath: manifestPath,
		ControlPath: controlPath, OutputDir: runDir, CheckpointPath: resumeCheckpoint,
		Launcher: launcher, Devices: devices,
		WorkflowID: workflowID, WorkflowSnapshotHash: snapshotHash,
	}
	if err := m.database.InsertRun(ctx, run); err != nil {
		return store.Run{}, err
	}
	m.mu.Lock()
	m.pending = append(m.pending, run)
	m.mu.Unlock()
	payload := map[string]any{
		"state":             store.StatusQueued,
		"workflow_id":       workflowID,
		"execution_profile": snapshot.ExecutionProfile,
	}
	_, _ = m.database.RecordEvent(ctx, store.Event{
		RunID: id, Type: "run_state", CreatedAt: now, Payload: payload, Reliable: true,
	})
	m.hub.Publish(events.Event{
		RunID: id, Type: "run_state", CreatedAt: now, Payload: payload, Reliable: true,
	})
	m.dispatch()
	return run, nil
}

func (m *Manager) List(ctx context.Context, limit int) ([]store.Run, error) {
	return m.database.ListRuns(ctx, limit)
}

func (m *Manager) Get(ctx context.Context, id string) (store.Run, error) {
	return m.database.GetRun(ctx, id)
}

func (m *Manager) Control(ctx context.Context, id, action string) error {
	run, err := m.database.GetRun(ctx, id)
	if err != nil {
		return err
	}
	if action == "kill" {
		return m.runner.Stop(ctx, id, true)
	}
	if action != "pause" && action != "resume" && action != "stop" && action != "checkpoint-pause" {
		return fmt.Errorf("unsupported action %q", action)
	}
	if run.Status != store.StatusRunning && run.Status != store.StatusPaused {
		return fmt.Errorf("run %s cannot accept %s while %s", id, action, run.Status)
	}
	m.mu.Lock()
	sequence := m.commands[id] + 1
	m.commands[id] = sequence
	m.mu.Unlock()
	commandName := strings.ReplaceAll(action, "-", "_")
	command := runner.ControlCommand{
		Sequence:       sequence,
		Command:        commandName,
		SaveCheckpoint: action == "stop" || action == "checkpoint-pause",
	}
	if err := m.runner.WriteControl(run, command); err != nil {
		return err
	}
	switch action {
	case "pause":
		return m.database.UpdateRunStatus(ctx, id, "pause_requested", "")
	case "resume":
		return m.database.UpdateRunStatus(ctx, id, store.StatusRunning, "")
	case "stop":
		return m.database.UpdateRunStatus(ctx, id, store.StatusStopping, "")
	case "checkpoint-pause":
		return m.database.UpdateRunStatus(ctx, id, store.StatusCheckpointPausing, "")
	}
	return nil
}

func (m *Manager) dispatch() {
	m.mu.Lock()
	var starts []store.Run
	for i := 0; i < len(m.pending); {
		run := m.pending[i]
		if !m.canStartLocked(run) {
			i++
			continue
		}
		m.pending = append(m.pending[:i], m.pending[i+1:]...)
		m.active[run.ID] = run
		for _, device := range run.Devices {
			m.deviceOwner[device] = run.ID
		}
		if len(run.Devices) == 0 {
			m.cpuActive = true
		}
		starts = append(starts, run)
	}
	m.mu.Unlock()
	for _, run := range starts {
		go m.launch(run)
	}
}

func (m *Manager) canStartLocked(run store.Run) bool {
	if m.cfg.MaxParallelRuns > 0 && len(m.active) >= m.cfg.MaxParallelRuns {
		return false
	}
	if len(run.Devices) == 0 && m.cpuActive {
		return false
	}
	for _, device := range run.Devices {
		if _, busy := m.deviceOwner[device]; busy {
			return false
		}
	}
	return true
}

func (m *Manager) launch(run store.Run) {
	ctx := context.Background()
	_ = m.database.UpdateRunStatus(ctx, run.ID, store.StatusStarting, "")
	if err := m.runner.Start(ctx, run); err != nil {
		_ = m.database.UpdateRunStatus(ctx, run.ID, store.StatusFailed, err.Error())
		m.onRunnerExit(run.ID)
		return
	}
}

func (m *Manager) onRunnerExit(runID string) {
	m.mu.Lock()
	run, ok := m.active[runID]
	if ok {
		delete(m.active, runID)
		for _, device := range run.Devices {
			if m.deviceOwner[device] == runID {
				delete(m.deviceOwner, device)
			}
		}
		if len(run.Devices) == 0 {
			m.cpuActive = false
		}
	}
	m.mu.Unlock()
	m.dispatch()
}

func (m *Manager) resolveParent(ctx context.Context, parentID, checkpoint string) (store.Run, string, error) {
	if parentID == "" {
		if checkpoint != "" {
			return store.Run{}, "", errors.New("resume_checkpoint requires parent_run_id")
		}
		return store.Run{}, "", nil
	}
	parent, err := m.database.GetRun(ctx, parentID)
	if err != nil {
		return store.Run{}, "", fmt.Errorf("load parent run: %w", err)
	}
	if checkpoint == "" {
		checkpoint = parent.CheckpointPath
	}
	if checkpoint == "" {
		return store.Run{}, "", errors.New("parent run has no checkpoint")
	}
	if !filepath.IsAbs(checkpoint) {
		checkpoint = filepath.Join(parent.OutputDir, checkpoint)
	}
	checkpoint, err = filepath.Abs(checkpoint)
	if err != nil {
		return store.Run{}, "", err
	}
	rel, err := filepath.Rel(parent.OutputDir, checkpoint)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return store.Run{}, "", errors.New("checkpoint path escapes parent run directory")
	}
	info, err := os.Stat(checkpoint)
	if err != nil || !info.IsDir() {
		return store.Run{}, "", errors.New("checkpoint must be an existing directory under the parent run")
	}
	return parent, checkpoint, nil
}

func (m *Manager) validateDeviceIDs(ctx context.Context, devices []int) error {
	if len(devices) == 0 {
		return nil
	}
	if m.gpu == nil {
		return errors.New("NVIDIA GPU inventory is unavailable")
	}
	available, err := m.gpu.Collect(ctx)
	if err != nil {
		return fmt.Errorf("query NVIDIA GPU inventory: %w", err)
	}
	indices := make(map[int]bool, len(available))
	for _, device := range available {
		index, ok := numericIndex(device["index"])
		if ok {
			indices[index] = true
		}
	}
	for _, id := range devices {
		if !indices[id] {
			return fmt.Errorf("GPU device %d is not available", id)
		}
	}
	return nil
}

func numericIndex(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int32:
		return int(typed), true
	case float64:
		return int(typed), true
	default:
		return 0, false
	}
}

func normalizeDevices(input []int) ([]int, error) {
	seen := make(map[int]bool, len(input))
	result := append([]int(nil), input...)
	sort.Ints(result)
	for _, id := range result {
		if id < 0 {
			return nil, errors.New("GPU device ids must be non-negative")
		}
		if seen[id] {
			return nil, fmt.Errorf("GPU device %d was selected more than once", id)
		}
		seen[id] = true
	}
	return result, nil
}

func newID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("generate run id: %w", err)
	}
	return hex.EncodeToString(value[:]), nil
}
