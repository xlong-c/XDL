package store

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed schema.sql
var schemaFS embed.FS

const (
	StatusQueued            = "queued"
	StatusValidating        = "validating"
	StatusStarting          = "starting"
	StatusRunning           = "running"
	StatusPaused            = "paused"
	StatusPauseRequested    = "pause_requested"
	StatusStopping          = "stopping"
	StatusCheckpointPausing = "checkpoint_pausing"
	StatusCheckpointPaused  = "checkpoint_paused"
	StatusSucceeded         = "succeeded"
	StatusFailed            = "failed"
	StatusStopped           = "stopped"
	StatusKilled            = "killed"
	StatusInterrupted       = "interrupted"
)

type Run struct {
	ID             string
	ParentRunID    string
	TemplateID     string
	Status         string
	SubmittedAt    time.Time
	UpdatedAt      time.Time
	StartedAt      *time.Time
	FinishedAt     *time.Time
	ConfigJSON     map[string]any
	ConfigPath     string
	ManifestPath   string
	ControlPath    string
	OutputDir      string
	CheckpointPath string
	Launcher       string
	Devices        []int
	ProcessID      int
	LastStep       int
	LastEpoch      int
	ErrorMessage   string

	// workflow 关联字段; 模板 run 保持为空.
	WorkflowID           string
	WorkflowSnapshotHash string
	CompilerVersion      string
}

// Workflow 是一条不可变的 workflow 快照记录.
type Workflow struct {
	ID               string
	Version          int
	Name             string
	ExecutionProfile string
	SnapshotJSON     string
	SnapshotHash     string
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

type Event struct {
	ID        int64
	RunID     string
	Type      string
	CreatedAt time.Time
	Payload   map[string]any
	Reliable  bool
}

type MetricSample struct {
	ID         int64
	CreatedAt  time.Time
	GlobalStep int
	Epoch      int
	Metrics    map[string]float64
}

type Artifact struct {
	ID           int64
	RunID        string
	Kind         string
	RelativePath string
	CreatedAt    time.Time
	Metadata     map[string]any
}

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	if path == "" {
		return nil, errors.New("database path cannot be empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create database directory: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite database: %w", err)
	}
	db.SetMaxOpenConns(1)
	store := &Store{db: db}
	if err := store.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) migrate(ctx context.Context) error {
	schema, err := schemaFS.ReadFile("schema.sql")
	if err != nil {
		return fmt.Errorf("read sqlite schema: %w", err)
	}
	if _, err := s.db.ExecContext(ctx, "PRAGMA journal_mode = WAL"); err != nil {
		return fmt.Errorf("enable sqlite WAL: %w", err)
	}
	if _, err := s.db.ExecContext(ctx, "PRAGMA busy_timeout = 5000"); err != nil {
		return fmt.Errorf("configure sqlite busy timeout: %w", err)
	}
	if _, err := s.db.ExecContext(ctx, "PRAGMA foreign_keys = ON"); err != nil {
		return fmt.Errorf("enable sqlite foreign keys: %w", err)
	}
	if _, err := s.db.ExecContext(ctx, string(schema)); err != nil {
		return fmt.Errorf("migrate sqlite schema: %w", err)
	}
	if err := s.ensureColumns(ctx, "runs", map[string]string{
		"workflow_id":            "TEXT",
		"workflow_snapshot_hash": "TEXT",
		"compiler_version":       "TEXT",
	}); err != nil {
		return err
	}
	// 索引必须在列迁移之后创建, 否则旧数据库会因为缺少 workflow_id 列而失败.
	if _, err := s.db.ExecContext(ctx,
		"CREATE INDEX IF NOT EXISTS idx_runs_workflow ON runs(workflow_id)"); err != nil {
		return fmt.Errorf("create workflow run index: %w", err)
	}
	return nil
}

// ensureColumns 为已存在的表补齐新增列, 保证旧数据库可以原地升级.
func (s *Store) ensureColumns(ctx context.Context, table string, columns map[string]string) error {
	rows, err := s.db.QueryContext(ctx, "PRAGMA table_info("+table+")")
	if err != nil {
		return fmt.Errorf("inspect %s columns: %w", table, err)
	}
	existing := make(map[string]bool)
	for rows.Next() {
		var cid int
		var name, columnType string
		var notNull, pk int
		var defaultValue sql.NullString
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &pk); err != nil {
			_ = rows.Close()
			return fmt.Errorf("scan %s columns: %w", table, err)
		}
		existing[name] = true
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return fmt.Errorf("iterate %s columns: %w", table, err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close %s columns: %w", table, err)
	}
	for name, columnType := range columns {
		if existing[name] {
			continue
		}
		statement := fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, name, columnType)
		if _, err := s.db.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("add %s.%s column: %w", table, name, err)
		}
	}
	return nil
}

func (s *Store) InsertRun(ctx context.Context, run Run) error {
	configJSON, err := json.Marshal(run.ConfigJSON)
	if err != nil {
		return fmt.Errorf("marshal run config: %w", err)
	}
	devicesJSON, err := json.Marshal(run.Devices)
	if err != nil {
		return fmt.Errorf("marshal devices: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `
		INSERT INTO runs (
			id, parent_run_id, template_id, status, submitted_at, updated_at,
			config_json, config_path, manifest_path, control_path, output_dir,
			checkpoint_path, launcher, devices_json, last_step, last_epoch,
			workflow_id, workflow_snapshot_hash
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, run.ID, nullableString(run.ParentRunID), run.TemplateID, run.Status,
		run.SubmittedAt.UTC().Format(time.RFC3339Nano), run.UpdatedAt.UTC().Format(time.RFC3339Nano),
		string(configJSON), run.ConfigPath, run.ManifestPath, run.ControlPath, run.OutputDir,
		nullableString(run.CheckpointPath), run.Launcher, string(devicesJSON), run.LastStep, run.LastEpoch,
		nullableString(run.WorkflowID), nullableString(run.WorkflowSnapshotHash))
	if err != nil {
		return fmt.Errorf("insert run: %w", err)
	}
	return nil
}

func (s *Store) GetRun(ctx context.Context, id string) (Run, error) {
	row := s.db.QueryRowContext(ctx, `SELECT
		id, COALESCE(parent_run_id, ''), template_id, status, submitted_at,
		updated_at, started_at, finished_at, config_json, config_path,
		manifest_path, control_path, output_dir, COALESCE(checkpoint_path, ''),
		launcher, devices_json, COALESCE(process_id, 0), last_step, last_epoch,
		COALESCE(error_message, ''), COALESCE(workflow_id, ''),
		COALESCE(workflow_snapshot_hash, ''), COALESCE(compiler_version, '')
		FROM runs WHERE id = ?`, id)
	return scanRun(row)
}

func (s *Store) ListRuns(ctx context.Context, limit int) ([]Run, error) {
	if limit <= 0 || limit > 1000 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx, `SELECT
		id, COALESCE(parent_run_id, ''), template_id, status, submitted_at,
		updated_at, started_at, finished_at, config_json, config_path,
		manifest_path, control_path, output_dir, COALESCE(checkpoint_path, ''),
		launcher, devices_json, COALESCE(process_id, 0), last_step, last_epoch,
		COALESCE(error_message, ''), COALESCE(workflow_id, ''),
		COALESCE(workflow_snapshot_hash, ''), COALESCE(compiler_version, '')
		FROM runs ORDER BY submitted_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("list runs: %w", err)
	}
	defer rows.Close()
	var result []Run
	for rows.Next() {
		run, scanErr := scanRun(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate runs: %w", err)
	}
	return result, nil
}

// InsertWorkflow 保存一份不可变 workflow 快照.
func (s *Store) InsertWorkflow(ctx context.Context, item Workflow) error {
	if item.ID == "" {
		return errors.New("workflow id cannot be empty")
	}
	if item.SnapshotJSON == "" {
		return errors.New("workflow snapshot cannot be empty")
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO workflows (
			id, version, name, execution_profile, snapshot_json, snapshot_hash,
			created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, item.ID, item.Version, item.Name, item.ExecutionProfile, item.SnapshotJSON,
		item.SnapshotHash, item.CreatedAt.UTC().Format(time.RFC3339Nano),
		item.UpdatedAt.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return fmt.Errorf("insert workflow: %w", err)
	}
	return nil
}

// GetWorkflow 读取一份 workflow 快照.
func (s *Store) GetWorkflow(ctx context.Context, id string) (Workflow, error) {
	row := s.db.QueryRowContext(ctx, `SELECT
		id, version, name, execution_profile, snapshot_json, snapshot_hash,
		created_at, updated_at
		FROM workflows WHERE id = ?`, id)
	return scanWorkflow(row)
}

// ListWorkflows 列出 workflow 元数据, 不返回 snapshot_json 以控制响应体积.
func (s *Store) ListWorkflows(ctx context.Context, limit int) ([]Workflow, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.db.QueryContext(ctx, `SELECT
		id, version, name, execution_profile, '', snapshot_hash, created_at, updated_at
		FROM workflows ORDER BY created_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("list workflows: %w", err)
	}
	defer rows.Close()
	result := make([]Workflow, 0)
	for rows.Next() {
		item, scanErr := scanWorkflow(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate workflows: %w", err)
	}
	return result, nil
}

// SetCompilerVersion 记录 worker 实际使用的 compiler 版本.
func (s *Store) SetCompilerVersion(ctx context.Context, id, version string) error {
	if strings.TrimSpace(version) == "" {
		return nil
	}
	_, err := s.db.ExecContext(ctx, `UPDATE runs SET compiler_version = ?, updated_at = ? WHERE id = ?`,
		version, time.Now().UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return fmt.Errorf("set compiler version: %w", err)
	}
	return nil
}

func scanWorkflow(scanner interface{ Scan(...any) error }) (Workflow, error) {
	var item Workflow
	var created, updated string
	if err := scanner.Scan(
		&item.ID, &item.Version, &item.Name, &item.ExecutionProfile,
		&item.SnapshotJSON, &item.SnapshotHash, &created, &updated,
	); err != nil {
		return Workflow{}, err
	}
	var err error
	if item.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
		return Workflow{}, err
	}
	if item.UpdatedAt, err = time.Parse(time.RFC3339Nano, updated); err != nil {
		return Workflow{}, err
	}
	return item, nil
}

func (s *Store) UpdateRunStatus(ctx context.Context, id, status, message string) error {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, err := s.db.ExecContext(ctx, `UPDATE runs SET status = ?, updated_at = ?,
		error_message = CASE WHEN ? = '' THEN error_message ELSE ? END,
		finished_at = CASE WHEN ? IN (?, ?, ?, ?, ?) THEN COALESCE(finished_at, ?) ELSE finished_at END,
		started_at = CASE WHEN ? = ? THEN COALESCE(started_at, ?) ELSE started_at END
		WHERE id = ?`, status, now, message, message, status,
		StatusSucceeded, StatusFailed, StatusStopped, StatusKilled, StatusInterrupted, now,
		status, StatusRunning, now, id)
	if err != nil {
		return fmt.Errorf("update run status: %w", err)
	}
	return nil
}

func (s *Store) SetCheckpointPath(ctx context.Context, id, checkpointPath string) error {
	_, err := s.db.ExecContext(ctx, `UPDATE runs SET checkpoint_path = ?, updated_at = ? WHERE id = ?`, nullableString(checkpointPath), time.Now().UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return fmt.Errorf("set checkpoint path: %w", err)
	}
	return nil
}

func (s *Store) SetProcess(ctx context.Context, id string, pid int) error {
	_, err := s.db.ExecContext(ctx, `UPDATE runs SET process_id = ?, updated_at = ? WHERE id = ?`, pid, time.Now().UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return fmt.Errorf("set run process: %w", err)
	}
	return nil
}

func (s *Store) UpdateProgress(ctx context.Context, id string, step, epoch int) error {
	_, err := s.db.ExecContext(ctx, `UPDATE runs SET last_step = ?, last_epoch = ?, updated_at = ? WHERE id = ?`, step, epoch, time.Now().UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return fmt.Errorf("update run progress: %w", err)
	}
	return nil
}

func (s *Store) RecordEvent(ctx context.Context, event Event) (int64, error) {
	payload, err := json.Marshal(event.Payload)
	if err != nil {
		return 0, fmt.Errorf("marshal event payload: %w", err)
	}
	result, err := s.db.ExecContext(ctx, `INSERT INTO events(run_id, event_type, created_at, payload_json, reliable) VALUES (?, ?, ?, ?, ?)`, nullableString(event.RunID), event.Type, event.CreatedAt.UTC().Format(time.RFC3339Nano), string(payload), boolInt(event.Reliable))
	if err != nil {
		return 0, fmt.Errorf("insert event: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("read event id: %w", err)
	}
	return id, nil
}

func (s *Store) ListSystemEvents(ctx context.Context, afterID int64, limit int) ([]Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, run_id, event_type, created_at, payload_json, reliable FROM events WHERE run_id IS NULL AND id > ? ORDER BY id LIMIT ?`, afterID, limit)
	if err != nil {
		return nil, fmt.Errorf("list system events: %w", err)
	}
	defer rows.Close()
	result := make([]Event, 0)
	for rows.Next() {
		var event Event
		var runID sql.NullString
		var created, payload string
		var reliable int
		if err := rows.Scan(&event.ID, &runID, &event.Type, &created, &payload, &reliable); err != nil {
			return nil, fmt.Errorf("scan system event: %w", err)
		}
		if event.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
			return nil, fmt.Errorf("parse system event time: %w", err)
		}
		if err := json.Unmarshal([]byte(payload), &event.Payload); err != nil {
			return nil, fmt.Errorf("decode system event: %w", err)
		}
		event.RunID = runID.String
		event.Reliable = reliable != 0
		result = append(result, event)
	}
	return result, rows.Err()
}

func (s *Store) LatestSystemSample(ctx context.Context) (time.Time, map[string]any, error) {
	var created, payload string
	if err := s.db.QueryRowContext(ctx, `SELECT created_at, payload_json FROM system_samples ORDER BY id DESC LIMIT 1`).Scan(&created, &payload); err != nil {
		return time.Time{}, nil, err
	}
	at, err := time.Parse(time.RFC3339Nano, created)
	if err != nil {
		return time.Time{}, nil, err
	}
	value := make(map[string]any)
	if err := json.Unmarshal([]byte(payload), &value); err != nil {
		return time.Time{}, nil, err
	}
	return at, value, nil
}

func (s *Store) ListMetrics(ctx context.Context, runID string, limit int) ([]MetricSample, error) {
	if limit <= 0 || limit > 100000 {
		limit = 10000
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, created_at, global_step, epoch, metrics_json FROM metric_samples WHERE run_id = ? ORDER BY global_step LIMIT ?`, runID, limit)
	if err != nil {
		return nil, fmt.Errorf("list metrics: %w", err)
	}
	defer rows.Close()
	result := make([]MetricSample, 0)
	for rows.Next() {
		var sample MetricSample
		var created, payload string
		if err := rows.Scan(&sample.ID, &created, &sample.GlobalStep, &sample.Epoch, &payload); err != nil {
			return nil, fmt.Errorf("scan metric: %w", err)
		}
		if sample.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
			return nil, fmt.Errorf("parse metric time: %w", err)
		}
		if err := json.Unmarshal([]byte(payload), &sample.Metrics); err != nil {
			return nil, fmt.Errorf("decode metric payload: %w", err)
		}
		result = append(result, sample)
	}
	return result, rows.Err()
}

func (s *Store) RecordArtifact(ctx context.Context, artifact Artifact) error {
	metadata, err := json.Marshal(artifact.Metadata)
	if err != nil {
		return fmt.Errorf("marshal artifact metadata: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO artifacts(run_id, kind, relative_path, created_at, metadata_json) VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id, relative_path) DO UPDATE SET kind = excluded.kind, created_at = excluded.created_at, metadata_json = excluded.metadata_json`, artifact.RunID, artifact.Kind, artifact.RelativePath, artifact.CreatedAt.UTC().Format(time.RFC3339Nano), string(metadata))
	if err != nil {
		return fmt.Errorf("insert artifact: %w", err)
	}
	return nil
}

func (s *Store) ListArtifacts(ctx context.Context, runID string) ([]Artifact, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, kind, relative_path, created_at, metadata_json FROM artifacts WHERE run_id = ? ORDER BY created_at DESC`, runID)
	if err != nil {
		return nil, fmt.Errorf("list artifacts: %w", err)
	}
	defer rows.Close()
	result := make([]Artifact, 0)
	for rows.Next() {
		var artifact Artifact
		var created, metadata string
		artifact.RunID = runID
		if err := rows.Scan(&artifact.ID, &artifact.Kind, &artifact.RelativePath, &created, &metadata); err != nil {
			return nil, fmt.Errorf("scan artifact: %w", err)
		}
		if artifact.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
			return nil, fmt.Errorf("parse artifact time: %w", err)
		}
		if err := json.Unmarshal([]byte(metadata), &artifact.Metadata); err != nil {
			return nil, fmt.Errorf("decode artifact metadata: %w", err)
		}
		result = append(result, artifact)
	}
	return result, rows.Err()
}

func (s *Store) RecordMetric(ctx context.Context, runID string, createdAt time.Time, step, epoch int, metrics map[string]float64) error {
	payload, err := json.Marshal(metrics)
	if err != nil {
		return fmt.Errorf("marshal metrics: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO metric_samples(run_id, created_at, global_step, epoch, metrics_json) VALUES (?, ?, ?, ?, ?)`, runID, createdAt.UTC().Format(time.RFC3339Nano), step, epoch, string(payload))
	if err != nil {
		return fmt.Errorf("insert metrics: %w", err)
	}
	return nil
}

func (s *Store) RecordSystemSample(ctx context.Context, createdAt time.Time, payload map[string]any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal system sample: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO system_samples(created_at, payload_json) VALUES (?, ?)`, createdAt.UTC().Format(time.RFC3339Nano), string(encoded))
	if err != nil {
		return fmt.Errorf("insert system sample: %w", err)
	}
	return nil
}

func (s *Store) ListEvents(ctx context.Context, runID string, afterID int64, limit int) ([]Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, run_id, event_type, created_at, payload_json, reliable FROM events WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?`, runID, afterID, limit)
	if err != nil {
		return nil, fmt.Errorf("list events: %w", err)
	}
	defer rows.Close()
	result := make([]Event, 0)
	for rows.Next() {
		var event Event
		var runID sql.NullString
		var created string
		var payload string
		var reliable int
		if err := rows.Scan(&event.ID, &runID, &event.Type, &created, &payload, &reliable); err != nil {
			return nil, fmt.Errorf("scan event: %w", err)
		}
		if event.CreatedAt, err = time.Parse(time.RFC3339Nano, created); err != nil {
			return nil, fmt.Errorf("parse event time: %w", err)
		}
		if err := json.Unmarshal([]byte(payload), &event.Payload); err != nil {
			return nil, fmt.Errorf("decode event payload: %w", err)
		}
		event.RunID = runID.String
		event.Reliable = reliable != 0
		result = append(result, event)
	}
	return result, rows.Err()
}

func scanRun(scanner interface{ Scan(...any) error }) (Run, error) {
	var run Run
	var submitted, updated string
	var started, finished sql.NullString
	var config, devices string
	if err := scanner.Scan(
		&run.ID, &run.ParentRunID, &run.TemplateID, &run.Status, &submitted,
		&updated, &started, &finished, &config, &run.ConfigPath,
		&run.ManifestPath, &run.ControlPath, &run.OutputDir, &run.CheckpointPath,
		&run.Launcher, &devices, &run.ProcessID, &run.LastStep, &run.LastEpoch,
		&run.ErrorMessage, &run.WorkflowID, &run.WorkflowSnapshotHash,
		&run.CompilerVersion,
	); err != nil {
		return Run{}, err
	}
	var err error
	if run.SubmittedAt, err = time.Parse(time.RFC3339Nano, submitted); err != nil {
		return Run{}, err
	}
	if run.UpdatedAt, err = time.Parse(time.RFC3339Nano, updated); err != nil {
		return Run{}, err
	}
	if started.Valid {
		value, parseErr := time.Parse(time.RFC3339Nano, started.String)
		if parseErr != nil {
			return Run{}, parseErr
		}
		run.StartedAt = &value
	}
	if finished.Valid {
		value, parseErr := time.Parse(time.RFC3339Nano, finished.String)
		if parseErr != nil {
			return Run{}, parseErr
		}
		run.FinishedAt = &value
	}
	if err := json.Unmarshal([]byte(config), &run.ConfigJSON); err != nil {
		return Run{}, err
	}
	if err := json.Unmarshal([]byte(devices), &run.Devices); err != nil {
		return Run{}, err
	}
	return run, nil
}

func nullableString(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func SortRuns(runs []Run) {
	sort.Slice(runs, func(i, j int) bool {
		return runs[i].SubmittedAt.After(runs[j].SubmittedAt)
	})
}
